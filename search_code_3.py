import os
import copy
import queue
import time
import numpy as np
import torch
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
from utils.utils import CONSTANTS, dump_jsonl, json_to_graph, load_jsonl, make_needed_dir
from utils.metrics import hit
from Levenshtein import distance as levenshtein_distance

# Optional retrieval backends
try:
    import faiss
except ImportError:
    faiss = None
try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None
try:
    import hnswlib
except ImportError:
    hnswlib = None

# --- Embedders ---
class CodeEmbedder:
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.max_length = max_length

    def get_embedding(self, code_text):
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        else:
            emb = outputs[0].mean(dim=1).cpu().numpy()[0]
        return emb

class DummyEmbedder:
    def get_embedding(self, code_text):
        return np.zeros(128)

# --- Similarity Metrics ---
def cosine_similarity(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return np.dot(a, b) / (norm_a * norm_b + 1e-8)

def dot_product(a, b):
    return float(np.dot(a, b))

def l2_distance(a, b):
    return -float(np.linalg.norm(a - b))

def l1_distance(a, b):
    return -float(np.linalg.norm(a - b, ord=1))

class SimilarityScore:
    @staticmethod
    def get_embedder(choice="codet5"):
        if choice == "codet5":
            return CodeEmbedder("Salesforce/codet5p-110m-embedding", max_length=512)
        elif choice == "codebert":
            return CodeEmbedder("microsoft/codebert-base", max_length=512)
        elif choice == "graphcodebert":
            return CodeEmbedder("microsoft/graphcodebert-base", max_length=512)
        elif choice == "starcoder":
            return CodeEmbedder("bigcode/starcoder", max_length=2048)
        else:
            return DummyEmbedder()

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1, set2 = set(list1), set(list2)
        return float(len(set1 & set2) / (len(set1 | set2) + 1e-8))

    @staticmethod
    def text_edit_similarity(str1: str, str2: str):
        return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2), 1)

    @staticmethod
    def subgraph_edit_similarity(
        query_graph: nx.MultiDiGraph,
        graph: nx.MultiDiGraph,
        gamma=0.1,
        sim_fn=cosine_similarity
    ):
        # node-level similarity via embeddings
        qroot = max(query_graph.nodes)
        grot = max(graph.nodes)
        embedder = SimilarityScore.get_embedder()
        qcode = "".join(query_graph.nodes[qroot]['sourceLines'])
        gcode = "".join(graph.nodes[grot]['sourceLines'])
        qemb, gemb = embedder.get_embedding(qcode), embedder.get_embedding(gcode)
        node_sim = sim_fn(qemb, gemb)

        # BFS for subgraph matching
        node_match = {qroot: (grot, 0)}
        visited_q, visited_g = {qroot}, {grot}
        queue_pairs = [(qroot, grot, 0)]
        all_q_nodes = set(query_graph.nodes)
        all_g_nodes = set(graph.nodes)

        while queue_pairs:
            v, u, hop = queue_pairs.pop(0)
            q_nbrs = [n for n in (set(query_graph.successors(v)) | set(query_graph.predecessors(v))) if n not in visited_q]
            g_cands = [n for n in all_g_nodes if n not in visited_g]
            sims = []
            for qn in q_nbrs:
                qc = "".join(query_graph.nodes[qn]['sourceLines'])
                qemb = embedder.get_embedding(qc)
                for gn in g_cands:
                    gc = "".join(graph.nodes[gn]['sourceLines'])
                    gemb = embedder.get_embedding(gc)
                    sims.append((sim_fn(qemb, gemb), qn, gn))
            sims.sort(reverse=True, key=lambda x: x[0])
            for sim, qn, gn in sims[:1]:
                node_sim += (gamma ** (hop + 1)) * sim
                node_match[qn] = (gn, hop + 1)
                visited_q.add(qn)
                visited_g.add(gn)
                queue_pairs.append((qn, gn, hop + 1))

        # edge similarity
        edge_sim = 0
        for v, u, k in query_graph.edges:
            match_v = node_match.get(v)
            match_u = node_match.get(u)
            if match_v and match_u and graph.has_edge(match_v[0], match_u[0], key=k):
                edge_sim += gamma ** match_v[1]
        return node_sim + edge_sim

class CodeSearchWorker:
    def __init__(
        self,
        query_cases,
        output_path,
        mode='coarse2fine',
        gamma=0.1,
        max_top_k=CONSTANTS.max_search_top_k,
        remove_threshold=0,
        embedder_choice='codet5',
        retrieval_backend='faiss',
        similarity_metric='cosine',
        two_phase=False,
        graph_kernel=False
    ):
        self.query_cases = query_cases
        self.output_path = output_path
        self.mode = mode
        self.gamma = gamma
        self.max_top_k = max_top_k
        self.remove_threshold = remove_threshold
        self.two_phase = two_phase
        self.graph_kernel = graph_kernel

        # Init embedder
        self.embedder = SimilarityScore.get_embedder(embedder_choice)
        # similarity function
        if similarity_metric == 'dot':
            self.sim_fn = dot_product
        elif similarity_metric == 'l2':
            self.sim_fn = l2_distance
        elif similarity_metric == 'l1':
            self.sim_fn = l1_distance
        else:
            self.sim_fn = cosine_similarity

        # retrieval backend setup
        self.backend = retrieval_backend
        if self.backend == 'faiss' and faiss:
            self.index = None
        elif self.backend == 'annoy' and AnnoyIndex:
            self.annoy_index = None
        elif self.backend == 'hnsw' and hnswlib:
            self.hnsw_index = None

    def _is_context_after_hole(self, query_case, repo_case):
        hole = "/".join(query_case['metadata']['fpath_tuple'])
        repo = "/".join(repo_case['fpath_tuple'])
        return hole == repo and repo_case['max_line_no'] >= max(query_case['metadata']['forward_context_line_list'])

    def _text_search(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case): return repo_case, 0
        sim = SimilarityScore.text_jaccard_similarity(
            query_case['query_forward_encoding'], repo_case['key_forward_encoding']
        )
        return repo_case, sim

    def _graph_search(self, query_case, repo_case):
        qg = json_to_graph(query_case['query_forward_graph'])
        rg = json_to_graph(repo_case['key_forward_graph'])
        if len(rg.nodes) == 0 or self._is_context_after_hole(query_case, repo_case): return repo_case, 0
        if self.graph_kernel:
            return repo_case, 0  # graph kernel stub
        return repo_case, SimilarityScore.subgraph_edit_similarity(qg, rg, gamma=self.gamma, sim_fn=self.sim_fn)

    def run(self):
        results = []
        for case in self.query_cases:
            res = self._two_phase(case) if self.two_phase else self._one_phase(case)
            results.append(res)
        dump_jsonl(results, self.output_path)

    def _one_phase(self, query_case):
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        repo_cases = load_jsonl(os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl"))
        sims = []
        func = self._text_search if self.mode.startswith('coarse') else self._graph_search
        with ThreadPoolExecutor(max_workers=32) as exc:
            for repo_case, sim in exc.map(func, repo_cases):
                if sim >= self.remove_threshold:
                    sims.append((repo_case, sim))
        top = sorted(sims, key=lambda x: x[1])[-self.max_top_k:]
        out = copy.deepcopy(query_case)
        out['top_k_context'] = [(r['val'], r['statement'], r['key_forward_context'], r['fpath_tuple'], s) for r,s in top]
        return out

    def _two_phase(self, query_case):
        phase1 = self._one_phase(query_case)
        candidates = [{'val':v,'statement':st,'key_forward_context':ctx,'fpath_tuple':fp} for v,st,ctx,fp,_ in phase1['top_k_context']]
        sims = []
        with ThreadPoolExecutor(max_workers=32) as exc:
            for repo_case, sim in exc.map(self._graph_search, candidates):
                if sim >= self.remove_threshold:
                    sims.append((repo_case, sim))
        top2 = sorted(sims, key=lambda x: x[1])[-self.max_top_k:]
        phase1['top_k_context'] = [(r['val'], r['statement'], r['key_forward_context'], r['fpath_tuple'], s) for r,s in top2]
        return phase1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Graph-based Code Search")
    parser.add_argument('--query_cases', default="api_level", type=str)
    parser.add_argument('--mode', choices=['coarse','fine','coarse2fine'], default='coarse2fine')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--embedder', choices=['codet5','codebert','graphcodebert','starcoder','dummy'], default='codet5')
    parser.add_argument('--backend', choices=['faiss','annoy','hnsw','pinecone'], default='faiss')
    parser.add_argument('--metric', choices=['cosine','dot','l2','l1'], default='cosine')
    parser.add_argument('--two_phase', action='store_true')
    parser.add_argument('--graph_kernel', action='store_true')
    args = parser.parse_args()

    build_query_subgraph(f"{args.query_cases}.test.jsonl")
    cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl"))
    save_path = os.path.join(f"./search_results/{args.query_cases}.{args.mode}.{int(args.gamma*100)}.jsonl")
    make_needed_dir(save_path)
    worker = CodeSearchWorker(
        cases, save_path,
        mode=args.mode,
        gamma=args.gamma,
        embedder_choice=args.embedder,
        retrieval_backend=args.backend,
        similarity_metric=args.metric,
        two_phase=args.two_phase,
        graph_kernel=args.graph_kernel
    )
    worker.run()
    results = load_jsonl(save_path)
    h1, h5, h10 = hit(results, hits=[1,5,10])
    print(f"Hits@1: {h1:.4f}, Hits@5: {h5:.4f}, Hits@10: {h10:.4f}")

from Levenshtein import distance as levenshtein_distance
from functools import partial
from build_query_graph import build_query_subgraph
from utils.metrics import hit
import argparse

# --- Similarity Functions ---
def cosine_similarity(a, b):
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    return 0.0 if norm1 == 0 or norm2 == 0 else np.dot(a, b) / (norm1 * norm2)

def dot_product(a, b):
    return float(np.dot(a, b))

def l2_distance(a, b):
    return -float(np.linalg.norm(a - b))

def l1_distance(a, b):
    return -float(np.linalg.norm(a - b, ord=1))

# --- Embedders ---
class CodeEmbedder:
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.max_length = max_length
        # default sim_fn
        self.sim_fn = cosine_similarity

    def get_embedding(self, code_text):
        inputs = self.tokenizer(code_text, return_tensors="pt", truncation=True,
                                max_length=self.max_length, padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.cpu().numpy()[0]

# --- Similarity & Embedding Manager ---
class SimilarityScore:
    _embedder = None

    @classmethod
    def set_embedder(cls, choice="codet5"):
        if choice == "codet5":
            cls._embedder = CodeEmbedder("Salesforce/codet5p-110m-embedding", max_length=512)
        elif choice == "codebert":
            cls._embedder = CodeEmbedder("microsoft/codebert-base", max_length=512)
        elif choice == "graphcodebert":
            cls._embedder = CodeEmbedder("microsoft/graphcodebert-base", max_length=512)
        elif choice == "starcoder":
            cls._embedder = CodeEmbedder("bigcode/starcoder", max_length=2048)
        else:
            cls._embedder = CodeEmbedder()  # fallback
        return cls._embedder

    @classmethod
    def set_similarity_metric(cls, metric="cosine"):
        sim_map = {
            'cosine': cosine_similarity,
            'dot': dot_product,
            'l2': l2_distance,
            'l1': l1_distance
        }
        sim_fn = sim_map.get(metric, cosine_similarity)
        # assign to embedder instance
        cls._embedder.sim_fn = sim_fn

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1, set2 = set(list1), set(list2)
        return float(len(set1.intersection(set2)) / len(set1.union(set2)))

    @staticmethod
    def text_edit_similarity(str1, str2):
        return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

    @classmethod
    def subgraph_edit_similarity(cls, query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        embedder = cls._embedder
        # get codes
        query_code = "".join(query_graph.nodes[query_root]['sourceLines'])
        graph_code = "".join(graph.nodes[root]['sourceLines'])
        # compute node similarity via chosen sim_fn
        q_emb = embedder.get_embedding(query_code)
        g_emb = embedder.get_embedding(graph_code)
        node_sim = embedder.sim_fn(q_emb, g_emb)

        # BFS matching
        node_match = {query_root: (root, 0)}
        match_queue = queue.Queue()
        match_queue.put((query_root, root, 0))
        visited_q, visited_g = {query_root}, {root}
        all_g_nodes = set(graph.nodes)

        while not match_queue.empty():
            v, u, hop = match_queue.get()
            q_neighbors = (set(query_graph.neighbors(v)) | set(query_graph.predecessors(v))) - visited_q
            g_candidates = all_g_nodes - visited_g
            sims = []
            for qn in q_neighbors:
                q_code = "".join(query_graph.nodes[qn]['sourceLines'])
                q_emb = embedder.get_embedding(q_code)
                for gn in g_candidates:
                    g_code = "".join(graph.nodes[gn]['sourceLines'])
                    g_emb = embedder.get_embedding(g_code)
                    sims.append((embedder.sim_fn(q_emb, g_emb), qn, gn))
            sims.sort(key=lambda x: -x[0])
            for sim_val, qn, gn in sims:
                if qn not in visited_q and gn not in visited_g:
                    node_sim += (gamma ** (hop+1)) * sim_val
                    match_queue.put((qn, gn, hop+1))
                    node_match[qn] = (gn, hop+1)
                    visited_q.add(qn)
                    visited_g.add(gn)
                    break

        # edge similarity
        edge_sim = 0
        for v_q, u_q, k in query_graph.edges:
            m_v = node_match.get(v_q)
            m_u = node_match.get(u_q)
            if m_v and m_u and graph.has_edge(m_v[0], m_u[0], key=k):
                edge_sim += (gamma ** m_v[1])
        return node_sim + edge_sim

# --- Code Search Worker ---
class CodeSearchWorker:
    def __init__(self, query_cases, output_path, mode, gamma=None,
                 max_top_k=CONSTANTS.max_search_top_k, remove_threshold=0,
                 embedder_choice='codet5', similarity_metric='cosine'):
        self.query_cases = query_cases
        self.output_path = output_path
        self.mode = mode
        self.gamma = gamma
        self.max_top_k = max_top_k
        self.remove_threshold = remove_threshold
        # setup embedder & metric
        SimilarityScore.set_embedder(embedder_choice)
        SimilarityScore.set_similarity_metric(similarity_metric)

    @staticmethod
    def _is_context_after_hole(query_case, repo_case):
        h = "/".join(query_case['metadata']['fpath_tuple'])
        r = "/".join(repo_case['fpath_tuple'])
        if h != r: return False
        return repo_case['max_line_no'] >= max(query_case['metadata']['forward_context_line_list'])

    def _text_jaccard_similarity_wrapper(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0
        sim = SimilarityScore.text_jaccard_similarity(
            query_case['query_forward_encoding'], repo_case['key_forward_encoding'])
        return repo_case, sim

    def _graph_node_prior_similarity_wrapper(self, query_case, repo_case):
        qg = json_to_graph(query_case['query_forward_graph'])
        rg = json_to_graph(repo_case['key_forward_graph'])
        if len(rg.nodes)==0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0
        sim = SimilarityScore.subgraph_edit_similarity(qg, rg, gamma=self.gamma)
        return repo_case, sim

    def _find_top_k_context_one_phase(self, query_case):
        start = time.time()
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        search_res = copy.deepcopy(query_case)
        db = load_jsonl(os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl"))
        compute = self._text_jaccard_similarity_wrapper if self.mode=='coarse' else self._graph_node_prior_similarity_wrapper
        results = list(ThreadPoolExecutor(max_workers=32).map(partial(compute, query_case), db))
        filt = [(rc['val'], rc['statement'], rc['key_forward_context'], rc['fpath_tuple'], sim)
                for rc, sim in results if sim>=self.remove_threshold]
        search_res['top_k_context'] = sorted(filt, key=lambda x:x[-1])[-self.max_top_k:]
        end = time.time()
        search_res['text_runtime'] = end-start if self.mode=='coarse' else 0
        search_res['graph_runtime'] = 0 if self.mode=='coarse' else end-start
        print(f"case {query_case['metadata']['task_id']} finished")
        return search_res

    def _find_top_k_context_two_phase(self, query_case):
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        db = load_jsonl(os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl"))
        # phase1 text
        t0 = time.time()
        p1 = list(ThreadPoolExecutor(max_workers=32).map(
            partial(self._text_jaccard_similarity_wrapper, query_case), db))
        top1 = sorted(p1, key=lambda x:x[1])[-self.max_top_k:]
        t1 = time.time()
        # phase2 graph
        candidates = [case for case, _ in top1]
        p2 = list(ThreadPoolExecutor(max_workers=32).map(
            partial(self._graph_node_prior_similarity_wrapper, query_case), candidates))
        filt = [(rc['val'], rc['statement'], rc['key_forward_context'], rc['fpath_tuple'], sim)
                for rc, sim in p2 if sim>=self.remove_threshold]
        result = copy.deepcopy(query_case)
        result['top_k_context'] = sorted(filt, key=lambda x:x[-1])[-self.max_top_k:]
        t2 = time.time()
        result['text_runtime'], result['graph_runtime'] = t1-t0, t2-t1
        print(f"case {query_case['metadata']['task_id']} finished")
        return result

    def run(self):
        out = []
        for qc in self.query_cases:
            if self.mode in ('coarse', 'fine'):
                out.append(self._find_top_k_context_one_phase(qc))
            else:
                out.append(self._find_top_k_context_two_phase(qc))
        dump_jsonl(out, self.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_cases', default="api_level", type=str)
    parser.add_argument('--mode', default='coarse2fine', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--embedder', choices=['codet5','codebert','graphcodebert','starcoder'], default='codet5')
    parser.add_argument('--metric', choices=['cosine','dot','l2','l1'], default='cosine')
    args = parser.parse_args()

    build_query_subgraph(f"{args.query_cases}.test.jsonl")
    cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl"))
    save_path = os.path.join(f"./search_results/{args.query_cases}.{args.mode}.{int(args.gamma*100)}.jsonl")
    make_needed_dir(save_path)
    worker = CodeSearchWorker(
        cases, save_path, args.mode, gamma=args.gamma,
        embedder_choice=args.embedder,
        similarity_metric=args.metric
    )
    worker.run()
    res = load_jsonl(worker.output_path)
    h1, h5, h10 = hit(res, hits=[1,5,10])
    print(f"Hits@1: {h1:.4f}, Hits@5: {h5:.4f}, Hits@10: {h10:.4f}")
