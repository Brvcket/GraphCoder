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

from Levenshtein import distance as levenshtein_distance
from functools import partial
from build_query_graph import build_query_subgraph
import argparse

from code_embedder import CodeEmbedder
from similarity_functions import cosine_similarity, dot_product, l2_distance, l1_distance


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
                    node_sim += (gamma ** (hop + 1)) * sim_val
                    match_queue.put((qn, gn, hop + 1))
                    node_match[qn] = (gn, hop + 1)
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
        self.embedder_choice = embedder_choice
        # setup embedder & metric
        SimilarityScore.set_embedder(embedder_choice)
        SimilarityScore.set_similarity_metric(similarity_metric)

    @staticmethod
    def _is_context_after_hole(query_case, repo_case):
        h = "/".join(query_case['metadata']['fpath_tuple'])
        r = "/".join(repo_case['fpath_tuple'])
        if h != r: return False
        return repo_case['max_line_no'] >= max(query_case['metadata']['forward_context_line_list'])

    def _text_search(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        embedder = SimilarityScore._embedder
        sim_fn = embedder.sim_fn

        # query embedding
        if 'query_embedding' in query_case:
            q_emb = np.array(query_case['query_embedding'], dtype=np.float32)
        else:
            qdata = "".join(query_case.get('query_forward_context', ''))
            q_emb = embedder.get_embedding(qdata)

        # repo embedding (precomputed?) or compute on the fly
        if 'embedding' in repo_case:
            g_emb = np.array(repo_case['embedding'], dtype=np.float32)
        else:
            rdata = repo_case.get('key_forward_context', '')
            g_emb = embedder.get_embedding(rdata)

        return repo_case, sim_fn(q_emb, g_emb)

    def _graph_search(self, query_case, repo_case):
        qg = json_to_graph(query_case['query_forward_graph'])
        rg = json_to_graph(repo_case['key_forward_graph'])
        if len(rg.nodes) == 0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        return repo_case, SimilarityScore.subgraph_edit_similarity(qg, rg, gamma=self.gamma)

    def _find_top_k_context_one_phase(self, query_case):
        repo = query_case['metadata']['task_id'].split('/')[0]
        emb_file = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo}.{self.embedder_choice}.with_emb.jsonl")
        db_file = emb_file if os.path.exists(emb_file) else os.path.join(CONSTANTS.graph_database_save_dir,
                                                                         f"{repo}.jsonl")
        repo_cases = load_jsonl(db_file)

        compute = self._text_search if self.mode == 'coarse' else self._graph_search
        sims = []
        for rc, score in ThreadPoolExecutor(max_workers=32).map(partial(compute, query_case), repo_cases):
            if score >= self.remove_threshold:
                sims.append((rc, score))

        topk = sorted(sims, key=lambda x: x[1])[-self.max_top_k:]
        out = copy.deepcopy(query_case)
        out['top_k_context'] = [(r['val'], r['statement'], r['key_forward_context'], r['fpath_tuple'], s) for r, s in
                                topk]
        return out

    def _find_top_k_context_two_phase(self, query_case):
        repo = query_case['metadata']['task_id'].split('/')[0]
        emb_file = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo}.{self.embedder_choice}.with_emb.jsonl")
        db_file = emb_file if os.path.exists(emb_file) else os.path.join(CONSTANTS.graph_database_save_dir,
                                                                         f"{repo}.jsonl")
        db = load_jsonl(db_file)
        t0 = time.time()
        p1 = list(ThreadPoolExecutor(max_workers=32).map(
            partial(self._text_search, query_case), db))
        top1 = sorted(p1, key=lambda x: x[1])[-self.max_top_k:]
        t1 = time.time()
        # phase2 graph
        candidates = [case for case, _ in top1]
        p2 = list(ThreadPoolExecutor(max_workers=32).map(
            partial(self._graph_search, query_case), candidates))
        filt = [(rc['val'], rc['statement'], rc['key_forward_context'], rc['fpath_tuple'], sim)
                for rc, sim in p2 if sim >= self.remove_threshold]
        result = copy.deepcopy(query_case)
        result['top_k_context'] = sorted(filt, key=lambda x: x[-1])[-self.max_top_k:]
        t2 = time.time()
        result['text_runtime'], result['graph_runtime'] = t1 - t0, t2 - t1
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


def precompute_repo_embeddings(repo_name: str, embedder, force: bool = False):
    """
    Reads <graph_database>/<repo_name>.jsonl,
    for each case uses key_forward_encoding (if non-empty) else key_forward_context
    to compute case["embedding"], then writes <repo_name>.with_emb.jsonl.
    """
    in_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl")
    out_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.{embedder}.with_emb.jsonl")

    if os.path.exists(out_path) and not force:
        print(f"[precompute] {out_path} exists; skipping (use --force).")
        return

    cases = load_jsonl(in_path)
    for case in cases:
        tokens = case.get("key_forward_encoding") or []
        data = tokens if tokens else case.get("key_forward_context", "")
        case["embedding"] = embedder.get_embedding(data).tolist()

    dump_jsonl(cases, out_path)
    print(f"[precompute] saved embeddings to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_cases', default="api_level", type=str)
    parser.add_argument('--mode', default='coarse2fine', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--embedder', choices=['codet5', 'codebert', 'graphcodebert', 'starcoder'], default='codet5')
    parser.add_argument('--metric', choices=['cosine', 'dot', 'l2', 'l1'], default='cosine')
    args = parser.parse_args()

    build_query_subgraph(f"{args.query_cases}.test.jsonl")
    cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl"))
    save_path = os.path.join(f"./search_results/{args.query_cases}.{args.mode}.{int(args.gamma * 100)}.jsonl")
    make_needed_dir(save_path)
    worker = CodeSearchWorker(
        cases, save_path, args.mode, gamma=args.gamma,
        embedder_choice=args.embedder,
        similarity_metric=args.metric
    )
    worker.run()
    res = load_jsonl(worker.output_path)
    h1, h5, h10 = hit(res, hits=[1, 5, 10])
    print(f"Hits@1: {h1:.4f}, Hits@5: {h5:.4f}, Hits@10: {h10:.4f}")
