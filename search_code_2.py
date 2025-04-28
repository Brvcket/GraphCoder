import os
import copy
import queue
import time
import numpy as np
import torch
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
from utils.utils import (CONSTANTS, dump_jsonl, json_to_graph,
                         load_jsonl, make_needed_dir, CodexTokenizer)
from utils.metrics import hit
from Levenshtein import distance as levenshtein_distance
from functools import partial
from build_query_graph import build_query_subgraph
import argparse

from code_embedder import CodeEmbedder
from similarity_functions import cosine_similarity, dot_product, l2_distance, l1_distance
from tqdm import tqdm


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
            cls._embedder = CodeEmbedder()
        return cls._embedder

    @classmethod
    def set_similarity_metric(cls, metric="cosine"):
        sim_map = {
            'cosine': cosine_similarity,
            'dot': dot_product,
            'l2': l2_distance,
            'l1': l1_distance
        }
        cls._embedder.sim_fn = sim_map.get(metric, cosine_similarity)

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1, set2 = set(list1), set(list2)
        return float(len(set1 & set2) / len(set1 | set2))

    @staticmethod
    def text_edit_similarity(str1, str2):
        return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

    @classmethod
    def subgraph_edit_similarity(cls, query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        # Embedding-based subgraph similarity (node + BFS + edge)
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        embedder = cls._embedder

        # node 0 similarity
        query_code = "".join(query_graph.nodes[query_root]['sourceLines'])
        graph_code = "".join(graph.nodes[root]['sourceLines'])
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

        # embedding setup
        SimilarityScore.set_embedder(embedder_choice)
        SimilarityScore.set_similarity_metric(similarity_metric)
        # token setup
        self.tokenizer = CodexTokenizer()
        self.embedder_choice=embedder_choice

    @staticmethod
    def _is_context_after_hole(query_case, repo_case):
        h = "/".join(query_case['metadata']['fpath_tuple'])
        r = "/".join(repo_case['fpath_tuple'])
        if h != r:
            return False
        return repo_case['max_line_no'] >= max(query_case['metadata']['forward_context_line_list'])

    # Phase1: embedding-based text
    def _text_search(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        embedder = SimilarityScore._embedder
        sim_fn = embedder.sim_fn
        # query emb
        if 'query_embedding' in query_case:
            q_emb = np.array(query_case['query_embedding'], dtype=np.float32)
        else:
            qdata = "".join(query_case.get('query_forward_context',''))
            q_emb = embedder.get_embedding(qdata)
        # repo emb
        if 'embedding' in repo_case:
            g_emb = np.array(repo_case['embedding'], dtype=np.float32)
        else:
            rdata = repo_case.get('key_forward_context','')
            g_emb = embedder.get_embedding(rdata)
        return repo_case, sim_fn(q_emb, g_emb)

    # Phase1: token-based text
    def _text_token_search(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        q_tokens = query_case.get('query_forward_encoding', [])
        r_tokens = repo_case.get('key_forward_encoding', [])
        sim = SimilarityScore.text_jaccard_similarity(q_tokens, r_tokens)
        return repo_case, sim

    # Phase2: embedding-based graph
    def _graph_search(self, query_case, repo_case):
        qg = json_to_graph(query_case['query_forward_graph'])
        rg = json_to_graph(repo_case['key_forward_graph'])
        if len(rg.nodes) == 0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        return repo_case, SimilarityScore.subgraph_edit_similarity(qg, rg, gamma=self.gamma)

    # Phase2: token-based graph
    def _graph_token_search(self, query_case, repo_case):
        qg = json_to_graph(query_case['query_forward_graph'])
        rg = json_to_graph(repo_case['key_forward_graph'])
        if len(rg.nodes) == 0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        # root jaccard on tokenized source
        root_q = max(qg.nodes)
        root_r = max(rg.nodes)
        qtoks = self.tokenizer.tokenize("".join(qg.nodes[root_q]['sourceLines']))
        rtoks = self.tokenizer.tokenize("".join(rg.nodes[root_r]['sourceLines']))
        node_sim = SimilarityScore.text_jaccard_similarity(qtoks, rtoks)
        # BFS matching
        node_match = {root_q: (root_r, 0)}
        match_queue = queue.Queue()
        match_queue.put((root_q, root_r, 0))
        visited_q, visited_r = {root_q}, {root_r}
        all_r = set(rg.nodes)
        while not match_queue.empty():
            v, u, hop = match_queue.get()
            q_neighbors = (set(qg.neighbors(v)) | set(qg.predecessors(v))) - visited_q
            r_candidates = all_r - visited_r
            sims = []
            for qn in q_neighbors:
                qn_toks = self.tokenizer.tokenize("".join(qg.nodes[qn]['sourceLines']))
                for rn in r_candidates:
                    rn_toks = self.tokenizer.tokenize("".join(rg.nodes[rn]['sourceLines']))
                    sims.append((SimilarityScore.text_jaccard_similarity(qn_toks, rn_toks), qn, rn))
            sims.sort(key=lambda x: -x[0])
            for sim_val, qn, rn in sims:
                if qn not in visited_q and rn not in visited_r:
                    node_sim += (self.gamma ** (hop + 1)) * sim_val
                    match_queue.put((qn, rn, hop + 1))
                    node_match[qn] = (rn, hop + 1)
                    visited_q.add(qn)
                    visited_r.add(rn)
                    break
        # edge similarity
        edge_sim = 0
        for v_q, u_q, key in qg.edges:
            m_v = node_match.get(v_q)
            m_u = node_match.get(u_q)
            if m_v and m_u and rg.has_edge(m_v[0], m_u[0], key=key):
                edge_sim += (self.gamma ** m_v[1])
        return repo_case, node_sim + edge_sim

    # One-phase search
    def _find_top_k_context_one_phase(self, query_case):
        repo = query_case['metadata']['task_id'].split('/')[0]
        emb_file = os.path.join(CONSTANTS.graph_database_save_dir, 
                                 f"{repo}.{self.embedder_choice}.with_emb.jsonl")
        db_file = emb_file if os.path.exists(emb_file) else \
                  os.path.join(CONSTANTS.graph_database_save_dir, f"{repo}.jsonl")
        db = load_jsonl(db_file)
        # choose compute fn
        if self.mode == 'coarse':
            compute = self._text_search
        else:
            compute = self._graph_search
        sims = []
        for rc, score in ThreadPoolExecutor(max_workers=32).map(partial(compute, query_case), db):
            if score >= self.remove_threshold:
                sims.append((rc, score))
        topk = sorted(sims, key=lambda x: x[1])[-self.max_top_k:]
        out = copy.deepcopy(query_case)
        out['top_k_context'] = [(r['val'], r['statement'], r['key_forward_context'], r['fpath_tuple'], s)
                                 for r, s in topk]
        return out

    # Two-phase search
    def _find_top_k_context_two_phase(self, query_case):
        repo = query_case['metadata']['task_id'].split('/')[0]
        emb_file = os.path.join(CONSTANTS.graph_database_save_dir,
                                 f"{repo}.{self.embedder_choice}.with_emb.jsonl")
        db_file = emb_file if os.path.exists(emb_file) else \
                  os.path.join(CONSTANTS.graph_database_save_dir, f"{repo}.jsonl")
        db = load_jsonl(db_file)

        # pick functions based on mode
        first, second = self.mode.split('_')
        p1_fn = self._text_token_search if first == 'token' else self._text_search
        p2_fn = self._graph_token_search if second == 'token' else self._graph_search

        # phase1
        t0 = time.time()
        p1 = list(ThreadPoolExecutor(max_workers=32)
                  .map(partial(p1_fn, query_case), db))
        top1 = sorted(p1, key=lambda x: x[1])[-self.max_top_k:]
        t1 = time.time()

        # phase2
        candidates = [case for case, _ in top1]
        p2 = list(ThreadPoolExecutor(max_workers=32)
                  .map(partial(p2_fn, query_case), candidates))
        filtered = [(rc['val'], rc['statement'], rc['key_forward_context'], rc['fpath_tuple'], sim)
                    for rc, sim in p2 if sim >= self.remove_threshold]
        out = copy.deepcopy(query_case)
        out['top_k_context'] = sorted(filtered, key=lambda x: x[-1])[-self.max_top_k:]
        out['text_runtime'], out['graph_runtime'] = t1 - t0, time.time() - t1
        # print(f"case {query_case['metadata']['task_id']} finished")
        return out

    def run(self):
        results = []
        for qc in tqdm(self.query_cases):
            if self.mode in ('coarse', 'fine'):
                results.append(self._find_top_k_context_one_phase(qc))
            else:
                results.append(self._find_top_k_context_two_phase(qc))
        dump_jsonl(results, self.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_cases', default="api_level", type=str)
    parser.add_argument('--mode',
                        choices=['coarse', 'fine',
                                 'token_token', 'token_embed',
                                 'embed_token', 'embed_embed'],
                        default='embed_embed')
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--embedder',
                        choices=['codet5', 'codebert', 'graphcodebert', 'starcoder'],
                        default='codet5')
    parser.add_argument('--metric',
                        choices=['cosine', 'dot', 'l2', 'l1'],
                        default='cosine')
    args = parser.parse_args()

    build_query_subgraph(f"{args.query_cases}.test.jsonl")
    cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir,
                                    f"{args.query_cases}.test.jsonl"))

    save_path = os.path.join(f"./search_results/{args.query_cases}.{args.mode}.{int(args.gamma*100)}.jsonl")
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
