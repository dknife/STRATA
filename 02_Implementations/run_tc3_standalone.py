"""
Standalone benchmark for TC3 D-STORM-GraphBLAS.

Must run in a separate process from BC5 GB-bfs to avoid
GraphBLAS initialization conflict (GrB_init called twice).

Usage: python run_tc3_standalone.py
"""

import time
import json
import sys
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.loader import load_graph, make_undirected

RUNS = 3
FACEBOOK_PATH = os.path.join(ROOT, '..', '04_Datasets',
                              'real-world', 'facebook_combined.txt')


def build_graphs():
    graphs = []

    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=True)
    A = make_undirected(A)
    graphs.append(('Facebook', A, {'n': A.shape[0], 'm': A.nnz, 'type': 'social', 'diam': 8}))

    G = nx.barabasi_albert_graph(2000, 5, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('BA-2000', A, {'n': 2000, 'm': A.nnz, 'type': 'BA', 'diam': nx.diameter(G)}))

    G = nx.watts_strogatz_graph(2000, 6, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('WS-2000', A, {'n': 2000, 'm': A.nnz, 'type': 'WS', 'diam': nx.diameter(G)}))

    G = nx.erdos_renyi_graph(2000, 0.005, seed=42)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('ER-2000', A, {'n': A.shape[0], 'm': A.nnz, 'type': 'ER', 'diam': nx.diameter(G)}))

    G = nx.grid_2d_graph(45, 45)
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('Grid-45x45', A, {'n': A.shape[0], 'm': A.nnz, 'type': 'Grid', 'diam': nx.diameter(G)}))

    G = nx.powerlaw_cluster_graph(2000, 5, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('PLC-2000', A, {'n': 2000, 'm': A.nnz, 'type': 'PLC', 'diam': nx.diameter(G)}))

    return graphs


def main():
    from TC3_DSTORM_GraphBLAS.apsp import run_apsp
    from scipy.sparse.csgraph import shortest_path

    print("Building graphs...")
    graphs = build_graphs()
    results = {}

    for graph_name, A, info in graphs:
        n = info['n']
        print(f"\n# {graph_name}: n={n}, m={info['m']}, diam={info['diam']}")

        # Reference
        D_ref = shortest_path(A, directed=False, unweighted=True).astype(np.int32)
        D_ref[np.isinf(D_ref.astype(np.float64))] = 0

        times = []
        D = None
        for run in range(RUNS):
            s = time.perf_counter()
            D = run_apsp(A, verbose=False)
            t = time.perf_counter() - s
            times.append(t)
            print(f"  run {run+1}/{RUNS}: {t:.4f}s")

        best = min(times)
        ok = np.array_equal(D_ref, D)
        print(f"  best: {best:.4f}s  {'PASS' if ok else 'FAIL'}")
        results[graph_name] = best

    print(f"\n{'='*50}")
    print("TC3 D-STORM-GraphBLAS Results")
    print(f"{'='*50}")
    for name, t in results.items():
        print(f"  {name:<15} {t:.4f}s")

    # Save
    out_path = os.path.join(ROOT, 'tc3_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
