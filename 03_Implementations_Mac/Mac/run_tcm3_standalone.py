"""
Standalone benchmark for GraphBLAS methods: BCM5 + TCM3.

Must run in a separate process from the main benchmark to avoid
GraphBLAS initialization conflict (GrB_init called twice).

Usage:
  cd 03_Implementations/Mac
  pip install numpy scipy networkx suitesparse-graphblas tqdm
  python run_tcm3_standalone.py
"""

import time
import json
import sys
import os
import platform
import numpy as np
import scipy.sparse as sp
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.loader import load_graph, make_undirected

RUNS = 3
FACEBOOK_PATH = os.path.join(ROOT, '..', '..', '04_Datasets',
                              'real-world', 'facebook_combined.txt')


def build_graphs():
    graphs = []

    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=True)
    A = make_undirected(A)
    graphs.append(('Facebook', A, {'n': A.shape[0], 'm': A.nnz,
                                    'type': 'social', 'diam': 8}))

    G = nx.barabasi_albert_graph(2000, 5, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('BA-2000', A, {'n': 2000, 'm': A.nnz,
                                   'type': 'BA', 'diam': nx.diameter(G)}))

    G = nx.watts_strogatz_graph(2000, 6, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('WS-2000', A, {'n': 2000, 'm': A.nnz,
                                   'type': 'WS', 'diam': nx.diameter(G)}))

    G = nx.erdos_renyi_graph(2000, 0.005, seed=42)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('ER-2000', A, {'n': A.shape[0], 'm': A.nnz,
                                   'type': 'ER', 'diam': nx.diameter(G)}))

    G = nx.grid_2d_graph(45, 45)
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('Grid-45x45', A, {'n': A.shape[0], 'm': A.nnz,
                                      'type': 'Grid', 'diam': nx.diameter(G)}))

    G = nx.powerlaw_cluster_graph(2000, 5, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('PLC-2000', A, {'n': 2000, 'm': A.nnz,
                                    'type': 'PLC', 'diam': nx.diameter(G)}))

    return graphs


def main():
    print("=" * 70)
    print("Mac GraphBLAS Benchmark: BCM5 (GB-bfs) + TCM3 (GB-frontier)")
    print("=" * 70)
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python:    {platform.python_version()}")
    print()

    # Import GraphBLAS methods (same process = shared GrB_init)
    from BCM5_GB_bfs.apsp import run_apsp as bcm5_apsp
    from TCM3_STRATA_GraphBLAS.apsp import run_apsp as tcm3_apsp
    from scipy.sparse.csgraph import shortest_path

    print("Building graphs...")
    graphs = build_graphs()

    bcm5_results = {}
    tcm3_results = {}

    for graph_name, A, info in graphs:
        n = info['n']
        print(f"\n{'=' * 70}")
        print(f"# {graph_name}: n={n}, m={info['m']}, diam={info['diam']}")
        print(f"{'=' * 70}")

        # Reference
        D_ref = shortest_path(A, directed=False, unweighted=True).astype(np.int32)
        D_ref[np.isinf(D_ref.astype(np.float64))] = 0

        # BCM5 GB-bfs
        print(f"\n  BCM5 GB-bfs:")
        times = []
        D = None
        for run in range(RUNS):
            s = time.perf_counter()
            D = bcm5_apsp(A, verbose=False)
            t = time.perf_counter() - s
            times.append(t)
            print(f"    run {run + 1}/{RUNS}: {t:.4f}s")
        best = min(times)
        ok = np.array_equal(D_ref, D)
        print(f"    best: {best:.4f}s  {'PASS' if ok else 'FAIL'}")
        bcm5_results[graph_name] = {'time': best, 'check': 'PASS' if ok else 'FAIL'}

        # TCM3 GB-frontier
        print(f"\n  TCM3 GB-frontier:")
        times = []
        D = None
        for run in range(RUNS):
            s = time.perf_counter()
            D = tcm3_apsp(A, verbose=False)
            t = time.perf_counter() - s
            times.append(t)
            print(f"    run {run + 1}/{RUNS}: {t:.4f}s")
        best = min(times)
        ok = np.array_equal(D_ref, D)
        print(f"    best: {best:.4f}s  {'PASS' if ok else 'FAIL'}")
        tcm3_results[graph_name] = {'time': best, 'check': 'PASS' if ok else 'FAIL'}

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  {'Graph':<15} {'BCM5 GB-bfs':>12} {'TCM3 GB-front':>14}")
    print(f"  {'-' * 43}")
    for graph_name in bcm5_results:
        b5 = bcm5_results[graph_name]['time']
        t3 = tcm3_results[graph_name]['time']
        print(f"  {graph_name:<15} {b5:>11.4f}s {t3:>13.4f}s")

    # Save
    out_path = os.path.join(ROOT, 'mac_graphblas_results.json')
    results = {'BCM5': bcm5_results, 'TCM3': tcm3_results}
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
