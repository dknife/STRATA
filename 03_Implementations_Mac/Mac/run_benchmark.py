"""
Mac Benchmark: 6 CPU methods x 6 graphs.
(BCM5/TCM3 GraphBLAS methods run separately via run_tcm3_standalone.py)

Graphs:
  1. Facebook      (n=4039, social, diam~8)
  2. BA-2000       (n=2000, scale-free, diam~5)
  3. WS-2000       (n=2000, small-world, diam~8)
  4. ER-2000       (n=2000, random, diam~6)
  5. Grid-45x45    (n=2025, lattice, diam=88)
  6. PLC-2000      (n=2000, powerlaw-cluster, diam~5)

Methods: BCM1..BCM4, TCM1..TCM2
(BCM5, TCM3 require separate process due to GrB_init conflict)

Protocol: 3 runs, report minimum.

Usage:
  cd 03_Implementations/Mac
  pip install numpy scipy networkx tqdm
  python run_benchmark.py
"""

import time
import json
import sys
import os
import platform
import importlib
import numpy as np
import scipy.sparse as sp
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.loader import load_graph, make_undirected

# ── Config ──────────────────────────────────────────────────
RUNS = 3
MAX_DENSE_N = 10000
FACEBOOK_PATH = os.path.join(ROOT, '..', '..', '04_Datasets',
                              'real-world', 'facebook_combined.txt')

# Methods: (ID, label, module_path, needs_graphblas)
# BCM5 and TCM3 excluded — run via run_tcm3_standalone.py
METHODS = [
    ('BCM1', 'NetworkX',           'BCM1_NetworkX.apsp',        False),
    ('BCM2', 'SciPy',              'BCM2_SciPy.apsp',           False),
    ('BCM3', 'I-AORM',             'BCM3_IAORM.apsp',           False),
    ('BCM4', 'M-AORM',             'BCM4_MAORM.apsp',           False),
    ('TCM1', 'STRATA-SpMM',       'TCM1_STRATA_SpMM.apsp',    False),
    ('TCM2', 'STRATA-NumpyBLAS',  'TCM2_STRATA_NumpyBLAS.apsp', False),
]


# ── Graph builders ──────────────────────────────────────────

def build_graphs():
    graphs = []

    # 1. Facebook
    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=True)
    A = make_undirected(A)
    graphs.append(('Facebook', A, {'n': A.shape[0], 'm': A.nnz,
                                    'type': 'social', 'diam': 8}))

    # 2. BA-2000
    G = nx.barabasi_albert_graph(2000, 5, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    graphs.append(('BA-2000', A, {'n': 2000, 'm': A.nnz,
                                   'type': 'BA', 'diam': d}))

    # 3. WS-2000
    G = nx.watts_strogatz_graph(2000, 6, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    graphs.append(('WS-2000', A, {'n': 2000, 'm': A.nnz,
                                   'type': 'WS', 'diam': d}))

    # 4. ER-2000
    G = nx.erdos_renyi_graph(2000, 0.005, seed=42)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    graphs.append(('ER-2000', A, {'n': A.shape[0], 'm': A.nnz,
                                   'type': 'ER', 'diam': d}))

    # 5. Grid-45x45
    G = nx.grid_2d_graph(45, 45)
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    graphs.append(('Grid-45x45', A, {'n': A.shape[0], 'm': A.nnz,
                                      'type': 'Grid', 'diam': d}))

    # 6. PLC-2000
    G = nx.powerlaw_cluster_graph(2000, 5, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    graphs.append(('PLC-2000', A, {'n': 2000, 'm': A.nnz,
                                    'type': 'PLC', 'diam': d}))

    return graphs


def bench_method(mod, A, k):
    """Run method 3 times, return (min_time, D)."""
    times = []
    D = None
    for _ in range(RUNS):
        s = time.perf_counter()
        D = mod.run_apsp(A, k=k, verbose=False)
        times.append(time.perf_counter() - s)
    return min(times), D


def print_env_info():
    """Print macOS environment information."""
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python:    {platform.python_version()}")
    print(f"NumPy:     {np.__version__}")
    print(f"SciPy:     {sp.__version__ if hasattr(sp, '__version__') else 'unknown'}")
    import scipy
    print(f"SciPy:     {scipy.__version__}")
    print(f"NetworkX:  {nx.__version__}")

    # Check BLAS backend
    np_config = np.__config__
    if hasattr(np_config, 'show'):
        print("\nNumPy BLAS config:")
        np_config.show()
    print()


def main():
    print("=" * 70)
    print("Mac APSP Benchmark (CPU only)")
    print("=" * 70)
    print_env_info()

    print("Building graphs...")
    graphs = build_graphs()

    # Pre-load modules
    modules = {}
    for mid, label, mod_path, needs_gb in METHODS:
        try:
            modules[mid] = importlib.import_module(mod_path)
            print(f"  {mid} {label}: loaded")
        except ImportError as e:
            print(f"  {mid} {label}: IMPORT ERROR ({e})")

    all_results = {}

    for graph_name, A, info in graphs:
        n = info['n']
        print(f"\n{'=' * 70}")
        print(f"# {graph_name}: n={n}, m={info['m']}, "
              f"type={info['type']}, diam={info['diam']}")
        print(f"{'=' * 70}")

        graph_results = {}
        D_ref = None

        for mid, label, mod_path, needs_gb in METHODS:
            if mid not in modules:
                continue

            # Skip dense methods for large n
            if mid in ('BCM3', 'BCM4', 'TCM2') and n > MAX_DENSE_N:
                print(f"  {mid} {label:<22} SKIPPED (n>{MAX_DENSE_N})")
                continue

            mod = modules[mid]

            try:
                t, D = bench_method(mod, A, k=-1)
            except Exception as e:
                print(f"  {mid} {label:<22} ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Correctness check
            D_int = D.astype(np.int32)
            if D_ref is None:
                D_ref = D_int
                ok_str = 'ref'
            else:
                ok = np.array_equal(D_ref, D_int)
                ok_str = 'PASS' if ok else 'MISMATCH'

            graph_results[mid] = {'label': label, 'time': t, 'check': ok_str}
            print(f"  {mid} {label:<22} {t:>8.4f}s  {ok_str}")

        # Summary table
        if graph_results:
            scipy_t = graph_results.get('BCM2', {}).get('time', 1.0)
            fastest_t = min(r['time'] for r in graph_results.values())

            print(f"\n  {'ID':<5} {'Method':<22} {'Time':>9} {'vs SciPy':>9} "
                  f"{'vs best':>9}")
            print(f"  {'-' * 57}")
            for mid, r in sorted(graph_results.items(),
                                  key=lambda x: x[1]['time']):
                vs_scipy = scipy_t / r['time'] if r['time'] > 0 else 0
                vs_best = fastest_t / r['time'] if r['time'] > 0 else 0
                marker = ' *' if r['time'] == fastest_t else ''
                print(f"  {mid:<5} {r['label']:<22} {r['time']:>8.4f}s "
                      f"{vs_scipy:>8.1f}x {vs_best:>8.2f}x{marker}")

        all_results[graph_name] = {
            'info': info,
            'results': {mid: {'label': r['label'], 'time': r['time'],
                               'check': r['check']}
                        for mid, r in graph_results.items()}
        }

    # ── Save JSON ────────────────────────────────────────────
    out_path = os.path.join(ROOT, 'mac_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Grand summary ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("# Grand Summary: per-method across all graphs")
    print(f"{'=' * 70}")

    method_totals = {}
    for graph_name, data in all_results.items():
        for mid, r in data['results'].items():
            if mid not in method_totals:
                method_totals[mid] = {'label': r['label'], 'times': []}
            method_totals[mid]['times'].append(r['time'])

    print(f"  {'ID':<5} {'Method':<22} {'Graphs':>6} {'Total':>9} {'Mean':>9}")
    print(f"  {'-' * 55}")
    for mid, info in sorted(method_totals.items(),
                             key=lambda x: sum(x[1]['times'])):
        total = sum(info['times'])
        mean = total / len(info['times'])
        print(f"  {mid:<5} {info['label']:<22} {len(info['times']):>6} "
              f"{total:>8.3f}s {mean:>8.4f}s")

    print(f"\nNote: BCM5 (GB-bfs) and TCM3 (GB-frontier) require separate run:")
    print(f"  python run_tcm3_standalone.py")


if __name__ == '__main__':
    main()
