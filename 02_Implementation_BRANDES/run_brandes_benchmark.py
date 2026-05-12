"""
Brandes Betweenness Centrality Benchmark
=========================================
BB1  Brandes-Python       Pure Python per-source BFS (n loops)
BB2  Brandes-C (igraph)   C-level Brandes via igraph
TB1  Matrix-Brandes       STRATA SpMM forward + SpMM backward (2d calls)

Protocol: 3 runs, minimum time reported.
Correctness: all methods compared element-wise (rtol=1e-6).
"""

import time
import json
import sys
import os
import importlib
import numpy as np
import scipy.sparse as sp
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATASET_ROOT = os.path.join(ROOT, '..', '04_Datasets')
IMPL_ROOT = os.path.join(ROOT, '..', '02_Implementations')

FACEBOOK_PATH = os.path.join(DATASET_ROOT, 'real-world', 'facebook_combined.txt')

RUNS = 3

METHODS = [
    ('TB2', 'Matrix-Brandes-C (STRATA)', 'TB2_BrandesC_STRATA.brandes'),
    ('BB2', 'Brandes-C (igraph)',         'BB2_BrandesC.brandes'),
    ('TB1', 'Matrix-Brandes-Numba',       'TB1_BrandesSTRATA.brandes'),
    ('BB1', 'Brandes-Python',            'BB1_BrandesPython.brandes'),
]


# ── Graph builders ──────────────────────────────────────────

def load_facebook():
    """Load Facebook ego-network (undirected)."""
    sys.path.insert(0, IMPL_ROOT)
    from common.loader import load_graph, make_undirected
    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=True)
    A = make_undirected(A)
    return A


def build_graphs():
    graphs = []

    # 1. Facebook
    try:
        A = load_facebook()
        G_nx = nx.from_scipy_sparse_array(A)
        d = nx.diameter(G_nx)
        graphs.append(('Facebook', A,
                        {'n': A.shape[0], 'm': A.nnz, 'diam': d}))
    except Exception as e:
        print(f"  Facebook: SKIP ({e})")

    # 2. BA-2000
    G = nx.barabasi_albert_graph(2000, 5, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('BA-2000', A,
                    {'n': 2000, 'm': A.nnz, 'diam': nx.diameter(G)}))

    # 3. WS-2000
    G = nx.watts_strogatz_graph(2000, 6, 0.3, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('WS-2000', A,
                    {'n': 2000, 'm': A.nnz, 'diam': nx.diameter(G)}))

    # 4. BA-500  (quick test)
    G = nx.barabasi_albert_graph(500, 5, seed=42)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    graphs.append(('BA-500', A,
                    {'n': 500, 'm': A.nnz, 'diam': nx.diameter(G)}))

    return graphs


# ── Benchmark runner ────────────────────────────────────────

def bench(mod, A):
    """Run method RUNS times, return (min_time, cb)."""
    times = []
    cb = None
    for _ in range(RUNS):
        s = time.perf_counter()
        cb = mod.run_brandes(A, verbose=False)
        times.append(time.perf_counter() - s)
    return min(times), cb


def main():
    print("Building graphs...")
    graphs = build_graphs()

    # Pre-load modules
    modules = {}
    for mid, label, mod_path in METHODS:
        try:
            modules[mid] = importlib.import_module(mod_path)
        except ImportError as e:
            print(f"  {mid} {label}: IMPORT ERROR ({e})")

    all_results = {}

    for graph_name, A, info in graphs:
        n = info['n']
        print(f"\n{'=' * 65}")
        print(f"# {graph_name}: n={n}, m={info['m']}, diam={info['diam']}")
        print(f"{'=' * 65}")

        graph_results = {}
        cb_ref = None
        ref_mid = None

        for mid, label, mod_path in METHODS:
            if mid not in modules:
                continue

            mod = modules[mid]

            try:
                t, cb = bench(mod, A)
            except Exception as e:
                print(f"  {mid} {label:<30} ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            # Correctness check
            if cb_ref is None:
                cb_ref = cb
                ref_mid = mid
                ok_str = 'ref'
            else:
                ok = np.allclose(cb_ref, cb, rtol=1e-6, atol=1e-8)
                if not ok:
                    diff = np.abs(cb_ref - cb)
                    ok_str = f'MISMATCH (max_err={diff.max():.2e})'
                else:
                    ok_str = 'PASS'

            graph_results[mid] = {'label': label, 'time': t, 'check': ok_str}
            print(f"  {mid} {label:<30} {t:>8.3f}s  {ok_str}")

        # Summary table
        if len(graph_results) >= 2:
            fastest = min(r['time'] for r in graph_results.values())
            bb1_t = graph_results.get('BB1', {}).get('time')

            print(f"\n  {'ID':<4} {'Method':<30} {'Time':>9} "
                  f"{'vs BB1':>9} {'vs best':>9}")
            print(f"  {'-' * 65}")
            for mid, r in sorted(graph_results.items(),
                                  key=lambda x: x[1]['time']):
                vs_bb1 = (bb1_t / r['time']) if bb1_t and r['time'] > 0 else 0
                vs_best = fastest / r['time'] if r['time'] > 0 else 0
                marker = ' *' if r['time'] == fastest else ''
                print(f"  {mid:<4} {r['label']:<30} {r['time']:>8.3f}s "
                      f"{vs_bb1:>8.1f}x {vs_best:>8.2f}x{marker}")

            # Top-5 centrality comparison
            if cb_ref is not None:
                top5 = np.argsort(cb_ref)[-5:][::-1]
                print(f"\n  Top-5 central nodes: {top5.tolist()}")
                print(f"  Top-5 values: {[f'{cb_ref[i]:.1f}' for i in top5]}")

        all_results[graph_name] = {
            'info': info,
            'results': {mid: {'label': r['label'], 'time': r['time'],
                               'check': r['check']}
                        for mid, r in graph_results.items()}
        }

    # ── Save JSON ────────────────────────────────────────────
    out_path = os.path.join(ROOT, 'brandes_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Grand summary ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("# Grand Summary")
    print(f"{'=' * 65}")

    method_totals = {}
    for gname, data in all_results.items():
        for mid, r in data['results'].items():
            if mid not in method_totals:
                method_totals[mid] = {'label': r['label'], 'times': [],
                                       'graphs': []}
            method_totals[mid]['times'].append(r['time'])
            method_totals[mid]['graphs'].append(gname)

    print(f"  {'ID':<4} {'Method':<30} {'Graphs':>6} {'Total':>9} {'Mean':>9}")
    print(f"  {'-' * 62}")
    for mid, info in sorted(method_totals.items(),
                             key=lambda x: sum(x[1]['times'])):
        total = sum(info['times'])
        mean = total / len(info['times'])
        print(f"  {mid:<4} {info['label']:<30} {len(info['times']):>6} "
              f"{total:>8.3f}s {mean:>8.4f}s")


if __name__ == '__main__':
    main()
