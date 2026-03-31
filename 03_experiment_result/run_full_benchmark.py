"""
Full Benchmark: All methods × All datasets × All variants

Methods (11 variants):
  NetworkX          — BFS APSP (Python)
  SciPy             — shortest_path (C Dijkstra)
  GraphBLAS-bfs     — per-source masked BFS (GrB_vxm)
  GraphBLAS-frontier— D-STORM logic via GrB_mxm
  I-AORM            — Incremental AORM (dense, Python loop)
  M-AORM            — Matrix-mult AORM (dense, BLAS)
  D-STORM-Sparse    — Sparse I-STORM (Cython fused pruning)
  D-STORM-Dense     — Dense M-STORM
  GPU-Dense         — cuBLAS dense matmul
  GPU-Sparse        — cuSPARSE SpMM

Datasets:
  1. Facebook (n=4039, real-world social)
  2. BA scalability (n=500, 1000, 2000, 3000, 5000)
  3. Topology (BA, ER, WS, Grid, PLC at n≈2000)

Protocol: 1 warmup (GPU only) + 3 runs, report minimum.
"""

import time
import json
import sys
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

# ── Path setup ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)

STORM_DIR = os.path.join(PROJ, '02_STORM_Implement')
AORM_DIR = os.path.join(PROJ, 'AORM-main')
GB_DIR = os.path.join(PROJ, '03_Baseline_GraphBLAS')
GPU_DIR = os.path.join(PROJ, '02_STORM_GPU_Implement')

for p in [STORM_DIR, GB_DIR, GPU_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Config ──────────────────────────────────────────────────
RUNS = 3
GPU_WARMUP = 1
FACEBOOK_PATH = os.path.join(AORM_DIR, 'datasets', 'real-world', 'facebook_combined.txt')
MAX_DENSE_N = 10000  # skip dense methods above this


# ── Graph builders ──────────────────────────────────────────
def load_facebook():
    """Load Facebook social graph as undirected CSR."""
    from storm.loader import load_graph, make_undirected
    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=True)
    A = make_undirected(A)
    return A, 'Facebook', {'n': A.shape[0], 'm': A.nnz, 'type': 'social', 'diam': 8}


def make_ba(n, m_edges=5, seed=42):
    G = nx.barabasi_albert_graph(n, m_edges, seed=seed)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    return A, f'BA-{n}', {'n': n, 'm': A.nnz, 'type': 'BA', 'diam': d}


def make_er(n=2000, p=0.005, seed=42):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    return A, 'ER-2000', {'n': A.shape[0], 'm': A.nnz, 'type': 'ER', 'diam': d}


def make_ws(n=2000, k=6, p=0.3, seed=42):
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    return A, 'WS-2000', {'n': n, 'm': A.nnz, 'type': 'WS', 'diam': d}


def make_grid(side=45):
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    return A, f'Grid-{side}x{side}', {'n': A.shape[0], 'm': A.nnz, 'type': 'Grid', 'diam': d}


def make_plc(n=2000, m_edges=5, p=0.3, seed=42):
    G = nx.powerlaw_cluster_graph(n, m_edges, p, seed=seed)
    A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    d = nx.diameter(G)
    return A, 'PLC-2000', {'n': n, 'm': A.nnz, 'type': 'PLC', 'diam': d}


# ── Method wrappers ─────────────────────────────────────────
def time_method(fn, runs=RUNS, warmup=0):
    """Run fn() with optional warmup, return (min_time, result)."""
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(runs):
        s = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - s)
    return min(times), result


def run_networkx(A):
    G = nx.from_scipy_sparse_array(A)
    def fn():
        return dict(nx.all_pairs_shortest_path_length(G))
    return time_method(fn)


def run_scipy(A):
    def fn():
        return scipy_shortest_path(A, directed=False, unweighted=True)
    return time_method(fn)


def run_gb_bfs(A):
    from graphblas_apsp import graphblas_bfs_apsp
    def fn():
        return graphblas_bfs_apsp(A, verbose=False)
    return time_method(fn)


def run_gb_frontier(A):
    from graphblas_apsp import graphblas_frontier_apsp
    def fn():
        return graphblas_frontier_apsp(A, verbose=False)
    return time_method(fn)


def run_iaorm(A):
    """I-AORM from original AORM-main."""
    sys.path.insert(0, AORM_DIR)
    try:
        from aorm import AormIterator
        from apsp import apsp_AormIterator
        n = A.shape[0]
        A_dense = A.toarray().astype(float)
        def fn():
            return apsp_AormIterator(A_dense, method='edge')
        return time_method(fn)
    except Exception as e:
        print(f"    [I-AORM error: {e}]")
        return None, None
    finally:
        sys.path.pop(0)


def run_maorm(A):
    """M-AORM from original AORM-main."""
    sys.path.insert(0, AORM_DIR)
    try:
        from aorm import AormIterator
        from apsp import apsp_AormIterator
        A_dense = A.toarray().astype(float)
        def fn():
            return apsp_AormIterator(A_dense, method='matmult')
        return time_method(fn)
    except Exception as e:
        print(f"    [M-AORM error: {e}]")
        return None, None
    finally:
        sys.path.pop(0)


def run_dstorm_sparse(A):
    from storm.apsp import storm_apsp
    def fn():
        return storm_apsp(A, verbose=False)
    return time_method(fn)


def run_dstorm_dense(A):
    from storm.apsp import storm_apsp_dense
    A_dense = A.toarray().astype(np.float32)
    def fn():
        return storm_apsp_dense(A_dense, verbose=False)
    return time_method(fn)


def run_gpu_dense(A):
    from storm_gpu.apsp import gpu_storm_apsp_dense
    def fn():
        return gpu_storm_apsp_dense(A, verbose=False)
    return time_method(fn, warmup=GPU_WARMUP)


def run_gpu_sparse(A):
    from storm_gpu.apsp import gpu_storm_apsp
    def fn():
        return gpu_storm_apsp(A, verbose=False)
    return time_method(fn, warmup=GPU_WARMUP)



# ── All methods registry ────────────────────────────────────
ALL_METHODS = [
    ('NetworkX',           'networkx',  run_networkx,       False),
    ('SciPy',              'scipy',     run_scipy,          False),
    ('GraphBLAS-bfs',      'graphblas', run_gb_bfs,         False),
    ('GraphBLAS-frontier', 'graphblas', run_gb_frontier,    False),
    ('I-AORM',             'aorm',      run_iaorm,          True),   # dense only
    ('M-AORM',             'aorm',      run_maorm,          True),   # dense only
    ('D-STORM-Sparse',     'dstorm',    run_dstorm_sparse,  False),
    ('D-STORM-Dense',      'dstorm',    run_dstorm_dense,   True),   # dense only
    ('GPU-Dense',          'gpu',       run_gpu_dense,      True),   # dense only
    ('GPU-Sparse',         'gpu',       run_gpu_sparse,     False),
]


def benchmark_one(A, graph_name, graph_info, methods=None):
    """Run all methods on one graph. Returns list of result dicts."""
    n = graph_info['n']
    results = []
    print(f"\n{'='*60}")
    print(f"  {graph_name}: n={graph_info['n']}, m={graph_info['m']}, "
          f"d={graph_info.get('diam','?')}, type={graph_info['type']}")
    print(f"{'='*60}")

    for name, family, fn, is_dense in ALL_METHODS:
        if methods and name not in methods:
            continue

        # Skip dense methods for large graphs
        if is_dense and n > MAX_DENSE_N:
            print(f"  {name:22s} — SKIPPED (n={n} > {MAX_DENSE_N})")
            results.append({
                'method': name, 'family': family,
                'graph': graph_name, **graph_info,
                'time_s': None, 'skipped': True, 'reason': f'n>{MAX_DENSE_N}'
            })
            continue

        try:
            t, D = fn(A)
            if t is None:
                print(f"  {name:22s} — FAILED")
                results.append({
                    'method': name, 'family': family,
                    'graph': graph_name, **graph_info,
                    'time_s': None, 'skipped': True, 'reason': 'error'
                })
                continue

            print(f"  {name:22s}  {t:>8.4f}s")
            results.append({
                'method': name, 'family': family,
                'graph': graph_name, **graph_info,
                'time_s': round(t, 6),
            })
        except Exception as e:
            print(f"  {name:22s} — ERROR: {e}")
            results.append({
                'method': name, 'family': family,
                'graph': graph_name, **graph_info,
                'time_s': None, 'skipped': True, 'reason': str(e)[:80]
            })

    return results


# ── Main benchmark suite ────────────────────────────────────
def main():
    all_results = []

    # ─── Experiment A: Facebook ──────────────────────────────
    print("\n" + "#"*60)
    print("# EXPERIMENT A: Facebook Social Network")
    print("#"*60)
    A_fb, name_fb, info_fb = load_facebook()
    all_results.extend(benchmark_one(A_fb, name_fb, info_fb))

    # ─── Experiment B: BA Scalability ────────────────────────
    print("\n" + "#"*60)
    print("# EXPERIMENT B: BA Graph Scalability")
    print("#"*60)
    for n_val in [500, 1000, 2000, 3000, 5000]:
        A_ba, name_ba, info_ba = make_ba(n_val)
        all_results.extend(benchmark_one(A_ba, name_ba, info_ba))

    # ─── Experiment C: Topology Effect ───────────────────────
    print("\n" + "#"*60)
    print("# EXPERIMENT C: Topology Effect (n~2000)")
    print("#"*60)
    for builder in [
        lambda: make_ba(2000),
        make_er,
        make_ws,
        make_grid,
        make_plc,
    ]:
        A_t, name_t, info_t = builder()
        all_results.extend(benchmark_one(A_t, name_t, info_t))

    # ─── Save results ────────────────────────────────────────
    out_path = os.path.join(ROOT, 'full_benchmark_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'protocol': {
                'runs': RUNS,
                'gpu_warmup': GPU_WARMUP,
                'metric': 'min_of_runs',
                'date': '2026-03-31',
                'cpu': 'measured on current machine',
                'gpu': 'NVIDIA GeForce RTX 5080, CUDA 12.9, CuPy 14.0.1',
            },
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n# Results saved to {out_path}")

    # ─── Print summary table ─────────────────────────────────
    print_summary(all_results)


def print_summary(results):
    """Print formatted summary tables."""
    from collections import defaultdict

    # Group by graph
    by_graph = defaultdict(list)
    for r in results:
        by_graph[r['graph']].append(r)

    print("\n" + "="*90)
    print(" FULL BENCHMARK SUMMARY")
    print("="*90)

    for graph_name, entries in by_graph.items():
        info = entries[0]
        print(f"\n--- {graph_name} (n={info['n']}, m={info['m']}, d={info.get('diam','?')}) ---")
        print(f"  {'Method':<22s} {'Time (s)':>10s} {'vs fastest':>12s}")
        print(f"  {'-'*46}")

        valid = [(e['method'], e['time_s']) for e in entries if e.get('time_s') is not None]
        if not valid:
            print("  (no valid results)")
            continue

        fastest = min(t for _, t in valid)
        for method, t in sorted(valid, key=lambda x: x[1]):
            ratio = t / fastest
            marker = ' *' if t == fastest else ''
            print(f"  {method:<22s} {t:>10.4f} {ratio:>11.1f}x{marker}")


if __name__ == '__main__':
    main()
