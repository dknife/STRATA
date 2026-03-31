"""
GraphBLAS APSP Baseline

Benchmarks SuiteSparse:GraphBLAS-based APSP against D-STORM and other methods.

Usage:
    # GraphBLAS frontier APSP (closest to D-STORM logic)
    python main.py -m frontier -i graph.txt -r

    # GraphBLAS per-source BFS APSP (standard LAGraph pattern)
    python main.py -m bfs -i graph.txt -r

    # Compare all GraphBLAS methods + D-STORM + SciPy + NetworkX
    python main.py -m compare -i graph.txt -r

    # Hop-constrained APSP
    python main.py -m frontier -i graph.txt -r -k 5
"""

import argparse
import time
import sys
import os
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphBLAS APSP Baseline for D-STORM comparison"
    )
    parser.add_argument(
        '-m', '--method', default='frontier',
        choices=['bfs', 'level', 'frontier', 'compare'],
        help='Method: bfs (per-source vxm), level (batch SpMM), '
             'frontier (D-STORM logic via GraphBLAS), compare (all methods)'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input graph file path'
    )
    parser.add_argument(
        '-k', '--order', type=int, default=-1,
        help='Reachability constraint k-order (-1 for full)'
    )
    parser.add_argument(
        '-r', '--realworld', action='store_true',
        help='Real-world network (edge-list format)'
    )
    parser.add_argument(
        '-d', '--directed', action='store_true', default=False,
        help='Treat as directed graph'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output file for distance matrix'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='Number of timing runs (report minimum)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load graph ──────────────────────────────────────────────
    from loader import load_graph, make_undirected, is_connected

    fmt = 'edgelist' if args.realworld else None
    A, n, m = load_graph(args.input, fmt=fmt, directed=True)

    if not args.directed:
        A = make_undirected(A)
        m = A.nnz

    print(f"# Graph [{args.input}]: |V|={n}, |E|={m}, "
          f"avg_degree={m/n:.2f}, density={m/(n*n):.6f}")

    if not is_connected(A):
        print("# Warning: graph is disconnected")

    k = args.order if args.order > 0 else -1

    # ── Method dispatch ─────────────────────────────────────────
    if args.method == 'compare':
        run_compare(A, k, args)
    else:
        run_graphblas(A, k, args)


def run_graphblas(A, k, args):
    """Run a single GraphBLAS APSP method."""
    from graphblas_apsp import graphblas_apsp

    method = args.method
    print(f"# Running GraphBLAS-{method} (k={'full' if k < 0 else k})")

    times = []
    D = None
    for run in range(args.runs):
        start = time.perf_counter()
        D = graphblas_apsp(A, method=method, k=k, verbose=(run == 0))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"#   run {run+1}/{args.runs}: {elapsed:.3f}s")

    best = min(times)
    print(f"# GraphBLAS-{method} completed: {best:.3f}s (best of {args.runs})")
    print(f"#   D nonzero: {np.count_nonzero(D)}, "
          f"max distance: {D.max()}")

    if args.output and D is not None:
        np.savetxt(args.output, D, delimiter=' ', fmt='%d')
        print(f"# Distance matrix saved to {args.output}")


def run_compare(A, k, args):
    """Benchmark all available methods including D-STORM and SciPy."""
    import networkx as nx

    n = A.shape[0]
    results = {}
    D_ref = None

    # ── GraphBLAS methods ───────────────────────────────────────
    from graphblas_apsp import graphblas_apsp

    for gb_method in ['bfs', 'frontier']:
        label = f'GB-{gb_method}'
        print(f"\n# Running {label}...")
        times = []
        D = None
        for run in range(args.runs):
            start = time.perf_counter()
            D = graphblas_apsp(A, method=gb_method, k=k, verbose=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results[label] = min(times)
        if D_ref is None:
            D_ref = D
        print(f"#   {label}: {min(times):.3f}s (best of {args.runs})")

    # ── D-STORM (if available) ──────────────────────────────────
    storm_dir = os.path.join(os.path.dirname(__file__),
                             '..', '02_STORM_Implement')
    if os.path.isdir(storm_dir):
        sys.path.insert(0, storm_dir)
        try:
            from storm.apsp import storm_apsp
            print(f"\n# Running D-STORM...")
            times = []
            for run in range(args.runs):
                start = time.perf_counter()
                D_storm = storm_apsp(A, k=k, verbose=False)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            results['D-STORM'] = min(times)
            print(f"#   D-STORM: {min(times):.3f}s (best of {args.runs})")

            # Verify correctness
            if D_ref is not None:
                if sp.issparse(D_storm):
                    D_storm_dense = D_storm.toarray()
                else:
                    D_storm_dense = D_storm
                match = np.array_equal(D_ref, D_storm_dense.astype(np.int32))
                print(f"#   D-STORM vs GB-bfs correctness: {'PASS' if match else 'MISMATCH'}")
        except ImportError as e:
            print(f"#   D-STORM not available: {e}")
        finally:
            sys.path.pop(0)

    # ── SciPy (C Dijkstra) ──────────────────────────────────────
    from scipy.sparse.csgraph import shortest_path
    print(f"\n# Running SciPy shortest_path...")
    times = []
    for run in range(args.runs):
        start = time.perf_counter()
        D_scipy = shortest_path(A, directed=False, unweighted=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    results['SciPy'] = min(times)
    print(f"#   SciPy: {min(times):.3f}s (best of {args.runs})")

    # ── NetworkX BFS ────────────────────────────────────────────
    G = nx.from_scipy_sparse_array(A)
    print(f"\n# Running NetworkX BFS...")
    times = []
    for run in range(args.runs):
        start = time.perf_counter()
        if k > 0:
            dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
        else:
            dict(nx.all_pairs_shortest_path_length(G))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    results['NetworkX'] = min(times)
    print(f"#   NetworkX: {min(times):.3f}s (best of {args.runs})")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"# Benchmark results (n={n}, m={A.nnz}, k={'full' if k < 0 else k})")
    print(f"# {'Method':<18} {'Time (s)':>10} {'vs NX':>10} {'vs SciPy':>10}")
    print(f"# {'-'*50}")

    nx_time = results.get('NetworkX', 1.0)
    scipy_time = results.get('SciPy', 1.0)
    fastest = min(results.values())

    for method, t in sorted(results.items(), key=lambda x: x[1]):
        vs_nx = nx_time / t if t > 0 else float('inf')
        vs_scipy = scipy_time / t if t > 0 else float('inf')
        marker = ' *' if t == fastest else ''
        print(f"# {method:<18} {t:>10.3f} {vs_nx:>9.1f}x {vs_scipy:>9.1f}x{marker}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
