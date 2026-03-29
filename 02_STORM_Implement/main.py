"""
STORM: Scalable Topological Order Reachability Matrix

Main entry point for running STORM experiments.

Usage:
    # Sparse I-STORM (default, recommended)
    python main.py -m storm -i graph.txt -r

    # Dense M-STORM (legacy, small graphs only)
    python main.py -m dense -i graph.gpickle

    # GPU-accelerated STORM
    python main.py -m gpu -i graph.txt -r

    # Hop-constrained APSP
    python main.py -m storm -i graph.txt -r -k 5

    # Compute reachability profile (STORM-PE)
    python main.py -m pe -i graph.txt -r -k 10

    # Compute girth
    python main.py -m girth -i graph.txt -r

    # Compare with NetworkX
    python main.py -m nx -i graph.txt -r
"""

import argparse
import time
import sys
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="STORM: Scalable Topological Order Reachability Matrix"
    )
    parser.add_argument(
        '-m', '--method', default='storm',
        choices=['storm', 'dense', 'gpu', 'pe', 'girth', 'nx', 'compare'],
        help='Method (storm: sparse I-STORM, dense: M-STORM, gpu: GPU-STORM, '
             'pe: STORM-PE profiles, girth: cycle detection, nx: NetworkX, '
             'compare: benchmark all methods)'
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
        '--pe-dim', type=int, default=10,
        help='Max K for STORM-PE reachability profile'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load graph ──────────────────────────────────────────────
    from storm.loader import load_graph

    fmt = 'edgelist' if args.realworld else None
    A, n, m = load_graph(args.input, fmt=fmt, directed=True)
    print(f"# Graph [{args.input}]: |V|={n}, |E|={m}, "
          f"avg_degree={m/n:.2f}, density={m/(n*n):.6f}")

    if not args.directed:
        from storm.loader import make_undirected
        A = make_undirected(A)
        m = A.nnz
        print(f"# Undirected: |E|={m}")

    from storm.loader import is_connected
    if not is_connected(A):
        print(f"# Warning: graph is disconnected")

    k = args.order if args.order > 0 else -1

    # ── Method dispatch ─────────────────────────────────────────
    if args.method == 'storm':
        run_storm(A, k, args)

    elif args.method == 'dense':
        run_dense(A, k, args)

    elif args.method == 'gpu':
        run_gpu(A, k, args)

    elif args.method == 'pe':
        run_pe(A, args)

    elif args.method == 'girth':
        run_girth(A, k, args)

    elif args.method == 'nx':
        run_networkx(A, k, args)

    elif args.method == 'compare':
        run_compare(A, k, args)


def run_storm(A, k, args):
    """Run sparse I-STORM APSP."""
    from storm.apsp import storm_apsp

    print(f"# Running Sparse I-STORM (k={'max' if k < 0 else k})")
    start = time.perf_counter()
    D = storm_apsp(A, k=k, verbose=True)
    elapsed = time.perf_counter() - start

    if sp.issparse(D):
        print(f"# I-STORM completed: {elapsed:.3f}s, D.nnz={D.nnz}")
    else:
        print(f"# I-STORM completed: {elapsed:.3f}s")

    if args.output:
        _save_distance(D, args.output)


def run_dense(A, k, args):
    """Run dense M-STORM APSP."""
    from storm.apsp import storm_apsp_dense

    if A.shape[0] > 20000:
        print(f"# Warning: Dense mode with n={A.shape[0]} requires "
              f"~{A.shape[0]**2 * 4 / 1e9:.1f}GB per matrix. Consider -m storm.")

    A_dense = A.toarray().astype(np.float32)

    print(f"# Running Dense M-STORM (k={'max' if k < 0 else k})")
    start = time.perf_counter()
    D = storm_apsp_dense(A_dense, k=k, verbose=True)
    elapsed = time.perf_counter() - start

    print(f"# M-STORM completed: {elapsed:.3f}s")

    if args.output:
        _save_distance(D, args.output)


def run_gpu(A, k, args):
    """Run GPU-accelerated STORM APSP."""
    from storm.gpu import gpu_storm_apsp, is_gpu_available

    if not is_gpu_available():
        print("# GPU not available. Falling back to CPU sparse STORM.")
        run_storm(A, k, args)
        return

    print(f"# Running GPU-STORM (k={'max' if k < 0 else k})")
    start = time.perf_counter()
    D = gpu_storm_apsp(A, k=k, verbose=True)
    elapsed = time.perf_counter() - start

    print(f"# GPU-STORM completed: {elapsed:.3f}s")

    if args.output:
        _save_distance(D, args.output)


def run_pe(A, args):
    """Compute STORM-PE reachability profiles."""
    from storm.pe import compute_reachability_profile

    K = args.pe_dim
    directed = args.directed

    print(f"# Computing STORM-PE profiles (K={K}, directed={directed})")
    start = time.perf_counter()
    profiles = compute_reachability_profile(A, K=K, directed=directed)
    elapsed = time.perf_counter() - start

    print(f"# STORM-PE completed: {elapsed:.3f}s, "
          f"shape={profiles.shape}")
    print(f"# Profile stats: mean={profiles.mean():.2f}, "
          f"max={profiles.max():.0f}, nonzero={np.count_nonzero(profiles)}")

    if args.output:
        np.savetxt(args.output, profiles, fmt='%.1f')
        print(f"# Profiles saved to {args.output}")


def run_girth(A, k, args):
    """Compute graph girth."""
    from storm.girth import storm_girth, storm_girth_per_node

    print(f"# Computing graph girth")
    start = time.perf_counter()
    g = storm_girth(A, k_max=k, verbose=True)
    elapsed = time.perf_counter() - start

    if g > 0:
        print(f"# Girth: {g} (computed in {elapsed:.3f}s)")
    else:
        print(f"# Graph is acyclic (checked in {elapsed:.3f}s)")


def run_networkx(A, k, args):
    """Run NetworkX APSP for comparison."""
    import networkx as nx
    import pandas as pd

    if args.directed:
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_scipy_sparse_array(A)

    print(f"# Running NetworkX APSP (k={'max' if k < 0 else k})")
    start = time.perf_counter()
    if k > 0:
        path = dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    else:
        path = dict(nx.all_pairs_shortest_path_length(G))
    elapsed = time.perf_counter() - start

    print(f"# NetworkX completed: {elapsed:.3f}s")

    if args.output:
        n = A.shape[0]
        D = np.zeros((n, n), dtype=np.float32)
        for src, dists in path.items():
            for tgt, dist in dists.items():
                D[src, tgt] = dist
        np.fill_diagonal(D, 0)
        _save_distance(D, args.output)


def run_compare(A, k, args):
    """Benchmark all available methods."""
    import networkx as nx

    results = {}
    n = A.shape[0]

    # Sparse STORM
    from storm.apsp import storm_apsp
    start = time.perf_counter()
    D_storm = storm_apsp(A, k=k, verbose=False)
    results['I-STORM'] = time.perf_counter() - start

    # Dense STORM (only if small enough)
    if n <= 10000:
        from storm.apsp import storm_apsp_dense
        A_dense = A.toarray().astype(np.float32)
        start = time.perf_counter()
        D_dense = storm_apsp_dense(A_dense, k=k, verbose=False)
        results['M-STORM'] = time.perf_counter() - start

    # GPU STORM
    from storm.gpu import is_gpu_available
    if is_gpu_available():
        from storm.gpu import gpu_storm_apsp
        start = time.perf_counter()
        D_gpu = gpu_storm_apsp(A, k=k, verbose=False)
        results['GPU-STORM'] = time.perf_counter() - start

    # NetworkX
    if args.directed:
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_scipy_sparse_array(A)

    start = time.perf_counter()
    if k > 0:
        dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    else:
        dict(nx.all_pairs_shortest_path_length(G))
    results['NetworkX'] = time.perf_counter() - start

    # Print results
    print(f"\n# Benchmark results (n={n}, m={A.nnz}, k={'max' if k < 0 else k})")
    print(f"# {'Method':<15} {'Time (s)':>10} {'Speedup vs NX':>15}")
    print(f"# {'-'*42}")

    nx_time = results.get('NetworkX', 1.0)
    for method, t in sorted(results.items(), key=lambda x: x[1]):
        speedup = nx_time / t if t > 0 else float('inf')
        marker = ' *' if t == min(results.values()) else ''
        print(f"# {method:<15} {t:>10.3f} {speedup:>14.1f}x{marker}")


def _save_distance(D, filepath):
    """Save distance matrix to file."""
    if sp.issparse(D):
        D = D.toarray()
    np.savetxt(filepath, D, delimiter=' ', fmt='%g')
    print(f"# Distance matrix saved to {filepath}")


if __name__ == '__main__':
    main()
