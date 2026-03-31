"""
STORM-GPU: CUDA-accelerated APSP

Usage:
    # GPU sparse (cuSPARSE SpMM) — default
    python main.py -m sparse -i graph.txt -r

    # GPU dense (cuBLAS matmul) — small graphs
    python main.py -m dense -i graph.txt -r

    # GPU fused kernel (custom CUDA) — most optimized
    python main.py -m fused -i graph.txt -r

    # Compare GPU vs CPU methods
    python main.py -m compare -i graph.txt -r
"""

import argparse
import time
import sys
import os
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="STORM-GPU: CUDA-accelerated APSP"
    )
    parser.add_argument(
        '-m', '--method', default='fused',
        choices=['sparse', 'dense', 'fused', 'compare'],
        help='GPU method: sparse (cuSPARSE), dense (cuBLAS), '
             'fused (custom CUDA kernel), compare (all methods)'
    )
    parser.add_argument('-i', '--input', required=True, help='Input graph file')
    parser.add_argument('-k', '--order', type=int, default=-1, help='Hop constraint (-1=full)')
    parser.add_argument('-r', '--realworld', action='store_true', help='Edge-list format')
    parser.add_argument('-d', '--directed', action='store_true', default=False)
    parser.add_argument('-o', '--output', default=None, help='Output distance matrix')
    parser.add_argument('--runs', type=int, default=3, help='Timing runs (report min)')
    parser.add_argument('--warmup', type=int, default=1, help='GPU warmup runs')
    return parser.parse_args()


def main():
    args = parse_args()

    from storm_gpu.loader import load_graph, make_undirected, is_connected

    fmt = 'edgelist' if args.realworld else None
    A, n, m = load_graph(args.input, fmt=fmt, directed=True)

    if not args.directed:
        A = make_undirected(A)
        m = A.nnz

    print(f"# Graph [{args.input}]: |V|={n}, |E|={m}, "
          f"avg_deg={m/n:.1f}, density={m/(n*n):.6f}")

    if not is_connected(A):
        print("# Warning: graph is disconnected")

    k = args.order if args.order > 0 else -1

    if args.method == 'compare':
        run_compare(A, k, args)
    else:
        run_gpu(A, k, args)


def run_gpu(A, k, args):
    """Run a single GPU method."""
    from storm_gpu.apsp import gpu_storm_apsp, gpu_storm_apsp_dense, gpu_storm_apsp_fused

    methods = {
        'sparse': ('GPU-Sparse', gpu_storm_apsp),
        'dense': ('GPU-Dense', gpu_storm_apsp_dense),
        'fused': ('GPU-Fused', gpu_storm_apsp_fused),
    }
    label, fn = methods[args.method]
    print(f"# Running {label} (k={'full' if k < 0 else k})")

    # Warmup
    for _ in range(args.warmup):
        fn(A, k=k, verbose=False)

    times = []
    D = None
    for run in range(args.runs):
        start = time.perf_counter()
        D = fn(A, k=k, verbose=(run == 0))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"#   run {run+1}/{args.runs}: {elapsed:.3f}s")

    best = min(times)
    print(f"# {label} completed: {best:.3f}s (best of {args.runs})")

    if args.output and D is not None:
        if sp.issparse(D):
            D = D.toarray()
        np.savetxt(args.output, D, delimiter=' ', fmt='%g')
        print(f"# Saved to {args.output}")


def run_compare(A, k, args):
    """Benchmark GPU vs CPU methods."""
    import networkx as nx
    from storm_gpu.apsp import gpu_storm_apsp, gpu_storm_apsp_dense, gpu_storm_apsp_fused

    n = A.shape[0]
    results = {}
    D_ref = None

    # ── GPU methods ─────────────────────────────────────────────
    gpu_methods = [
        ('GPU-Fused', gpu_storm_apsp_fused),
        ('GPU-Sparse', gpu_storm_apsp),
    ]
    if n <= 10000:
        gpu_methods.append(('GPU-Dense', gpu_storm_apsp_dense))

    for label, fn in gpu_methods:
        print(f"\n# Running {label}...")
        # Warmup
        fn(A, k=k, verbose=False)
        times = []
        D = None
        for _ in range(args.runs):
            start = time.perf_counter()
            D = fn(A, k=k, verbose=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results[label] = min(times)
        if D_ref is None:
            if sp.issparse(D):
                D_ref = D.toarray().astype(np.int32)
            else:
                D_ref = D.astype(np.int32)
        print(f"#   {label}: {min(times):.3f}s")

    # ── CPU D-STORM (if available) ──────────────────────────────
    storm_dir = os.path.join(os.path.dirname(__file__), '..', '02_STORM_Implement')
    if os.path.isdir(storm_dir):
        sys.path.insert(0, storm_dir)
        try:
            from storm.apsp import storm_apsp
            print(f"\n# Running CPU D-STORM...")
            times = []
            for _ in range(args.runs):
                start = time.perf_counter()
                D_storm = storm_apsp(A, k=k, verbose=False)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            results['CPU-STORM'] = min(times)
            print(f"#   CPU-STORM: {min(times):.3f}s")

            # Verify correctness
            if D_ref is not None:
                D_s = D_storm.toarray().astype(np.int32) if sp.issparse(D_storm) else D_storm.astype(np.int32)
                match = np.array_equal(D_ref, D_s)
                print(f"#   GPU vs CPU correctness: {'PASS' if match else 'MISMATCH'}")
                if not match:
                    diff = np.abs(D_ref - D_s)
                    print(f"#   Max diff: {diff.max()}, mismatched pairs: {np.count_nonzero(diff)}")
        except ImportError as e:
            print(f"#   CPU-STORM not available: {e}")
        finally:
            sys.path.pop(0)

    # ── SciPy ───────────────────────────────────────────────────
    from scipy.sparse.csgraph import shortest_path
    print(f"\n# Running SciPy...")
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        shortest_path(A, directed=False, unweighted=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    results['SciPy'] = min(times)
    print(f"#   SciPy: {min(times):.3f}s")

    # ── NetworkX ────────────────────────────────────────────────
    G = nx.from_scipy_sparse_array(A)
    print(f"\n# Running NetworkX...")
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        if k > 0:
            dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
        else:
            dict(nx.all_pairs_shortest_path_length(G))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    results['NetworkX'] = min(times)
    print(f"#   NetworkX: {min(times):.3f}s")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"# Benchmark (n={n}, m={A.nnz}, k={'full' if k < 0 else k})")
    print(f"# {'Method':<18} {'Time (s)':>10} {'vs NX':>10} {'vs SciPy':>10} {'vs CPU':>10}")
    print(f"# {'-'*58}")

    nx_time = results.get('NetworkX', 1.0)
    scipy_time = results.get('SciPy', 1.0)
    cpu_time = results.get('CPU-STORM', scipy_time)
    fastest = min(results.values())

    for method, t in sorted(results.items(), key=lambda x: x[1]):
        vs_nx = nx_time / t if t > 0 else 0
        vs_scipy = scipy_time / t if t > 0 else 0
        vs_cpu = cpu_time / t if t > 0 else 0
        marker = ' *' if t == fastest else ''
        print(f"# {method:<18} {t:>10.3f} {vs_nx:>9.1f}x {vs_scipy:>9.1f}x {vs_cpu:>9.1f}x{marker}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
