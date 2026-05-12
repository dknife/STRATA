"""
Unified Benchmark Runner — All 12 APSP Implementations.

Usage:
    python run_all.py -i graph.txt -r                # all methods
    python run_all.py -i graph.txt -r --cpu-only      # CPU methods only
    python run_all.py -i graph.txt -r --gpu-only      # GPU methods only
    python run_all.py -i graph.txt -r --skip BC5 TC3  # skip GraphBLAS
"""

import argparse
import importlib
import time
import sys
import os
import numpy as np
import scipy.sparse as sp

# Add parent paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.loader import load_graph, make_undirected, is_connected


# ── Method registry ──────────────────────────────────────────

METHODS = [
    # (ID, label, module_path, requires_gpu, requires_graphblas)
    ('BC1', 'NetworkX',           'BC1_NetworkX.apsp',        False, False),
    ('BC2', 'SciPy',              'BC2_SciPy.apsp',           False, False),
    ('BC3', 'I-AORM',             'BC3_IAORM.apsp',           False, False),
    ('BC4', 'M-AORM',             'BC4_MAORM.apsp',           False, False),
    ('BC5', 'GB-bfs',             'BC5_GB_bfs.apsp',          False, True),
    ('TC1', 'STRATA-SpMM-Cython', 'TC1_STRATA_SpMM_Cython.apsp', False, False),
    ('TC2', 'STRATA-NumpyBLAS',  'TC2_STRATA_NumpyBLAS.apsp',  False, False),
    ('TC3', 'STRATA-GraphBLAS',   'TC3_STRATA_GraphBLAS.apsp', False, True),
    ('BG1', 'GPU-PerSrc-BFS',     'BG1_GPU_PerSrc_BFS.apsp', True,  False),
    ('TG1', 'STRATA-cuBLAS',     'TG1_STRATA_cuBLAS.apsp',   True,  False),
    ('TG2', 'STRATA-CUDA',      'TG2_STRATA_CUDA.apsp',     True,  False),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Unified APSP Benchmark")
    parser.add_argument('-i', '--input', required=True, help='Input graph file')
    parser.add_argument('-r', '--realworld', action='store_true')
    parser.add_argument('-k', '--order', type=int, default=-1)
    parser.add_argument('-d', '--directed', action='store_true', default=False)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--skip', nargs='*', default=[], help='Method IDs to skip')
    parser.add_argument('--max-dense-n', type=int, default=10000,
                        help='Skip dense methods above this n')
    return parser.parse_args()


def main():
    args = parse_args()

    fmt = 'edgelist' if args.realworld else None
    A, n, m = load_graph(args.input, fmt=fmt, directed=True)
    if not args.directed:
        A = make_undirected(A)
        m = A.nnz

    print(f"{'='*65}")
    print(f"# Graph [{args.input}]: |V|={n}, |E|={m}, "
          f"avg_deg={m/n:.1f}, density={m/(n*n):.6f}")
    if not is_connected(A):
        print("# Warning: graph is disconnected")
    print(f"{'='*65}")

    k = args.order if args.order > 0 else -1
    results = {}
    D_ref = None

    for mid, label, module_path, needs_gpu, needs_gb in METHODS:
        # Filter
        if mid in args.skip:
            continue
        if args.cpu_only and needs_gpu:
            continue
        if args.gpu_only and not needs_gpu:
            continue

        # Skip dense methods for large graphs
        if mid in ('BC3', 'BC4', 'TC2', 'TG1') and n > args.max_dense_n:
            print(f"\n# [{mid}] {label} — SKIPPED (n={n} > {args.max_dense_n})")
            continue

        # Try importing
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            print(f"\n# [{mid}] {label} — IMPORT ERROR: {e}")
            continue

        print(f"\n# [{mid}] {label}")

        # Warmup (GPU only)
        warmup = args.warmup if needs_gpu else 0
        try:
            for _ in range(warmup):
                mod.run_apsp(A, k=k, verbose=False)
        except Exception as e:
            print(f"  WARMUP ERROR: {e}")
            continue

        # Timed runs
        times = []
        D = None
        try:
            for run in range(args.runs):
                start = time.perf_counter()
                D = mod.run_apsp(A, k=k, verbose=False)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  run {run+1}/{args.runs}: {elapsed:.4f}s")
        except Exception as e:
            print(f"  RUNTIME ERROR: {e}")
            continue

        best = min(times)
        results[mid] = (label, best)

        # Correctness check vs first result
        if D is not None:
            D_int = D.astype(np.int32) if D.dtype != np.int32 else D
            if D_ref is None:
                D_ref = D_int
                print(f"  best: {best:.4f}s (reference)")
            else:
                match = np.array_equal(D_ref, D_int)
                print(f"  best: {best:.4f}s ({'PASS' if match else 'MISMATCH'})")

    # ── Summary ──────────────────────────────────────────────
    if not results:
        print("\nNo methods ran successfully.")
        return

    scipy_t = results.get('BC2', (None, None))[1]
    if scipy_t is None:
        scipy_t = list(results.values())[0][1]

    print(f"\n{'='*65}")
    print(f"# Summary (n={n}, m={m}, k={'full' if k < 0 else k})")
    print(f"# {'ID':<4} {'Method':<22} {'Time (s)':>10} {'vs SciPy':>10}")
    print(f"# {'-'*48}")

    fastest = min(v[1] for v in results.values())
    for mid, (label, t) in sorted(results.items(), key=lambda x: x[1][1]):
        ratio = scipy_t / t if t > 0 else 0
        marker = ' *' if t == fastest else ''
        print(f"# {mid:<4} {label:<22} {t:>10.4f} {ratio:>9.1f}x{marker}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
