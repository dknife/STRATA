"""
GPU Benchmark: EdgeMan_BFS_G vs EdgeMan_DLevel_G on Facebook graph.

Also compares against CPU Cython versions for reference.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.prepare_data        # first time only
    python -m EdgeManipulation.run_benchmark_gpu
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgeManipulation.EdgeMan_BFS_G.edge_update import (
    insert_edge as gpu_bfs_insert,
    delete_edge as gpu_bfs_delete,
)
from EdgeManipulation.EdgeMan_DLevel_G.edge_update import (
    insert_edge as gpu_dlevel_insert,
    delete_edge as gpu_dlevel_delete,
)
from EdgeManipulation.EdgeMan_BFS.edge_update import (
    insert_edge as cpu_bfs_insert,
    delete_edge as cpu_bfs_delete,
)
from EdgeManipulation.EdgeMan_DLevel.edge_update import (
    insert_edge as cpu_dlevel_insert,
    delete_edge as cpu_dlevel_delete,
)
from EdgeManipulation.EdgeMan_BFS import edge_update as bfs_mod
from EdgeManipulation.EdgeMan_DLevel import edge_update as dlevel_mod

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_data():
    a_path = os.path.join(DATA_DIR, 'facebook_A.npz')
    d_path = os.path.join(DATA_DIR, 'facebook_D.npy')
    if not os.path.exists(a_path) or not os.path.exists(d_path):
        print("Data files not found. Run prepare_data first.")
        sys.exit(1)
    A = sp.load_npz(a_path).tocsr()
    D = np.load(d_path)
    return A, D


def pick_edges_to_delete(A, D, n_edges=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    idx = rng.choice(len(rows), size=min(n_edges, len(rows)), replace=False)
    return list(zip(rows[idx].tolist(), cols[idx].tolist()))


def pick_edges_to_insert(A, D, n_edges=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    n = A.shape[0]
    edges = []
    A_dense = A.toarray()
    while len(edges) < n_edges:
        a = rng.integers(0, n)
        b = rng.integers(0, n)
        if a != b and A_dense[a, b] == 0 and D[a, b] > 1:
            edges.append((min(a, b), max(a, b)))
    return edges


def warmup_gpu():
    """Warmup GPU with a small operation."""
    import cupy as cp
    x = cp.zeros(100)
    cp.cuda.Stream.null.synchronize()


def benchmark_deletion(edges, A_orig, D_orig):
    print(f"\n{'='*78}")
    print(f" Edge Deletion Benchmark ({len(edges)} edges)")
    print(f"{'='*78}")
    print(f" {'edge':<16} | {'CPU-BFS':>10} {'CPU-DLev':>10} | {'GPU-BFS':>10} {'GPU-DLev':>10} | {'aff':>5} {'lev':>4} {'match':>6}")
    print(f" {'-'*16}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*5}-{'-'*4}-{'-'*6}")

    for a, b in edges:
        # CPU BFS
        D_c1 = D_orig.copy()
        A_c1 = A_orig.copy()
        t0 = time.perf_counter()
        A_c1, n1 = cpu_bfs_delete(D_c1, A_c1, a, b)
        t_cpu_bfs = time.perf_counter() - t0

        # CPU DLevel
        D_c2 = D_orig.copy()
        A_c2 = A_orig.copy()
        t0 = time.perf_counter()
        A_c2, n2, lev2 = cpu_dlevel_delete(D_c2, A_c2, a, b)
        t_cpu_dlev = time.perf_counter() - t0

        # GPU BFS
        D_g1 = D_orig.copy()
        A_g1 = A_orig.copy()
        t0 = time.perf_counter()
        A_g1, n3 = gpu_bfs_delete(D_g1, A_g1, a, b)
        t_gpu_bfs = time.perf_counter() - t0

        # GPU DLevel
        D_g2 = D_orig.copy()
        A_g2 = A_orig.copy()
        t0 = time.perf_counter()
        A_g2, n4, lev4 = gpu_dlevel_delete(D_g2, A_g2, a, b)
        t_gpu_dlev = time.perf_counter() - t0

        # Verify all match
        ok = (np.array_equal(D_c1, D_c2) and
              np.array_equal(D_c1, D_g1) and
              np.array_equal(D_c1, D_g2))
        ok_str = "OK" if ok else "FAIL"

        print(f" ({a:>4},{b:>4}) D={D_orig[a,b]} | "
              f"{t_cpu_bfs:>9.4f}s {t_cpu_dlev:>9.4f}s | "
              f"{t_gpu_bfs:>9.4f}s {t_gpu_dlev:>9.4f}s | "
              f"{n1:>5} {lev2:>4} {ok_str:>6}")


def benchmark_insertion(edges, A_orig, D_orig):
    print(f"\n{'='*78}")
    print(f" Edge Insertion Benchmark ({len(edges)} edges)")
    print(f"{'='*78}")
    print(f" {'edge':<16} | {'CPU':>10} | {'GPU':>10} | {'changed':>10} {'match':>6}")
    print(f" {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-{'-'*6}")

    for a, b in edges:
        # CPU (BFS version, same formula)
        D_c = D_orig.copy()
        A_c = A_orig.copy()
        t0 = time.perf_counter()
        A_c = cpu_bfs_insert(D_c, A_c, a, b)
        t_cpu = time.perf_counter() - t0

        # GPU
        D_g = D_orig.copy()
        A_g = A_orig.copy()
        t0 = time.perf_counter()
        A_g = gpu_bfs_insert(D_g, A_g, a, b)
        t_gpu = time.perf_counter() - t0

        ok = np.array_equal(D_c, D_g)
        n_changed = np.sum(D_c != D_orig)

        print(f" ({a:>4},{b:>4}) D={D_orig[a,b]} | "
              f"{t_cpu:>9.4f}s | "
              f"{t_gpu:>9.4f}s | "
              f"{n_changed:>10} {'OK' if ok else 'FAIL':>6}")


def main():
    print("Loading Facebook graph data...")
    A, D = load_data()
    n = A.shape[0]
    print(f"  n={n}, edges={A.nnz//2}, diameter={D.max()}")
    print(f"  CPU BFS backend:    {'Cython' if bfs_mod._USE_CYTHON else 'Python'}")
    print(f"  CPU DLevel backend: {'Cython' if dlevel_mod._USE_CYTHON else 'Python'}")
    print(f"  GPU backends:       CuPy + CUDA kernels")

    warmup_gpu()
    print("  GPU warmup done.")

    rng = np.random.default_rng(42)
    del_edges = pick_edges_to_delete(A, D, n_edges=10, rng=rng)
    ins_edges = pick_edges_to_insert(A, D, n_edges=10, rng=np.random.default_rng(123))

    benchmark_insertion(ins_edges, A, D)
    benchmark_deletion(del_edges, A, D)

    print(f"\n{'='*78}")
    print(" Done.")


if __name__ == '__main__':
    main()
