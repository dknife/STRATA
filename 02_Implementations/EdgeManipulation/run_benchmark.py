"""
Benchmark: EdgeMan_BFS vs EdgeMan_DLevel on Facebook graph.

Compares correctness and performance of edge insertion and deletion
using source-based (BFS) and level-based (DLevel) approaches.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.prepare_data      # first time only
    python -m EdgeManipulation.run_benchmark
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgeManipulation.EdgeMan_BFS import edge_update as bfs_mod
from EdgeManipulation.EdgeMan_DLevel import edge_update as dlevel_mod

bfs_insert = bfs_mod.insert_edge
bfs_delete = bfs_mod.delete_edge

from EdgeManipulation.EdgeMan_DLevel.edge_update import (
    insert_edge as dlevel_insert,
    delete_edge as dlevel_delete,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_data():
    a_path = os.path.join(DATA_DIR, 'facebook_A.npz')
    d_path = os.path.join(DATA_DIR, 'facebook_D.npy')

    if not os.path.exists(a_path) or not os.path.exists(d_path):
        print("Data files not found. Run prepare_data first:")
        print("  python -m EdgeManipulation.prepare_data")
        sys.exit(1)

    A = sp.load_npz(a_path).tocsr()
    D = np.load(d_path)
    return A, D


def pick_edges_to_delete(A, D, n_edges=10, rng=None):
    """Pick random existing edges that are on some shortest paths."""
    if rng is None:
        rng = np.random.default_rng(42)
    A_coo = A.tocoo()
    # Only pick upper-triangle edges (undirected)
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    idx = rng.choice(len(rows), size=min(n_edges, len(rows)), replace=False)
    return list(zip(rows[idx].tolist(), cols[idx].tolist()))


def pick_edges_to_insert(A, D, n_edges=10, rng=None):
    """Pick random non-edges (pairs with D[a,b] > 1)."""
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


def verify_against_full_bfs(D, A_csr, label=""):
    """Verify D by recomputing APSP with SciPy."""
    from scipy.sparse.csgraph import shortest_path
    D_ref = shortest_path(A_csr, directed=False, unweighted=True).astype(np.int32)
    D_ref[np.isinf(D_ref.astype(np.float64))] = 0
    match = np.array_equal(D, D_ref)
    if not match:
        diff = np.sum(D != D_ref)
        print(f"  [{label}] MISMATCH: {diff} entries differ")
    return match


def benchmark_deletion(edges, A_orig, D_orig):
    print(f"\n{'='*70}")
    print(f" Edge Deletion Benchmark ({len(edges)} edges)")
    print(f"{'='*70}")

    for idx, (a, b) in enumerate(edges):
        print(f"\n--- Delete edge ({a}, {b}), D[{a},{b}]={D_orig[a,b]} ---")

        # BFS version
        D_bfs = D_orig.copy()
        A_bfs = A_orig.copy()
        t0 = time.perf_counter()
        A_bfs, n_aff_bfs = bfs_delete(D_bfs, A_bfs, a, b)
        t_bfs = time.perf_counter() - t0

        # DLevel version
        D_dlevel = D_orig.copy()
        A_dlevel = A_orig.copy()
        t0 = time.perf_counter()
        A_dlevel, n_aff_dlevel, levels = dlevel_delete(D_dlevel, A_dlevel, a, b)
        t_dlevel = time.perf_counter() - t0

        # Correctness: both should match
        match = np.array_equal(D_bfs, D_dlevel)
        match_str = "OK" if match else "MISMATCH"

        # Verify against full APSP recompute (for first few)
        if idx < 3:
            full_ok = verify_against_full_bfs(D_dlevel, A_dlevel, "full-verify")
            match_str += f", full-verify={'OK' if full_ok else 'FAIL'}"

        print(f"  BFS:    {t_bfs:.4f}s, affected_sources={n_aff_bfs}")
        print(f"  DLevel: {t_dlevel:.4f}s, affected_sources={n_aff_dlevel}, levels_checked={levels}")
        print(f"  Match: {match_str}")
        if n_aff_bfs != n_aff_dlevel:
            print(f"  ** Affected diff: BFS={n_aff_bfs} vs DLevel={n_aff_dlevel}")


def benchmark_insertion(edges, A_orig, D_orig):
    print(f"\n{'='*70}")
    print(f" Edge Insertion Benchmark ({len(edges)} edges)")
    print(f"{'='*70}")

    for idx, (a, b) in enumerate(edges):
        print(f"\n--- Insert edge ({a}, {b}), D[{a},{b}]={D_orig[a,b]} ---")

        # BFS version
        D_bfs = D_orig.copy()
        A_bfs = A_orig.copy()
        t0 = time.perf_counter()
        A_bfs = bfs_insert(D_bfs, A_bfs, a, b)
        t_bfs = time.perf_counter() - t0

        # DLevel version (same formula, should be identical)
        D_dlevel = D_orig.copy()
        A_dlevel = A_orig.copy()
        t0 = time.perf_counter()
        A_dlevel = dlevel_insert(D_dlevel, A_dlevel, a, b)
        t_dlevel = time.perf_counter() - t0

        match = np.array_equal(D_bfs, D_dlevel)

        # Full verification for first few
        if idx < 3:
            full_ok = verify_against_full_bfs(D_dlevel, A_dlevel, "full-verify")
            full_str = f", full-verify={'OK' if full_ok else 'FAIL'}"
        else:
            full_str = ""

        n_changed = np.sum(D_bfs != D_orig)
        print(f"  BFS:    {t_bfs:.4f}s")
        print(f"  DLevel: {t_dlevel:.4f}s")
        print(f"  Match: {'OK' if match else 'MISMATCH'}{full_str}, pairs_changed={n_changed}")


def main():
    print("Loading Facebook graph data...")
    A, D = load_data()
    n = A.shape[0]
    print(f"  n={n}, edges={A.nnz//2}, diameter={D.max()}")
    print(f"  BFS backend:    {'Cython' if bfs_mod._USE_CYTHON else 'Python'}")
    print(f"  DLevel backend: {'Cython' if dlevel_mod._USE_CYTHON else 'Python'}")

    rng = np.random.default_rng(42)
    del_edges = pick_edges_to_delete(A, D, n_edges=10, rng=rng)
    ins_edges = pick_edges_to_insert(A, D, n_edges=10, rng=np.random.default_rng(123))

    benchmark_insertion(ins_edges, A, D)
    benchmark_deletion(del_edges, A, D)

    print(f"\n{'='*70}")
    print(" Done.")


if __name__ == '__main__':
    main()
