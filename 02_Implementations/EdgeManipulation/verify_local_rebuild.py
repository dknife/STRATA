"""
Correctness verification for delete_edge_local (Phase-2-driven local rebuild).

For each test edge, deletes it via three methods and checks that all three
produce element-wise identical distance matrices:
  1. BFS baseline (delete_edge from EdgeMan_BFS)
  2. DLevel current (delete_edge from EdgeMan_DLevel; Phase 1+2 instrumentation + Phase 3 full BFS)
  3. DLevel + local rebuild (delete_edge_local; Phase 1+2 + Phase 3 local rebuild)

Also cross-checks against scipy.sparse.csgraph.shortest_path on G'.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.verify_local_rebuild
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgeManipulation.EdgeMan_BFS import edge_update as bfs_mod
from EdgeManipulation.EdgeMan_DLevel import edge_update as dlevel_mod


def compute_D_unweighted(A_csr):
    """Full APSP via SciPy; encode unreachable as 0 to match the D=0 overload."""
    D = shortest_path(A_csr, directed=False, unweighted=True)
    D[np.isinf(D)] = 0
    return D.astype(np.int32)


def small_synthetic_graph():
    """
    A small graph where deleting a single edge changes distances for several pairs.
    Vertices 0..7, with one 'bridge' edge whose deletion forces detours.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 4),
        (2, 6),  # the bridge
    ]
    n = 8
    rows = []
    cols = []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    data = np.ones(len(rows), dtype=np.int32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A, edges


def verify_one_deletion(A_orig, D_orig, a, b, label=""):
    """Run all three deletion variants on (a,b) and compare."""
    # Method 1: BFS baseline
    D_bfs = D_orig.copy()
    A_bfs = A_orig.copy()
    A_bfs, n_aff_bfs = bfs_mod.delete_edge(D_bfs, A_bfs, a, b)

    # Method 2: DLevel current (full BFS in Phase 3)
    D_dl = D_orig.copy()
    A_dl = A_orig.copy()
    A_dl, n_aff_dl, levels_dl = dlevel_mod.delete_edge(D_dl, A_dl, a, b)

    # Method 3: DLevel + local rebuild (new)
    D_lr = D_orig.copy()
    A_lr = A_orig.copy()
    A_lr, n_aff_lr, levels_lr = dlevel_mod.delete_edge_local(D_lr, A_lr, a, b)

    # Cross-check against full APSP recompute on A'
    D_ref = compute_D_unweighted(A_lr)

    match_bfs = np.array_equal(D_bfs, D_ref)
    match_dl = np.array_equal(D_dl, D_ref)
    match_lr = np.array_equal(D_lr, D_ref)

    status = "OK" if (match_bfs and match_dl and match_lr) else "FAIL"
    print(f"  {label} delete ({a},{b}) D[a,b]={D_orig[a, b]}: "
          f"BFS={'OK' if match_bfs else 'X'} "
          f"DLevel={'OK' if match_dl else 'X'} "
          f"LocalRebuild={'OK' if match_lr else 'X'} "
          f"|S_aff|=BFS{n_aff_bfs}/DL{n_aff_dl}/LR{n_aff_lr}  [{status}]")

    if not match_lr:
        diff_idx = np.argwhere(D_lr != D_ref)
        print(f"    LR vs ref differs at {len(diff_idx)} entries; first 5:")
        for i, j in diff_idx[:5]:
            print(f"      ({i},{j}): LR={D_lr[i, j]}, ref={D_ref[i, j]}")
    return match_bfs and match_dl and match_lr


def main():
    print("=" * 70)
    print(" 1. Small synthetic graph (n=8, with bridge edge)")
    print("=" * 70)
    A, edges = small_synthetic_graph()
    D = compute_D_unweighted(A)
    n_pass = 0
    for u, v in edges:
        if verify_one_deletion(A, D, u, v, label="[small]"):
            n_pass += 1
    print(f"  Small graph: {n_pass}/{len(edges)} passed")
    print()

    print("=" * 70)
    print(" 2. Facebook graph (n=4039)")
    print("=" * 70)
    a_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_A.npz')
    d_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_D.npy')
    if not os.path.exists(a_path):
        print("  Facebook data not found; run prepare_data first. Skipping.")
        return

    A = sp.load_npz(a_path).tocsr()
    D = np.load(d_path)
    n = A.shape[0]
    print(f"  loaded n={n}, edges={A.nnz // 2}, diameter={D.max()}")

    # Pick the same 10 edges as run_benchmark
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows), size=10, replace=False)
    test_edges = list(zip(rows[idx].tolist(), cols[idx].tolist()))

    n_pass = 0
    for u, v in test_edges:
        if verify_one_deletion(A, D, u, v, label="[fb]"):
            n_pass += 1
    print(f"  Facebook: {n_pass}/{len(test_edges)} passed")


if __name__ == '__main__':
    main()
