"""
Wall-clock benchmark: BFS baseline vs DLevel current vs DLevel+local rebuild.

Replicates the |S_aff| partitioning of Table tab:deletion in the paper:
  - typical: |S_aff| = 2-3
  - medium:  |S_aff| ~ 12
  - worst:   |S_aff| ~ 2407
on the Facebook graph (n=4039).

Each variant is timed over multiple runs (min of N).

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.bench_local_rebuild
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgeManipulation.EdgeMan_BFS import edge_update as bfs_mod
from EdgeManipulation.EdgeMan_DLevel import edge_update as dlevel_mod

REPEATS = 5

# Cython is used for all three variants now (fair, same-language comparison).
print(f"BFS backend:    {'Cython' if bfs_mod._USE_CYTHON else 'Python'}")
print(f"DLevel backend: {'Cython' if dlevel_mod._USE_CYTHON else 'Python'}")
print(f"DLevel+LocalRebuild backend: "
      f"{'Cython' if dlevel_mod._USE_CYTHON_LOCAL else 'Python'}")


def bench_one(D_orig, A_orig, a, b, label=""):
    """Time the three variants over REPEATS runs, taking the minimum."""
    n_aff_check = None

    times_bfs = []
    for _ in range(REPEATS):
        D = D_orig.copy()
        A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff = bfs_mod.delete_edge(D, A, a, b)
        times_bfs.append(time.perf_counter() - t0)
        n_aff_check = n_aff
    t_bfs = min(times_bfs)

    times_dl = []
    for _ in range(REPEATS):
        D = D_orig.copy()
        A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff, _ = dlevel_mod.delete_edge(D, A, a, b)
        times_dl.append(time.perf_counter() - t0)
    t_dl = min(times_dl)

    times_lr = []
    for _ in range(REPEATS):
        D = D_orig.copy()
        A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff, _ = dlevel_mod.delete_edge_local(D, A, a, b)
        times_lr.append(time.perf_counter() - t0)
    t_lr = min(times_lr)

    sp_bfs_lr = t_bfs / t_lr if t_lr > 0 else float('inf')
    sp_dl_lr = t_dl / t_lr if t_lr > 0 else float('inf')

    print(f"  [{label}] edge=({a:5d},{b:5d}) |S_aff|={n_aff_check:5d}:  "
          f"BFS={t_bfs*1000:7.2f}ms  DLevel={t_dl*1000:7.2f}ms  "
          f"LocalRebuild={t_lr*1000:7.2f}ms  "
          f"speedup(LR vs BFS)={sp_bfs_lr:6.2f}x  speedup(LR vs DL)={sp_dl_lr:6.2f}x")

    return n_aff_check, t_bfs, t_dl, t_lr


def find_edges_by_saff(A, D, rng, target_saff_range, n_samples=40, n_take=3):
    """Sample edges and find ones whose deletion gives |S_aff| in target_saff_range."""
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    candidates = rng.choice(len(rows), size=n_samples, replace=False)

    found = []
    low, high = target_saff_range
    for idx in candidates:
        u, v = int(rows[idx]), int(cols[idx])
        D_tmp = D.copy()
        A_tmp = A.copy()
        _, n_aff = bfs_mod.delete_edge(D_tmp, A_tmp, u, v)
        if low <= n_aff <= high:
            found.append((u, v, n_aff))
            if len(found) >= n_take:
                break
    return found


def main():
    a_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_A.npz')
    d_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_D.npy')

    A = sp.load_npz(a_path).tocsr()
    D = np.load(d_path)
    n = A.shape[0]
    print(f"Facebook graph: n={n}, m={A.nnz // 2}, diameter={D.max()}")
    print(f"Repeats per timing: {REPEATS} (reporting min)")
    print()

    rng = np.random.default_rng(42)

    # Use the original 10 random edges (same as run_benchmark / Table tab:deletion)
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    idx = rng.choice(len(rows), size=10, replace=False)
    test_edges_orig = list(zip(rows[idx].tolist(), cols[idx].tolist()))

    print("=" * 100)
    print(" Original 10 random edges (same as Table tab:deletion in paper)")
    print("=" * 100)

    results = []
    for u, v in test_edges_orig:
        results.append(bench_one(D, A, u, v, label="orig"))

    print()
    print("=" * 100)
    print(" Summary by |S_aff| bucket")
    print("=" * 100)

    # Partition by |S_aff|
    buckets = {
        "typical (|S_aff| in 2-3)": [],
        "medium  (|S_aff| in 5-50)": [],
        "worst   (|S_aff| > 100)": [],
    }
    for n_aff, t_bfs, t_dl, t_lr in results:
        if 2 <= n_aff <= 3:
            key = "typical (|S_aff| in 2-3)"
        elif 5 <= n_aff <= 50:
            key = "medium  (|S_aff| in 5-50)"
        elif n_aff > 100:
            key = "worst   (|S_aff| > 100)"
        else:
            continue
        buckets[key].append((n_aff, t_bfs, t_dl, t_lr))

    for label, items in buckets.items():
        if not items:
            print(f"  {label}: no samples")
            continue
        avg_bfs = np.mean([x[1] for x in items])
        avg_dl = np.mean([x[2] for x in items])
        avg_lr = np.mean([x[3] for x in items])
        speedup_vs_bfs = avg_bfs / avg_lr if avg_lr > 0 else float('inf')
        speedup_vs_dl = avg_dl / avg_lr if avg_lr > 0 else float('inf')
        print(f"  {label}  n_cases={len(items)}:")
        print(f"    avg BFS={avg_bfs*1000:7.2f}ms  avg DLevel={avg_dl*1000:7.2f}ms  "
              f"avg LocalRebuild={avg_lr*1000:7.2f}ms")
        print(f"    LocalRebuild speedup: {speedup_vs_bfs:.2f}x (vs BFS), "
              f"{speedup_vs_dl:.2f}x (vs DLevel)")


if __name__ == '__main__':
    main()
