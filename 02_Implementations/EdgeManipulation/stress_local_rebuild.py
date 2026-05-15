"""
Stress test: many random deletions, collect timing + |S_aff| distribution.

Each random edge is deleted via BFS / DLevel / DLevel+LocalRebuild;
all three should produce the same D (verified for a sample);
timing is the min of REPEATS runs.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.stress_local_rebuild
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EdgeManipulation.EdgeMan_BFS import edge_update as bfs_mod
from EdgeManipulation.EdgeMan_DLevel import edge_update as dl_mod

REPEATS = 3
N_EDGES = 60


def bench_one(D_orig, A_orig, a, b, diameter):
    """Return (n_aff, t_bfs, t_dl, t_lr) min times across REPEATS."""
    t_bfs_list = []
    for _ in range(REPEATS):
        D = D_orig.copy(); A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff = bfs_mod.delete_edge(D, A, a, b)
        t_bfs_list.append(time.perf_counter() - t0)

    t_dl_list = []
    for _ in range(REPEATS):
        D = D_orig.copy(); A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff_dl, _ = dl_mod.delete_edge(D, A, a, b)
        t_dl_list.append(time.perf_counter() - t0)

    t_lr_list = []
    for _ in range(REPEATS):
        D = D_orig.copy(); A = A_orig.copy()
        t0 = time.perf_counter()
        A, n_aff_lr, _ = dl_mod.delete_edge_local(D, A, a, b, diameter=diameter)
        t_lr_list.append(time.perf_counter() - t0)

    return n_aff, min(t_bfs_list), min(t_dl_list), min(t_lr_list)


def main():
    a_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_A.npz')
    d_path = os.path.join(os.path.dirname(__file__), 'data', 'facebook_D.npy')
    A = sp.load_npz(a_path).tocsr()
    D = np.load(d_path)
    n = A.shape[0]
    diameter = int(D.max())
    print(f"Facebook: n={n}, m={A.nnz // 2}, diameter={diameter}")
    print(f"Repeats: {REPEATS}, N_EDGES: {N_EDGES}")

    rng = np.random.default_rng(2026)
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    rows = A_coo.row[mask]
    cols = A_coo.col[mask]
    idx = rng.choice(len(rows), size=N_EDGES, replace=False)
    edges = list(zip(rows[idx].tolist(), cols[idx].tolist()))

    # Warmup
    for u, v in edges[:3]:
        bench_one(D, A, u, v, diameter)

    results = []  # (n_aff, t_bfs, t_dl, t_lr)
    for i, (u, v) in enumerate(edges):
        r = bench_one(D, A, u, v, diameter)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_EDGES}] last edge ({u},{v}) |S_aff|={r[0]} "
                  f"BFS={r[1]*1000:.1f}ms LR={r[3]*1000:.1f}ms")

    # Bucket by |S_aff|
    buckets = {
        "  tiny  (|S_aff|=1)   ": [],
        "  light (|S_aff|=2-3) ": [],
        "  medium (|S_aff|=4-50)": [],
        "  large  (|S_aff|=51-500)": [],
        "  huge   (|S_aff|>500)": [],
    }
    for n_aff, t_bfs, t_dl, t_lr in results:
        if n_aff == 1:
            buckets["  tiny  (|S_aff|=1)   "].append((n_aff, t_bfs, t_dl, t_lr))
        elif n_aff <= 3:
            buckets["  light (|S_aff|=2-3) "].append((n_aff, t_bfs, t_dl, t_lr))
        elif n_aff <= 50:
            buckets["  medium (|S_aff|=4-50)"].append((n_aff, t_bfs, t_dl, t_lr))
        elif n_aff <= 500:
            buckets["  large  (|S_aff|=51-500)"].append((n_aff, t_bfs, t_dl, t_lr))
        else:
            buckets["  huge   (|S_aff|>500)"].append((n_aff, t_bfs, t_dl, t_lr))

    print()
    print("=" * 110)
    print(f"{'bucket':28s} {'n':>4s} {'med |S|':>8s} {'BFS (ms)':>10s} {'DLevel (ms)':>12s} "
          f"{'LR (ms)':>10s} {'LR/BFS':>8s} {'LR/DL':>8s}")
    print("=" * 110)

    all_results = results
    for label, items in buckets.items():
        if not items:
            print(f"{label:28s} (no samples)")
            continue
        med_saff = sorted([x[0] for x in items])[len(items) // 2]
        med_bfs = sorted([x[1] for x in items])[len(items) // 2]
        med_dl = sorted([x[2] for x in items])[len(items) // 2]
        med_lr = sorted([x[3] for x in items])[len(items) // 2]
        sp_bfs = med_bfs / med_lr if med_lr > 0 else 0
        sp_dl = med_dl / med_lr if med_lr > 0 else 0
        print(f"{label:28s} {len(items):>4d} {med_saff:>8d} {med_bfs*1000:>10.2f} "
              f"{med_dl*1000:>12.2f} {med_lr*1000:>10.2f} {sp_bfs:>8.2f} {sp_dl:>8.2f}")

    # Overall aggregate
    all_bfs = sum(x[1] for x in all_results)
    all_dl = sum(x[2] for x in all_results)
    all_lr = sum(x[3] for x in all_results)
    print("-" * 110)
    print(f"{'TOTAL (sum of mins)':28s} {len(all_results):>4d} {'-':>8s} {all_bfs*1000:>10.2f} "
          f"{all_dl*1000:>12.2f} {all_lr*1000:>10.2f} {all_bfs/all_lr:>8.2f} {all_dl/all_lr:>8.2f}")


if __name__ == '__main__':
    main()
