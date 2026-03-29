"""
Grid APSP Visualization: SciPy (exact) vs STORM (incremental k=2,4,8,16,32)
Undiscovered pairs at order k are assigned distance k+1.
"""

import sys, os, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.csgraph import shortest_path as scipy_apsp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_STORM_Implement'))
from storm.core_cython import cython_storm_apsp

OUT = os.path.dirname(__file__)

# === Generate Grid graph ===
grid_size = 20
G = nx.grid_2d_graph(grid_size, grid_size)
G = nx.convert_node_labels_to_integers(G)
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).astype(np.float32).tocsr()
diam = 2 * (grid_size - 1)
print(f"Grid {grid_size}x{grid_size}: n={n}, m={A.nnz}, diameter={diam}")

# === SciPy exact APSP ===
gc.collect()
t0 = time.perf_counter()
D_exact = scipy_apsp(A, directed=False)
t_scipy = time.perf_counter() - t0
print(f"\nSciPy exact: {t_scipy*1000:.2f} ms")

# === STORM at each target k ===
target_ks = [2, 4, 8, 16, 32]
storm_results = {}
storm_times = {}

for k in target_ks:
    gc.collect()
    t0 = time.perf_counter()
    D_k = cython_storm_apsp(A, k=k, verbose=False)
    t_k = time.perf_counter() - t0
    if sp.issparse(D_k):
        D_k = D_k.toarray()
    D_k = np.asarray(D_k, dtype=np.float64)

    # Assign k+1 to undiscovered pairs (excluding diagonal)
    D_filled = D_k.copy()
    undiscovered = (D_filled == 0)
    np.fill_diagonal(undiscovered, False)  # diagonal stays 0
    D_filled[undiscovered] = k + 1

    storm_results[k] = D_filled
    storm_times[k] = t_k

# === Print timing table ===
print(f"\n{'k':>4s}  {'STORM (ms)':>12s}  {'SciPy (ms)':>12s}  {'STORM/SciPy':>12s}  {'Discovered':>12s}  {'MAE':>8s}")
print("-" * 75)
for k in target_ks:
    D_filled = storm_results[k]
    t_k = storm_times[k]
    ratio = t_k / t_scipy

    total_pairs = n * (n - 1)
    discovered = np.sum((D_filled > 0) & (D_filled <= k))
    disc_pct = discovered / total_pairs * 100
    mae = np.mean(np.abs(D_exact - D_filled))

    print(f"{k:4d}  {t_k*1000:12.2f}  {t_scipy*1000:12.2f}  {ratio:12.2f}x  "
          f"{disc_pct:10.1f}%  {mae:8.2f}")

# === Generate images ===
vmax = D_exact.max()

# 1. SciPy exact
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im = ax.imshow(D_exact, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
ax.set_title(f'SciPy Exact APSP\n(d={int(vmax)}, time={t_scipy*1000:.1f}ms)', fontsize=13)
ax.set_xlabel('Node j'); ax.set_ylabel('Node i')
plt.colorbar(im, ax=ax, label='Distance')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grid_scipy_exact.png'), dpi=150)
plt.close()
print("\nSaved: grid_scipy_exact.png")

# 2. Each STORM k (with k+1 fill)
for k in target_ks:
    D_filled = storm_results[k]
    t_k = storm_times[k]
    ratio = t_k / t_scipy

    total_pairs = n * (n - 1)
    discovered = np.sum((D_filled > 0) & (D_filled <= k))
    disc_pct = discovered / total_pairs * 100

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(D_filled, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    ax.set_title(f'STORM k={k}  (undiscovered$\\rightarrow${k+1})\n'
                 f'discovered={disc_pct:.0f}%, '
                 f'time={t_k*1000:.1f}ms ({ratio:.2f}x SciPy)', fontsize=11)
    ax.set_xlabel('Node j'); ax.set_ylabel('Node i')
    plt.colorbar(im, ax=ax, label='Distance')
    plt.tight_layout()
    fname = f'grid_storm_k{k}.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# 3. 2x3 comparison — no colorbar, wider row spacing
fig, axes = plt.subplots(2, 3, figsize=(15, 12),
                         gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
axes = axes.flatten()

panels = [(k, storm_results[k], storm_times[k]) for k in target_ks]
panels.append((diam, D_exact, t_scipy))

for ax, (k, D, t) in zip(axes, panels):
    ax.imshow(D, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    if k == diam:
        label = f'SciPy Exact (d={diam})\n{t*1000:.1f}ms'
    else:
        total_pairs = n * (n - 1)
        disc = np.sum((D > 0) & (D <= k))
        disc_pct = disc / total_pairs * 100
        ratio = t / t_scipy
        label = f'STORM k={k} (→{k+1})\ndisc={disc_pct:.0f}%, {ratio:.2f}x SciPy'
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Node j', fontsize=10)
    ax.set_ylabel('Node i', fontsize=10)

plt.suptitle(f'Grid {grid_size}x{grid_size} (n={n}, d={diam}): '
             f'Undiscovered pairs assigned k+1', fontsize=14, y=0.98)
plt.savefig(os.path.join(OUT, 'grid_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: grid_comparison.png")

# 4. Error images
fig, axes = plt.subplots(1, len(target_ks), figsize=(4*len(target_ks), 4))
for ax, k in zip(axes, target_ks):
    diff = np.abs(D_exact - storm_results[k])
    im = ax.imshow(diff, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)
    zero_pct = (diff == 0).sum() / diff.size * 100
    ax.set_title(f'|Exact - k={k}|\n{zero_pct:.1f}% exact', fontsize=11)
    ax.set_xlabel('Node j', fontsize=9); ax.set_ylabel('Node i', fontsize=9)

plt.colorbar(im, ax=axes, shrink=0.8, label='|Error|')
plt.suptitle(f'Grid {grid_size}x{grid_size}: Approximation Error (undiscovered→k+1)',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 0.93, 0.93])
plt.savefig(os.path.join(OUT, 'grid_error.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: grid_error.png")
