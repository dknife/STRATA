"""
STORM Final Benchmark — Clean single-run experiment suite.
All results saved to storm_results.json.
"""
import sys, os, time, json, gc
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.csgraph import shortest_path as scipy_apsp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_STORM_Implement'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AORM-main'))

from storm.core_cython import cython_storm_apsp
from storm.loader import load_graph
from storm.dynamic import DynamicStorm
from apsp import apsp_AormIterator as aorm_apsp

BASE = os.path.join(os.path.dirname(__file__), '..', 'AORM-main', 'datasets')
R = {}

def bench(fn, repeat=3):
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, round(min(times), 4)

# ================================================================
# Exp 1: Facebook — 5-method absolute comparison
# ================================================================
print("=" * 60)
print("  Exp 1: Facebook (n=4039)")
print("=" * 60)

A_fb, n_fb, m_fb = load_graph(
    os.path.join(BASE, 'real-world', 'facebook_combined.txt'),
    fmt='edgelist', directed=False)
A_fb_dense = A_fb.toarray().astype(np.float64)
G_fb = nx.from_scipy_sparse_array(A_fb)

fb = {'n': n_fb, 'm': m_fb}
methods_fb = [
    ('STORM',  lambda: cython_storm_apsp(A_fb, verbose=False)),
    ('I-AORM', lambda: aorm_apsp(A_fb_dense.copy(), k=-1, method='edge')),
    ('M-AORM', lambda: aorm_apsp(A_fb_dense.copy(), k=-1, method='matmult')),
    ('SciPy',  lambda: scipy_apsp(A_fb, directed=False)),
    ('NX',     lambda: dict(nx.all_pairs_shortest_path_length(G_fb))),
]
D_ref = None
for name, fn in methods_fb:
    result, t = bench(fn)
    fb[name] = t
    if name == 'STORM':
        D_ref = result.toarray() if sp.issparse(result) else np.asarray(result)
        fb['diameter'] = int(D_ref.max())
    print(f"  {name:8s}: {t:.4f}s")

# Correctness check
D_nx = np.zeros((n_fb, n_fb))
nx_paths = dict(nx.all_pairs_shortest_path_length(G_fb))
for s, dists in nx_paths.items():
    for t_node, d in dists.items():
        D_nx[s, t_node] = d
fb['correct'] = bool(np.allclose(D_ref, D_nx))
print(f"  Correct: {fb['correct']}")
R['exp1_facebook'] = fb

# ================================================================
# Exp 2: BA Scalability
# ================================================================
print("\n" + "=" * 60)
print("  Exp 2: BA Scalability")
print("=" * 60)

ba = []
for ns in [500, 1000, 2000, 3000, 5000]:
    G = nx.barabasi_albert_graph(ns, 5, seed=42)
    Ab = nx.adjacency_matrix(G).astype(np.float32).tocsr()
    Ad = Ab.toarray().astype(np.float64)
    e = {'n': ns, 'm': Ab.nnz}

    _, e['STORM'] = bench(lambda: cython_storm_apsp(Ab, verbose=False))
    if ns <= 3000:
        _, e['I-AORM'] = bench(lambda: aorm_apsp(Ad.copy(), k=-1, method='edge'))
    else:
        e['I-AORM'] = None
    _, e['SciPy'] = bench(lambda: scipy_apsp(Ab, directed=False))
    _, e['NX'] = bench(lambda: dict(nx.all_pairs_shortest_path_length(G)))

    ia = f"{e['I-AORM']:.4f}" if e['I-AORM'] else "---"
    print(f"  n={ns:5d}: STORM={e['STORM']:.4f}  I-AORM={ia:>8s}  "
          f"SciPy={e['SciPy']:.4f}  NX={e['NX']:.4f}")
    ba.append(e)
R['exp2_ba_scale'] = ba

# ================================================================
# Exp 3: Topology comparison
# ================================================================
print("\n" + "=" * 60)
print("  Exp 3: Topologies (n~2000)")
print("=" * 60)

topo = []
configs = [
    ('BA',   lambda: nx.barabasi_albert_graph(2000, 5, seed=42)),
    ('ER',   lambda: nx.erdos_renyi_graph(2000, 0.005, seed=42)),
    ('WS',   lambda: nx.watts_strogatz_graph(2000, 6, 0.1, seed=42)),
    ('Grid', lambda: nx.grid_2d_graph(45, 45)),
    ('PLC',  lambda: nx.powerlaw_cluster_graph(2000, 5, 0.3, seed=42)),
]
for name, gen_fn in configs:
    G = gen_fn()
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    Ab = nx.adjacency_matrix(G).astype(np.float32).tocsr()
    Ad = Ab.toarray().astype(np.float64)
    e = {'topo': name, 'n': G.number_of_nodes(), 'm': Ab.nnz}

    D, e['STORM'] = bench(lambda: cython_storm_apsp(Ab, verbose=False))
    e['diam'] = int(D.max()) if sp.issparse(D) else int(np.max(D))
    _, e['I-AORM'] = bench(lambda: aorm_apsp(Ad.copy(), k=-1, method='edge'))
    _, e['SciPy'] = bench(lambda: scipy_apsp(Ab, directed=False))
    _, e['NX'] = bench(lambda: dict(nx.all_pairs_shortest_path_length(G)))

    print(f"  {name:5s} d={e['diam']:3d}: STORM={e['STORM']:.4f}  "
          f"I-AORM={e['I-AORM']:.4f}  SciPy={e['SciPy']:.4f}  NX={e['NX']:.4f}")
    topo.append(e)
R['exp3_topologies'] = topo

# ================================================================
# Exp 4: Hop-constrained APSP
# ================================================================
print("\n" + "=" * 60)
print("  Exp 4: Hop-constrained (Facebook)")
print("=" * 60)

hop = []
for k in range(2, 9):
    e = {'k': k}
    _, e['STORM'] = bench(lambda k=k: cython_storm_apsp(A_fb, k=k, verbose=False))
    _, e['I-AORM'] = bench(lambda k=k: aorm_apsp(A_fb_dense.copy(), k=k, method='edge'))
    _, e['NX'] = bench(lambda k=k: dict(nx.all_pairs_shortest_path_length(G_fb, cutoff=k)))
    print(f"  k={k}: STORM={e['STORM']:.4f}  I-AORM={e['I-AORM']:.4f}  NX={e['NX']:.4f}")
    hop.append(e)
R['exp4_hop'] = hop

# ================================================================
# Exp 5: Progressive sparsity
# ================================================================
print("\n" + "=" * 60)
print("  Exp 5: Progressive sparsity (Facebook)")
print("=" * 60)

from storm.core_optimized import OptimizedStormIterator
A_tmp = A_fb.astype(np.int8).copy()
A_tmp.data[:] = 1
it = OptimizedStormIterator(A_tmp, k=-1)
Rk_init = it.Rk
total_pairs = n_fb * (n_fb - 1)
cumul_nnz = Rk_init.nnz
cumul_dist = float(Rk_init.sum())

sparsity = [{'k': 1, 'nnz_Rk': Rk_init.nnz, 'cumul_nnz': cumul_nnz,
             'correctness': round(cumul_dist / 60222872, 6)}]
print(f"  k=1: nnz_Rk={Rk_init.nnz}")

for Rk_star, power in it:
    rk_nnz = Rk_star.nnz
    cumul_nnz += rk_nnz
    cumul_dist += float(power * rk_nnz)
    c = round(cumul_dist / 60222872, 6)
    sparsity.append({'k': power, 'nnz_Rk': rk_nnz, 'cumul_nnz': cumul_nnz,
                     'correctness': c})
    print(f"  k={power}: nnz_Rk={rk_nnz:>10d}  cumul={cumul_nnz:>10d}  C={c:.4f}")
R['exp5_sparsity'] = sparsity

# ================================================================
# Exp 6: D-STORM dynamic
# ================================================================
print("\n" + "=" * 60)
print("  Exp 6: D-STORM")
print("=" * 60)

rng = np.random.RandomState(42)
dyn = []
for ns in [500, 1000, 2000]:
    G = nx.barabasi_albert_graph(ns, 5, seed=42)
    Ab = nx.adjacency_matrix(G).astype(np.float32).tocsr()
    D_init = cython_storm_apsp(Ab, verbose=False)
    ds = DynamicStorm(Ab.copy(), D_init)

    add_times = []
    add_updates = []
    for _ in range(30):
        u, v = rng.randint(0, ns, size=2)
        while u == v or ds.A[u, v] != 0:
            u, v = rng.randint(0, ns, size=2)
        gc.collect()
        t0 = time.perf_counter()
        n_upd = ds.add_edge(u, v)
        add_times.append(time.perf_counter() - t0)
        add_updates.append(n_upd)

    _, full_t = bench(lambda: cython_storm_apsp(ds.A, verbose=False))

    avg_add = round(np.mean(add_times) * 1000, 3)
    avg_upd = round(np.mean(add_updates), 1)
    speedup = round(full_t / np.mean(add_times), 1)

    e = {'n': ns, 'add_avg_ms': avg_add, 'add_avg_pairs': avg_upd,
         'full_ms': round(full_t * 1000, 1), 'speedup': speedup}
    dyn.append(e)
    print(f"  n={ns}: ADD={avg_add:.2f}ms  pairs={avg_upd:.0f}  "
          f"full={full_t*1000:.1f}ms  speedup={speedup:.0f}x")
R['exp6_dynamic'] = dyn

# ================================================================
# Save
# ================================================================
outpath = os.path.join(os.path.dirname(__file__), 'storm_results.json')
with open(outpath, 'w') as f:
    json.dump(R, f, indent=2, default=str)
print(f"\n{'='*60}")
print(f"  All results saved to {outpath}")
print(f"{'='*60}")
