"""Verify sigma(s,t) computation against BFS-based Brandes sigma.

Tests TC1, TC2, and (if GPU available) TG2 with compute_sigma=True.
Compares results against NetworkX-style per-source BFS sigma computation.
"""

import sys
import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import deque


def bfs_sigma(A_csr, n):
    """Compute sigma(s,t) via per-source BFS (Brandes forward pass)."""
    sigma = np.zeros((n, n), dtype=np.float64)
    D = np.zeros((n, n), dtype=np.int32)
    np.fill_diagonal(sigma, 1.0)

    indptr = A_csr.indptr
    indices = A_csr.indices

    for s in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        sig = np.zeros(n, dtype=np.float64)
        sig[s] = 1.0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            for idx in range(indptr[v], indptr[v + 1]):
                w = indices[idx]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sig[w] += sig[v]
        D[s] = dist
        D[s, dist < 0] = 0  # unreachable → 0
        sigma[s] = sig

    return D, sigma


def test_method(name, run_apsp, A_csr, D_ref, sigma_ref):
    """Test a single method's sigma computation."""
    print(f"\n--- {name} ---")
    try:
        result = run_apsp(A_csr, compute_sigma=True, verbose=False)
        D, sigma = result
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    # Check D
    d_ok = np.array_equal(D, D_ref)
    print(f"  D matrix: {'PASS' if d_ok else 'FAIL'}")
    if not d_ok:
        diff = np.sum(D != D_ref)
        print(f"    {diff} mismatched entries")
        return False

    # Check sigma
    s_ok = np.allclose(sigma, sigma_ref, rtol=1e-10)
    print(f"  sigma matrix: {'PASS' if s_ok else 'FAIL'}")
    if not s_ok:
        diff_mask = ~np.isclose(sigma, sigma_ref, rtol=1e-10)
        n_diff = np.sum(diff_mask)
        print(f"    {n_diff} mismatched entries")
        idx = np.argwhere(diff_mask)
        for ii in range(min(5, len(idx))):
            r, c = idx[ii]
            print(f"    [{r},{c}]: got {sigma[r,c]}, expected {sigma_ref[r,c]}")
        return False

    print(f"  sigma stats: mean={sigma[sigma>0].mean():.1f}, "
          f"max={sigma.max():.0f}")
    return True


def main():
    # Generate test graphs
    graphs = {
        'BA-500': nx.barabasi_albert_graph(500, 5, seed=42),
        'WS-500': nx.watts_strogatz_graph(500, 6, 0.3, seed=42),
    }

    all_pass = True
    for gname, G in graphs.items():
        print(f"\n{'='*50}")
        print(f"Graph: {gname} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
        print(f"{'='*50}")

        A = nx.adjacency_matrix(G).astype(np.float32).tocsr()
        n = A.shape[0]

        # Reference: BFS sigma
        print("Computing BFS reference sigma...")
        D_ref, sigma_ref = bfs_sigma(A, n)
        print(f"  D diameter={D_ref.max()}, "
              f"sigma mean={sigma_ref[sigma_ref>0].mean():.1f}, "
              f"max={sigma_ref.max():.0f}")

        # Also verify D without sigma (backward compatibility)
        sys.path.insert(0, '.')

        # TC1
        from TC1_STRATA_SpMM_Cython.apsp import run_apsp as tc1_apsp
        # Test backward compatibility (no sigma)
        D_only = tc1_apsp(A, verbose=False)
        assert np.array_equal(D_only, D_ref), "TC1 D-only mode broken!"
        print("  TC1 backward compat: PASS")
        if not test_method('TC1 STRATA-SpMM', tc1_apsp, A, D_ref, sigma_ref):
            all_pass = False

        # TC2
        from TC2_STRATA_NumpyBLAS.apsp import run_apsp as tc2_apsp
        D_only = tc2_apsp(A, verbose=False)
        assert np.array_equal(D_only, D_ref), "TC2 D-only mode broken!"
        print("  TC2 backward compat: PASS")
        if not test_method('TC2 STRATA-Dense', tc2_apsp, A, D_ref, sigma_ref):
            all_pass = False

        # TG1 (GPU)
        try:
            from TG1_STRATA_DAWNiBFS.apsp import run_apsp as tg1_apsp
            D_only = tg1_apsp(A, verbose=False)
            assert np.array_equal(D_only, D_ref), "TG1 D-only mode broken!"
            print("  TG1 backward compat: PASS")
            if not test_method('TG1 STRATA-DAWNiBFS', tg1_apsp, A, D_ref, sigma_ref):
                all_pass = False
        except ImportError:
            print("\n--- TG1 STRATA-DAWNiBFS ---")
            print("  SKIP: CuPy not available")

        # TG2 (GPU)
        try:
            from TG2_STRATA_CUDA.apsp import run_apsp as tg2_apsp
            D_only = tg2_apsp(A, verbose=False)
            assert np.array_equal(D_only, D_ref), "TG2 D-only mode broken!"
            print("  TG2 backward compat: PASS")
            if not test_method('TG2 STRATA-CUDA', tg2_apsp, A, D_ref, sigma_ref):
                all_pass = False
        except ImportError:
            print("\n--- TG2 STRATA-CUDA ---")
            print("  SKIP: CuPy not available")

    print(f"\n{'='*50}")
    print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*50}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
