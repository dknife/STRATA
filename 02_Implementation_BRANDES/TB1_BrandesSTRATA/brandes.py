"""TB1: Matrix Brandes — STRATA CPU optimised Betweenness Centrality.

Forward pass:  Fused SpMM + prune in Numba (parallel over rows).
               Eliminates tocoo / mask / csr_matrix overhead entirely.
Backward pass: Numba JIT per-source delta accumulation (parallel).

When Numba unavailable: falls back to SciPy SpMM + NumPy prune.

Complexity: O(nm) — same as Brandes, reorganised as hop-batched
operations with compiled inner loops.
"""

import numpy as np
import scipy.sparse as sp

# ── Numba kernels ────────────────────────────────────────────

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _forward_fused(A_indptr, A_indices,
                       Rk_indptr, Rk_indices, Rk_data, Rk_nnz,
                       F, D, sigma, buf, n, power):
        """Fused SpMM + prune: A × Rk, keeping only new pairs.

        Parallel over result rows i.  Each thread writes only to
        buf[i, :], so no race conditions.
        Returns CSR arrays for next frontier + nnz count.
        """
        # Clear accumulation buffer (parallel)
        for i in prange(n):
            for k in range(n):
                buf[i, k] = 0.0

        # Parallel SpMM + prune
        for i in prange(n):
            for ae in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[ae]
                for re in range(Rk_indptr[j], Rk_indptr[j + 1]):
                    k = Rk_indices[re]
                    if F[i, k] == 0:
                        buf[i, k] += Rk_data[re]

        # Count new entries per row (parallel)
        row_cnt = np.zeros(n, dtype=np.int64)
        for i in prange(n):
            c = np.int64(0)
            for k in range(n):
                if buf[i, k] > 0.0:
                    c += 1
            row_cnt[i] = c

        total = np.int64(0)
        for i in range(n):
            total += row_cnt[i]

        if total == 0:
            empty_i = np.zeros(n + 1, dtype=np.int64)
            empty_c = np.empty(0, dtype=np.int64)
            empty_d = np.empty(0, dtype=np.float64)
            return empty_i, empty_c, empty_d, np.int64(0)

        # Build CSR + update D / sigma / F
        new_indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            new_indptr[i + 1] = new_indptr[i] + row_cnt[i]

        new_indices = np.empty(total, dtype=np.int64)
        new_data = np.empty(total, dtype=np.float64)

        for i in range(n):
            pos = new_indptr[i]
            for k in range(n):
                v = buf[i, k]
                if v > 0.0:
                    F[i, k] = 1
                    D[i, k] = np.int32(power)
                    sigma[i, k] = v
                    new_indices[pos] = np.int64(k)
                    new_data[pos] = v
                    pos += 1

        return new_indptr, new_indices, new_data, total

    @njit(parallel=True, cache=True)
    def _backward_numba(D, sigma, indptr, indices, diam, n):
        """Per-source backward delta accumulation, parallel over sources."""
        delta = np.zeros((n, n), dtype=np.float64)

        for s in prange(n):
            for k in range(diam, 1, -1):
                for w in range(n):
                    if D[s, w] != k:
                        continue
                    coeff = (1.0 + delta[s, w]) / sigma[s, w]
                    for e in range(indptr[w], indptr[w + 1]):
                        v = indices[e]
                        if D[s, v] == k - 1:
                            delta[s, v] += sigma[s, v] * coeff

        return delta


# ── Forward implementations ─────────────────────────────────

def _forward_numba(A_csr, n, verbose):
    """Forward pass: fully fused Numba SpMM + prune."""
    A_indptr = A_csr.indptr.astype(np.int64)
    A_indices = A_csr.indices.astype(np.int64)

    F = np.zeros((n, n), dtype=np.uint8)
    D = np.zeros((n, n), dtype=np.int32)
    sigma = np.zeros((n, n), dtype=np.float64)
    buf = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        F[i, i] = 1
        sigma[i, i] = 1.0

    # Hop 1: frontier = adjacency edges
    Rk_indptr = A_indptr.copy()
    Rk_indices = A_indices.copy()
    Rk_data = np.ones(len(A_indices), dtype=np.float64)
    Rk_nnz = len(A_indices)

    coo = A_csr.tocoo()
    F[coo.row, coo.col] = 1
    D[coo.row, coo.col] = 1
    sigma[coo.row, coo.col] = 1.0

    power = 1
    while Rk_nnz > 0:
        power += 1
        Rk_indptr, Rk_indices, Rk_data, Rk_nnz = _forward_fused(
            A_indptr, A_indices,
            Rk_indptr, Rk_indices, Rk_data, Rk_nnz,
            F, D, sigma, buf, n, power)
        if verbose and Rk_nnz > 0:
            print(f"    forward k={power}: nnz={Rk_nnz}")

    return D, sigma


def _forward_scipy(A_csr, n, verbose):
    """Forward pass: SciPy SpMM + NumPy prune (fallback)."""
    F = np.zeros((n, n), dtype=np.uint8)
    np.fill_diagonal(F, 1)
    D = np.zeros((n, n), dtype=np.int32)
    sigma = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)

    Rk = A_csr.copy()
    Rk.data[:] = 1.0
    coo = Rk.tocoo()
    F[coo.row, coo.col] = 1
    D[coo.row, coo.col] = 1
    sigma[coo.row, coo.col] = 1.0

    power = 1
    while True:
        power += 1
        new_R = A_csr.dot(Rk).tocsr()
        coo = new_R.tocoo()
        mask = F[coo.row, coo.col] == 0
        nr, nc = coo.row[mask], coo.col[mask]
        if len(nr) == 0:
            break
        nv = coo.data[mask]
        F[nr, nc] = 1
        D[nr, nc] = power
        sigma[nr, nc] = nv
        Rk = sp.csr_matrix((nv, (nr, nc)), shape=(n, n))
        if verbose:
            print(f"    forward k={power}: nnz={len(nr)}")

    return D, sigma


def _backward_spmm(D, sigma, A_csr, diam, n, verbose):
    """Backward pass via SpMM (fallback)."""
    level_r, level_c = {}, {}
    for k in range(1, diam + 1):
        r, c = np.where(D == k)
        level_r[k] = r
        level_c[k] = c
    delta = np.zeros((n, n), dtype=np.float64)
    for k in range(diam, 1, -1):
        rk, ck = level_r[k], level_c[k]
        if len(rk) == 0:
            continue
        bv = (1.0 + delta[rk, ck]) / sigma[rk, ck]
        B = sp.csr_matrix((bv, (rk, ck)), shape=(n, n))
        temp = B.dot(A_csr).toarray()
        rm, cm = level_r[k - 1], level_c[k - 1]
        delta[rm, cm] += sigma[rm, cm] * temp[rm, cm]
        if verbose:
            print(f"    backward k={k}: |B|={len(rk)}")
    return delta


# ── Public API ───────────────────────────────────────────────

def run_brandes(A_csr, verbose=True):
    """Compute betweenness centrality via Matrix Brandes (STRATA).

    Args:
        A_csr: Adjacency matrix (scipy sparse or dense).
        verbose: Print progress.

    Returns:
        cb: Betweenness centrality vector (n,), unnormalized, undirected.
    """
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float64)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    if verbose:
        print(f"  TB1 Matrix-Brandes (STRATA): n={n}")

    # ── Forward ──
    if _HAS_NUMBA:
        if verbose:
            print("    forward: Numba fused SpMM+prune (parallel)")
        D, sigma = _forward_numba(A_csr, n, verbose)
    else:
        if verbose:
            print("    forward: SciPy SpMM fallback")
        D, sigma = _forward_scipy(A_csr, n, verbose)

    diam = int(D.max())
    if verbose:
        print(f"    diameter={diam}, sigma_mean={sigma[sigma > 0].mean():.1f}, "
              f"sigma_max={sigma.max():.0f}")

    # ── Backward ──
    if _HAS_NUMBA:
        if verbose:
            print("    backward: Numba JIT (parallel)")
        indptr = A_csr.indptr.astype(np.int64)
        indices_arr = A_csr.indices.astype(np.int64)
        delta = _backward_numba(D, sigma, indptr, indices_arr, diam, n)
    else:
        if verbose:
            print("    backward: SpMM fallback")
        delta = _backward_spmm(D, sigma, A_csr, diam, n, verbose)

    cb = delta.sum(axis=0) / 2.0
    return cb
