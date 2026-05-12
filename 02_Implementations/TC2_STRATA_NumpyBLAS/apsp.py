"""TC2: STRATA-Dense — NumPy BLAS matmul + Heaviside pruning.

Dense frontier propagation using BLAS-accelerated matrix multiply.
Same STRATA algebra as TC1 but operating on dense numpy arrays.

When compute_sigma=True, preserves integer multiplication values
(instead of Heaviside binarization) to compute sigma(s,t) — the number
of shortest paths between all pairs. Uses float64 for precision.
"""

import numpy as np
import scipy.sparse as sp


class DenseStrataIterator:
    """Dense BLAS-based reachability iterator (M-STRATA / STRATA-Dense).

    Normal mode:  R^(k+1)* = H( H(A @ R^k) - V )
    Sigma mode:   R^(k+1)* = (A @ R^k) * mask(not V), values preserved
    """

    def __init__(self, A, k=-1, compute_sigma=False):
        self.compute_sigma = compute_sigma
        dtype = np.float64 if compute_sigma else np.float32
        A = np.asarray(A, dtype=dtype)
        self.n = len(A)
        self.A = np.heaviside(A, 0).astype(dtype)
        self.Rk = self.A.copy()  # sigma=1 for direct neighbors
        self.V = np.heaviside(np.eye(self.n) + self.A, 0).astype(dtype)
        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1
        temp = self.A.dot(self.Rk)

        if self.compute_sigma:
            # Keep integer values; mask by visited set
            new_mask = (temp > 0) & (self.V < 0.5)
            if not new_mask.any():
                raise StopIteration
            Rk_star = temp * new_mask  # sigma values for new pairs only
            self.V[new_mask] = 1.0
        else:
            Rk_star = np.heaviside(
                np.heaviside(temp, 0) - self.V, 0).astype(np.float32)
            if Rk_star.sum() < 0.5:
                raise StopIteration
            self.V = Rk_star + self.V

        self.Rk = Rk_star
        return self.Rk, self.power


def run_apsp(A_csr, k=-1, verbose=True, compute_sigma=False):
    """APSP via STRATA dense (NumPy BLAS matmul).

    Args:
        A_csr: Adjacency matrix (sparse or dense).
        k: Hop constraint (-1 for full APSP).
        verbose: Show progress.
        compute_sigma: If True, also compute sigma(s,t) matrix.
            Returns (D, sigma) tuple instead of just D.
            sigma[i,j] = number of shortest paths from i to j.
    """
    if sp.issparse(A_csr):
        A = A_csr.toarray()
    else:
        A = np.asarray(A_csr)
    dtype = np.float64 if compute_sigma else np.float32
    A = A.astype(dtype)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0).astype(dtype)

    n = len(A)
    if verbose:
        print(f"  TC2 STRATA-Dense: n={n}" +
              (" (sigma mode)" if compute_sigma else ""))

    D = np.zeros((n, n), dtype=np.int32)
    D[A_bool > 0] = 1

    sigma = None
    if compute_sigma:
        sigma = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(sigma, 1.0)
        sigma[A_bool > 0] = 1.0

    for Rk, power in DenseStrataIterator(A_bool, k,
                                        compute_sigma=compute_sigma):
        mask = Rk > 0.5
        D[mask] = power
        if compute_sigma:
            sigma[mask] = Rk[mask]
        if verbose:
            print(f"    hop {power}: nnz(Rk)={int(mask.sum())}")

    if compute_sigma:
        return D, sigma
    return D
