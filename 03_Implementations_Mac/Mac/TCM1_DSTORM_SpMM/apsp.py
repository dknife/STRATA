"""TCM1: D-STORM-SpMM — SciPy SpMM + NumPy vectorized pruning (Mac).

Sparse frontier propagation with cumulative dense footprint.
Uses NumPy vectorized COO scan for pruning (no Cython on Mac).
"""

import numpy as np
import scipy.sparse as sp


class SparseStormIterator:
    """Sparse incremental k-order reachability iterator.

    Computes R^(k)* = H(A @ R^(k-1)*) AND NOT F via:
      - SciPy CSR SpMM for frontier expansion
      - NumPy vectorized COO scan for pruning
      - Dense uint8 footprint for O(1) lookup
    """

    def __init__(self, A_csr, k=-1):
        if not sp.issparse(A_csr):
            A_csr = sp.csr_matrix(A_csr)
        A_csr = A_csr.astype(np.float32)

        self.n = A_csr.shape[0]
        self.A = A_csr
        self.Rk = A_csr.copy()
        self.Rk.data[:] = 1.0

        # Dense Boolean footprint
        self.F_dense = np.zeros((self.n, self.n), dtype=np.uint8)
        np.fill_diagonal(self.F_dense, 1)
        Rk_coo = self.Rk.tocoo()
        self.F_dense[Rk_coo.row, Rk_coo.col] = 1

        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        new_R = self.A.dot(self.Rk).tocsr()

        # NumPy vectorized pruning
        coo = new_R.tocoo()
        rows, cols = coo.row, coo.col
        mask = self.F_dense[rows, cols] == 0
        new_rows = rows[mask]
        new_cols = cols[mask]
        if len(new_rows) == 0:
            raise StopIteration
        self.F_dense[new_rows, new_cols] = 1
        Rk_star = sp.csr_matrix(
            (np.ones(len(new_rows), dtype=np.float32),
             (new_rows, new_cols)),
            shape=(self.n, self.n))

        self.Rk = Rk_star
        return Rk_star, self.power


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via D-STORM sparse (NumPy vectorized pruning, Mac)."""
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    if verbose:
        print(f"  TCM1 D-STORM-SpMM: n={n}")

    D = np.zeros((n, n), dtype=np.int32)
    A_coo = A_csr.tocoo()
    D[A_coo.row, A_coo.col] = 1

    for Rk_star, power in SparseStormIterator(A_csr, k):
        coo = Rk_star.tocoo()
        D[coo.row, coo.col] = power
        if verbose:
            print(f"    hop {power}: nnz(R*)={Rk_star.nnz}")

    return D
