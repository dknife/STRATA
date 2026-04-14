"""BCM3: I-AORM — Incremental AORM (edge-wise, dense, Python loop) (Mac).

Original AORM framework (IEEE Access, 2021) using edge-wise
row accumulation with per-node neighbor lists.
"""

import numpy as np
import scipy.sparse as sp


class IAORMIterator:
    """Incremental AORM iterator (edge-wise multiplication).

    For each node, sums the rows of R^k corresponding to its neighbors:
        temp[node, :] = sum( Rk[neighbors[node], :] )
    Then applies Heaviside pruning to keep only first-time discoveries.
    """

    def __init__(self, A, k=-1):
        A = np.asarray(A, dtype=np.float64)
        np.fill_diagonal(A, 0)
        self.n = len(A)
        self.A = A
        self.Rk = A.copy()
        self.V = np.heaviside(np.eye(self.n) + A, 0)
        self.neigh = [np.nonzero(A[node, :])[0] for node in range(self.n)]
        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        temp = np.zeros((self.n, self.n))
        for node in range(self.n):
            temp[node, :] = self.Rk[self.neigh[node], :].sum(axis=0)

        temp = np.heaviside(np.heaviside(temp, 0) - self.V, 0)
        self.V = temp + self.V

        if temp.sum() < 0.5:
            raise StopIteration

        self.Rk = temp
        return self.Rk, self.power


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via I-AORM (incremental edge-wise, dense numpy)."""
    if sp.issparse(A_csr):
        A = A_csr.toarray().astype(np.float64)
    else:
        A = np.asarray(A_csr, dtype=np.float64)

    n = len(A)
    np.fill_diagonal(A, 0)

    if verbose:
        print(f"  BCM3 I-AORM: n={n}")

    D = np.zeros((n, n), dtype=np.int32)
    A_bool = np.heaviside(A, 0)
    D[A_bool > 0] = 1

    for Rk, power in IAORMIterator(A, k):
        D[Rk > 0.5] = power
        if verbose:
            print(f"    hop {power}: nnz(Rk)={int(Rk.sum())}")

    return D
