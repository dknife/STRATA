"""TC2: D-STORM-Dense — NumPy BLAS matmul + Heaviside pruning.

Dense frontier propagation using BLAS-accelerated matrix multiply.
Same D-STORM algebra as TC1 but operating on dense numpy arrays.
"""

import numpy as np
import scipy.sparse as sp


class DenseStormIterator:
    """Dense BLAS-based reachability iterator (M-STORM / D-STORM-Dense).

    R^(k+1)* = H( H(A @ R^k) - V )
    V = V + R^(k+1)*
    """

    def __init__(self, A, k=-1):
        A = np.asarray(A, dtype=np.float32)
        self.n = len(A)
        self.A = np.heaviside(A, 0).astype(np.float32)
        self.Rk = self.A.copy()
        self.V = np.heaviside(np.eye(self.n) + self.A, 0).astype(np.float32)
        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1
        temp = self.A.dot(self.Rk)
        temp = np.heaviside(np.heaviside(temp, 0) - self.V, 0).astype(np.float32)

        if temp.sum() < 0.5:
            raise StopIteration

        self.V = temp + self.V
        self.Rk = temp
        return self.Rk, self.power


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via D-STORM dense (NumPy BLAS matmul)."""
    if sp.issparse(A_csr):
        A = A_csr.toarray()
    else:
        A = np.asarray(A_csr)
    A = A.astype(np.float32)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0).astype(np.float32)

    n = len(A)
    if verbose:
        print(f"  TC2 D-STORM-Dense: n={n}")

    D = np.zeros((n, n), dtype=np.int32)
    D[A_bool > 0] = 1

    for Rk, power in DenseStormIterator(A_bool, k):
        D[Rk > 0.5] = power
        if verbose:
            print(f"    hop {power}: nnz(Rk)={int(Rk.sum())}")

    return D
