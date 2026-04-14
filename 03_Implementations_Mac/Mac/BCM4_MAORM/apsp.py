"""BCM4: M-AORM — Matrix-multiply AORM (dense BLAS matmul) (Mac).

Original AORM framework (IEEE Access, 2021) using dense
matrix multiplication (Accelerate BLAS on macOS).
"""

import numpy as np
import scipy.sparse as sp


class MAORMIterator:
    """Matrix-multiply AORM iterator (BLAS dense matmul).

    Computes R^(k+1) = A @ R^k using numpy dense matrix multiplication
    (Apple Accelerate BLAS on macOS).
    """

    def __init__(self, A, k=-1):
        A = np.asarray(A, dtype=np.float64)
        np.fill_diagonal(A, 0)
        self.n = len(A)
        self.A = np.heaviside(A, 0)
        self.Rk = self.A.copy()
        self.V = np.heaviside(np.eye(self.n) + self.A, 0)
        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        temp = self.A.dot(self.Rk)
        temp = np.heaviside(np.heaviside(temp, 0) - self.V, 0)
        self.V = temp + self.V

        if temp.sum() < 0.5:
            raise StopIteration

        self.Rk = temp
        return self.Rk, self.power


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via M-AORM (dense BLAS matmul, Accelerate on Mac)."""
    if sp.issparse(A_csr):
        A = A_csr.toarray().astype(np.float64)
    else:
        A = np.asarray(A_csr, dtype=np.float64)

    n = len(A)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0)

    if verbose:
        print(f"  BCM4 M-AORM: n={n}")

    D = np.zeros((n, n), dtype=np.int32)
    D[A_bool > 0] = 1

    for Rk, power in MAORMIterator(A, k):
        D[Rk > 0.5] = power
        if verbose:
            print(f"    hop {power}: nnz(Rk)={int(Rk.sum())}")

    return D
