"""TG1: GPU-Dense — cuBLAS dense matmul D-STORM.

D-STORM algebra on dense GPU arrays using cuBLAS matrix multiplication.
Best for small/dense graphs where cuBLAS throughput dominates.
"""

import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp


class GpuDenseStormIterator:
    """GPU dense STORM iterator (cuBLAS matmul + Heaviside pruning)."""

    def __init__(self, A, k=-1):
        if sp.issparse(A):
            A = A.toarray()
        A = np.asarray(A, dtype=np.float32)
        self.n = A.shape[0]
        A_gpu = cp.asarray(A)
        self.A = cp.heaviside(A_gpu, 0).astype(cp.float32)
        self.Rk = self.A.copy()
        self.V = cp.heaviside(cp.eye(self.n, dtype=cp.float32) + self.A, 0).astype(cp.float32)
        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration
        self.power += 1
        temp = self.A.dot(self.Rk)
        temp = cp.heaviside(cp.heaviside(temp, 0) - self.V, 0).astype(cp.float32)
        if float(temp.sum()) < 0.5:
            raise StopIteration
        self.V = temp + self.V
        self.Rk = temp
        return self.Rk, self.power


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via GPU dense D-STORM (cuBLAS matmul)."""
    if sp.issparse(A_csr):
        A = A_csr.toarray()
    else:
        A = np.asarray(A_csr)
    A = A.astype(np.float32)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0).astype(np.float32)

    n = len(A)
    if verbose:
        print(f"  TG1 GPU-Dense: n={n}")

    D_gpu = cp.zeros((n, n), dtype=cp.int32)
    D_gpu[cp.asarray(A_bool) > 0] = 1

    for Rk, power in GpuDenseStormIterator(A_bool, k):
        D_gpu[Rk > 0.5] = power
        if verbose:
            print(f"    hop {power}")

    return cp.asnumpy(D_gpu)
