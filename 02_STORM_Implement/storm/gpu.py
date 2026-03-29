"""
STORM GPU: GPU-accelerated reachability computation via CuPy.

Requires: cupy, cupyx (NVIDIA GPU with CUDA)
Falls back to CPU sparse if CuPy is not available.

Classes:
    GpuStormIterator - GPU sparse reachability iterator
Functions:
    gpu_storm_apsp   - GPU-accelerated APSP
"""

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class GpuStormIterator:
    """GPU-accelerated sparse STORM iterator.

    Performs sparse matrix operations on GPU using cuSPARSE backend.
    Achieves significant speedup for large graphs due to GPU parallelism.

    Args:
        A_scipy_csr: Adjacency matrix in scipy CSR format (transferred to GPU).
        k: Maximum reachability order. -1 for convergence-based stopping.

    Raises:
        ImportError: If CuPy is not installed.
    """

    def __init__(self, A_scipy_csr, k=-1):
        if not HAS_CUPY:
            raise ImportError(
                "CuPy is required for GPU acceleration. "
                "Install with: pip install cupy-cuda12x"
            )

        if not sp.issparse(A_scipy_csr):
            A_scipy_csr = sp.csr_matrix(A_scipy_csr)
        A_scipy_csr = A_scipy_csr.astype(np.float32)

        self.n = A_scipy_csr.shape[0]

        # Transfer to GPU
        self.A = cusp.csr_matrix(A_scipy_csr)

        # R^(1)* = H(A)
        self.Rk = self.A.copy()
        self.Rk.data = cp.ones_like(self.Rk.data)

        # Footprint F = I + R^(1)
        self.F = cusp.eye(self.n, format='csr', dtype=cp.float32) + self.Rk
        self.F.data = cp.minimum(self.F.data, 1.0)

        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        # GPU SpMM: A @ R^(k-1)*
        new_R = self.A.dot(self.Rk)
        new_R = new_R.tocsr()
        new_R.data = cp.ones_like(new_R.data)

        # Path pruning on GPU
        already_found = new_R.multiply(self.F)
        Rk_star = new_R - already_found
        Rk_star.eliminate_zeros()

        # Convergence check
        if Rk_star.nnz == 0:
            raise StopIteration

        # Update footprint
        self.F = self.F + Rk_star
        self.F.data = cp.minimum(self.F.data, 1.0)

        self.Rk = Rk_star
        return Rk_star, self.power

    def to_scipy(self, gpu_matrix):
        """Transfer a GPU sparse matrix back to CPU."""
        return gpu_matrix.get()


def gpu_storm_apsp(A, k=-1, verbose=True):
    """Compute APSP using GPU-accelerated STORM.

    Args:
        A: Adjacency matrix (scipy.sparse or np.ndarray).
        k: Maximum hop constraint (-1 for full APSP).
        verbose: Show progress bar.

    Returns:
        D: Distance matrix as scipy.sparse.csr_matrix (on CPU).
    """
    if not HAS_CUPY:
        raise ImportError("CuPy required. Install with: pip install cupy-cuda12x")

    from tqdm import tqdm

    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    A = A - sp.diags(A.diagonal())
    A.eliminate_zeros()

    # Initialize D on GPU
    D_gpu = cusp.csr_matrix(A.astype(np.float32))
    D_gpu.data = cp.ones_like(D_gpu.data)

    iterator = GpuStormIterator(A, k)
    pbar = tqdm(iterator, desc='GPU-STORM APSP', disable=not verbose)

    for Rk_star, power in pbar:
        D_gpu = D_gpu + Rk_star.multiply(float(power))
        pbar.set_postfix(order=power, nnz=Rk_star.nnz)

    # Transfer result back to CPU
    return D_gpu.get()


def is_gpu_available():
    """Check if GPU acceleration is available."""
    return HAS_CUPY
