"""
STORM-GPU APSP: CUDA-accelerated All-Pairs Shortest Paths.

Functions:
    gpu_storm_apsp        — Sparse GPU APSP (cuSPARSE SpMM)
    gpu_storm_apsp_dense  — Dense GPU APSP (cuBLAS matmul)
    gpu_storm_apsp_fused  — Fused-kernel GPU APSP (custom CUDA kernel)
"""

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from storm_gpu.cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp
import cupyx.scipy.sparse as cusp

from storm_gpu.core import (
    GpuSparseStormIterator,
    GpuDenseStormIterator,
    GpuFusedStormIterator,
)


def gpu_storm_apsp(A, k=-1, verbose=True):
    """APSP via GPU sparse STORM (cuSPARSE SpMM).

    Port of CPU storm_apsp to GPU. All SpMM and pruning operations
    run on GPU. Distance matrix is accumulated on GPU, then transferred
    back to CPU at the end.

    Args:
        A: Adjacency matrix (scipy.sparse or numpy array).
        k: Hop constraint (-1 for full APSP).
        verbose: Show progress.

    Returns:
        D: Distance matrix as scipy.sparse.csr_matrix (on CPU).
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    # Initialize D on GPU
    A_gpu = cusp.csr_matrix(A)
    D_gpu = A_gpu.copy()
    D_gpu.data = cp.ones_like(D_gpu.data)

    iterator = GpuSparseStormIterator(A, k)
    pbar = tqdm(iterator, desc='GPU-Sparse APSP', disable=not verbose)

    for Rk_star, power in pbar:
        D_gpu = D_gpu + Rk_star.multiply(float(power))
        pbar.set_postfix(order=power, nnz_Rk=Rk_star.nnz)

    # Transfer to CPU
    return D_gpu.get()


def gpu_storm_apsp_dense(A, k=-1, verbose=True):
    """APSP via GPU dense STORM (cuBLAS matmul).

    Best for small or dense graphs. All operations on dense GPU arrays.

    Args:
        A: Adjacency matrix.
        k: Hop constraint (-1 for full APSP).
        verbose: Show progress.

    Returns:
        D: Distance matrix as numpy array.
    """
    if sp.issparse(A):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float32)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0).astype(np.float32)

    D_gpu = cp.asarray(A_bool.copy())

    iterator = GpuDenseStormIterator(A_bool, k)
    pbar = tqdm(iterator, desc='GPU-Dense APSP', disable=not verbose)

    for Rk_star, power in pbar:
        D_gpu = D_gpu + float(power) * Rk_star
        pbar.set_postfix(order=power)

    return cp.asnumpy(D_gpu)


def gpu_storm_apsp_fused(A, k=-1, verbose=True):
    """APSP via GPU fused-kernel STORM (custom CUDA kernel).

    Uses a custom ElementwiseKernel for fused prune+footprint update,
    eliminating intermediate sparse temporaries. Dense footprint on GPU.

    This is the most optimized GPU path.

    Args:
        A: Adjacency matrix (scipy.sparse or numpy array).
        k: Hop constraint (-1 for full APSP).
        verbose: Show progress.

    Returns:
        D: Distance matrix as numpy array (int32).
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    n = A.shape[0]

    # Distance matrix on GPU (dense, int32)
    D_gpu = cp.zeros((n, n), dtype=cp.int32)

    # Record 1-hop distances
    A_coo = A.tocoo()
    D_gpu[A_coo.row, A_coo.col] = 1

    iterator = GpuFusedStormIterator(A, k)
    pbar = tqdm(iterator, desc='GPU-Fused APSP', disable=not verbose)

    for Rk_star, power in pbar:
        # Extract new shell COO and record distances
        coo = Rk_star.tocoo()
        rows = coo.row
        cols = coo.col
        D_gpu[rows, cols] = power
        pbar.set_postfix(order=power, nnz_Rk=Rk_star.nnz)

    return cp.asnumpy(D_gpu)
