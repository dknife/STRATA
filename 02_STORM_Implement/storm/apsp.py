"""
STORM APSP: All-Pairs Shortest Path computation via STORM framework.

Implements Algorithm 3 from the AORM paper using sparse matrices,
with support for arbitrary-hop constrained APSP.

Functions:
    storm_apsp             - Full APSP via incremental STORM
    storm_apsp_constrained - Hop-constrained approximate APSP
    storm_apsp_dense       - Dense APSP (legacy compatibility)
"""

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from storm.core import SparseStormIterator, DenseStormIterator


def storm_apsp(A, k=-1, verbose=True):
    """Compute all-pairs shortest path distance matrix via sparse STORM.

    D_{i,j} = length of the shortest path from v_i to v_j.
    Based on Theorem 1: D = sum_{k=1}^{d} k * R^(k)*

    Args:
        A: Adjacency matrix (scipy.sparse or np.ndarray).
            Converted to sparse CSR internally.
        k: Maximum hop constraint (-1 for full APSP).
        verbose: Show progress bar.

    Returns:
        D: Distance matrix as scipy.sparse.csr_matrix.
           D_{i,j} = shortest path distance, 0 if unreachable or i==j.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)

    # Remove self-loops
    A = A - sp.diags(A.diagonal())
    A.eliminate_zeros()

    n = A.shape[0]

    # Initialize D = A (1-hop distances)
    D = A.copy()
    D.data[:] = 1.0

    iterator = SparseStormIterator(A, k)
    pbar = tqdm(iterator, desc='STORM APSP', disable=not verbose)

    for Rk_star, power in pbar:
        # D = D + k * R^(k)* — accumulate distances
        D = D + Rk_star.multiply(power)
        pbar.set_postfix(order=power, nnz_Rk=Rk_star.nnz, nnz_D=D.nnz)

    return D


def storm_apsp_constrained(A, k_max, verbose=True):
    """Compute hop-constrained approximate APSP.

    Only considers paths of length <= k_max.
    Useful for large graphs where full APSP is unnecessary
    (e.g., traffic congestion prediction, local community detection).

    Args:
        A: Adjacency matrix.
        k_max: Maximum hop constraint.
        verbose: Show progress bar.

    Returns:
        D: Approximate distance matrix (paths longer than k_max are 0).
    """
    return storm_apsp(A, k=k_max, verbose=verbose)


def storm_apsp_dense(A, k=-1, verbose=True):
    """Compute APSP using dense M-STORM (legacy compatibility).

    For small graphs (n < 10000) where dense operations may be faster
    due to BLAS optimization.

    Args:
        A: Dense adjacency matrix (np.ndarray).
        k: Maximum hop constraint (-1 for full).
        verbose: Show progress bar.

    Returns:
        D: Dense distance matrix (np.ndarray).
    """
    A = np.asarray(A, dtype=np.float32)
    np.fill_diagonal(A, 0)
    A_bool = np.heaviside(A, 0).astype(np.float32)

    D = A_bool.copy()
    iterator = DenseStormIterator(A_bool, k)
    pbar = tqdm(iterator, desc='M-STORM APSP', disable=not verbose)

    for Rk_star, power in pbar:
        D = D + power * Rk_star
        pbar.set_postfix(order=power)

    return D


def storm_apsp_incremental(A, k_max, verbose=True):
    """Compute APSP incrementally, yielding D^(k) at each step.

    Useful for analyzing convergence or early stopping.

    Args:
        A: Adjacency matrix.
        k_max: Maximum hop constraint.
        verbose: Show progress bar.

    Yields:
        (D_k, k): Distance matrix at order k and current order.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    A = A - sp.diags(A.diagonal())
    A.eliminate_zeros()

    D = A.copy()
    D.data[:] = 1.0
    yield D.copy(), 1

    iterator = SparseStormIterator(A, k_max)
    pbar = tqdm(iterator, desc='STORM Incremental', disable=not verbose)

    for Rk_star, power in pbar:
        D = D + Rk_star.multiply(power)
        yield D.copy(), power
