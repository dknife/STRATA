"""
STORM Girth: Incremental girth computation via STORM.

The girth of a directed graph G is the length of its shortest cycle.
STORM computes this incrementally using the SEMIN operator on
diagonal elements of reachability matrices.

Functions:
    storm_girth           - Compute graph girth
    storm_girth_per_node  - Compute shortest cycle per node
"""

import numpy as np
import scipy.sparse as sp

from storm.core import SparseStormIterator


def storm_girth(A, k_max=-1, verbose=True):
    """Compute the girth (shortest cycle length) of a directed graph.

    Uses AORM Theorem 2: the girth g(G) = min_i C_{i,i} where
    C^(k) is computed from diagonal elements of A @ R̃^(k).

    Args:
        A: Adjacency matrix (scipy.sparse or np.ndarray).
        k_max: Maximum order to search (-1 for diameter).
        verbose: Print progress.

    Returns:
        girth: Length of shortest cycle, or -1 if acyclic.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    n = A.shape[0]

    # Use reachability without cycle pruning
    # to detect cycles via diagonal elements
    Rk = A.copy()
    Rk.data[:] = 1.0

    for power in range(2, n + 1 if k_max < 0 else k_max + 1):
        # R^(k) = A @ R^(k-1) (without path pruning, to detect cycles)
        Rk = A.dot(Rk)
        Rk = Rk.tocsr()
        if Rk.nnz == 0:
            break
        Rk.data[:] = 1.0

        # Check diagonal: nonzero diagonal means a cycle of length `power`
        diag = Rk.diagonal()
        if diag.sum() > 0.5:
            if verbose:
                n_cyclic = int((diag > 0.5).sum())
                print(f"Girth found: {power} ({n_cyclic} nodes in shortest cycles)")
            return power

    return -1  # acyclic


def storm_girth_per_node(A, k_max=-1):
    """Compute shortest cycle length passing through each node.

    Args:
        A: Adjacency matrix.
        k_max: Maximum order to search.

    Returns:
        cycles: np.ndarray of shape (n,).
            cycles[i] = shortest cycle through node i, or -1 if none.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    n = A.shape[0]

    cycles = np.full(n, -1, dtype=np.int32)
    found = np.zeros(n, dtype=bool)

    Rk = A.copy()
    Rk.data[:] = 1.0

    for power in range(2, n + 1 if k_max < 0 else k_max + 1):
        Rk = A.dot(Rk)
        Rk = Rk.tocsr()
        if Rk.nnz == 0:
            break
        Rk.data[:] = 1.0

        diag = Rk.diagonal()
        newly_found = (diag > 0.5) & ~found
        if newly_found.any():
            cycles[newly_found] = power
            found |= newly_found

        if found.all():
            break

    return cycles
