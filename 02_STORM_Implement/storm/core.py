"""
STORM Core: Sparse Reachability Matrix Iterators

Replaces AORM's dense NumPy implementation with scipy.sparse,
reducing space complexity from O(n^2) to O(nnz).

Classes:
    SparseStormIterator - Sparse incremental k-order reachability (I-STORM)
    DenseStormIterator  - Dense SIMD matrix multiplication (M-STORM, legacy compat)

Functions:
    storm_reachability  - Compute k-order optimal reachability matrix R^(k)*
"""

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class SparseStormIterator:
    """Sparse incremental k-order reachability matrix iterator.

    Computes R^(1), R^(2)*, R^(3)*, ..., R^(k)* using sparse matrix
    operations. Each R^(k)* contains nodes reachable for the first time
    at exactly k hops (optimal reachability with path pruning).

    This is the sparse equivalent of AORM's AormIterator with
    shortest_only=True, achieving O(nnz) space instead of O(n^2).

    Args:
        A: Adjacency matrix in scipy.sparse CSR format.
        k: Maximum reachability order. -1 for convergence-based stopping.
    """

    def __init__(self, A_csr, k=-1):
        if not sp.issparse(A_csr):
            A_csr = sp.csr_matrix(A_csr)
        A_csr = A_csr.astype(np.float32)

        self.n = A_csr.shape[0]
        self.A = A_csr

        # R^(1)* = H(A): boolean reachability at order 1
        self.Rk = A_csr.copy()
        self.Rk.data[:] = 1.0

        # Footprint F = I + R^(1): tracks all discovered reachable pairs
        self.F = sp.eye(self.n, format='csr', dtype=np.float32) + self.Rk
        self.F.data = np.minimum(self.F.data, 1.0)  # clamp to boolean

        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        """Compute next R^(k)* via sparse matrix multiplication + pruning."""
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        # Step 1: Sparse matrix multiply A @ R^(k-1)*
        new_R = self.A.dot(self.Rk)

        # Step 2: Heaviside — booleanize nonzeros
        new_R = new_R.tocsr()
        new_R.data[:] = 1.0

        # Step 3: Path pruning — remove already-discovered paths
        # R^(k)* = H(A @ R^(k-1)*) AND NOT F
        # Implemented as: new_R - new_R .* F, then eliminate zeros
        already_found = new_R.multiply(self.F)
        Rk_star = new_R - already_found
        Rk_star.eliminate_zeros()

        # Step 4: Convergence check (after pruning)
        if Rk_star.nnz == 0:
            raise StopIteration

        # Step 5: Update footprint
        self.F = self.F + Rk_star
        self.F.data = np.minimum(self.F.data, 1.0)

        # Store for next iteration
        self.Rk = Rk_star

        return Rk_star, self.power


class DenseStormIterator:
    """Dense SIMD-based reachability iterator (M-STORM).

    Backward-compatible with original AORM's matmult method.
    Use SparseStormIterator for large graphs.

    Args:
        A: Dense adjacency matrix (np.ndarray).
        k: Maximum reachability order. -1 for convergence-based stopping.
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

        # SIMD matrix multiply
        temp = self.A.dot(self.Rk)

        # Path pruning: H(H(temp) - V)
        temp = np.heaviside(np.heaviside(temp, 0) - self.V, 0).astype(np.float32)

        # Convergence check (after pruning)
        if temp.sum() < 0.5:
            raise StopIteration

        # Update footprint
        self.V = temp + self.V
        self.Rk = temp

        return self.Rk, self.power


def storm_reachability(A, k=-1, method='sparse'):
    """Compute k-order optimal reachability matrices.

    Args:
        A: Adjacency matrix (sparse or dense).
        k: Maximum order (-1 for full convergence).
        method: 'sparse' (default) or 'dense'.

    Yields:
        (R_k_star, power): Tuple of reachability matrix and current order.
    """
    if method == 'sparse':
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        yield from SparseStormIterator(A, k)
    elif method == 'dense':
        A = np.asarray(A)
        yield from DenseStormIterator(A, k)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sparse' or 'dense'.")
