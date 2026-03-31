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

    Uses a dense Boolean footprint (uint8) for O(1) lookup + in-place
    update, avoiding 3-step sparse intermediate matrix creation.
    When the Cython fused kernel is available, it is used instead.

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

        # Dense Boolean footprint for O(1) lookup (fused pruning path)
        self.F_dense = np.zeros((self.n, self.n), dtype=np.uint8)
        np.fill_diagonal(self.F_dense, 1)
        Rk_coo = self.Rk.tocoo()
        self.F_dense[Rk_coo.row, Rk_coo.col] = 1

        # Try to load Cython fused kernel
        try:
            from storm._storm_core import fused_prune_and_update
            self._fused_prune = fused_prune_and_update
        except ImportError:
            self._fused_prune = None

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
        new_R = new_R.tocsr()

        if self._fused_prune is not None:
            # Cython fused path: single C pass
            out_rows, out_cols, nnz_out = self._fused_prune(
                new_R, self.F_dense, self.n)
            if nnz_out == 0:
                raise StopIteration
            Rk_star = sp.csr_matrix(
                (np.ones(nnz_out, dtype=np.float32),
                 (out_rows.astype(np.intc), out_cols.astype(np.intc))),
                shape=(self.n, self.n))
        else:
            # NumPy vectorized path: COO extract → dense mask → filter
            coo = new_R.tocoo()
            rows, cols = coo.row, coo.col
            # Vectorized footprint lookup (no Python loop)
            mask = self.F_dense[rows, cols] == 0
            new_rows = rows[mask]
            new_cols = cols[mask]
            if len(new_rows) == 0:
                raise StopIteration
            # Update footprint in-place
            self.F_dense[new_rows, new_cols] = 1
            Rk_star = sp.csr_matrix(
                (np.ones(len(new_rows), dtype=np.float32),
                 (new_rows, new_cols)),
                shape=(self.n, self.n))

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
