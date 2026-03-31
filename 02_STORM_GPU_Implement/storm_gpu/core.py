"""
STORM-GPU Core: CUDA-accelerated sparse frontier propagation.

Three GPU execution strategies:
  1. GpuSparseStormIterator  — cuSPARSE SpMM + GPU element-wise masking
  2. GpuDenseStormIterator   — cuBLAS dense matmul (small/dense graphs)
  3. GpuFusedStormIterator   — Custom CUDA kernel for fused prune+update

All strategies maintain the same mathematical semantics as CPU STORM:
  R^(k)* = H(A @ R^(k-1)*) AND NOT F_{k-1}
  F_k = F_{k-1} OR R^(k)*
"""

import numpy as np
import scipy.sparse as sp

# Setup CUDA env before importing CuPy
from storm_gpu.cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp
import cupyx.scipy.sparse as cusp


class GpuSparseStormIterator:
    """GPU-accelerated sparse STORM iterator using cuSPARSE SpMM.

    Port of SparseStormIterator to GPU. All sparse matrix operations
    (SpMM, element-wise multiply, add) run on GPU via cuSPARSE.

    Args:
        A_csr: Adjacency matrix (scipy.sparse.csr_matrix, transferred to GPU).
        k: Maximum reachability order (-1 for full convergence).
    """

    def __init__(self, A_csr, k=-1):
        if not sp.issparse(A_csr):
            A_csr = sp.csr_matrix(A_csr)
        A_csr = A_csr.astype(np.float32)

        self.n = A_csr.shape[0]

        # Transfer adjacency to GPU
        self.A = cusp.csr_matrix(A_csr)

        # R^(1)* = H(A)
        self.Rk = self.A.copy()
        self.Rk.data = cp.ones_like(self.Rk.data)

        # Footprint F = I + R^(1)*
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

        # Step 1: GPU SpMM — A @ R^(k-1)*
        new_R = self.A.dot(self.Rk)
        new_R = new_R.tocsr()

        # Step 2: Booleanize
        new_R.data = cp.ones_like(new_R.data)

        # Step 3: Path pruning — remove already-discovered pairs
        already_found = new_R.multiply(self.F)
        Rk_star = new_R - already_found
        Rk_star.eliminate_zeros()

        # Step 4: Convergence check
        if Rk_star.nnz == 0:
            raise StopIteration

        # Step 5: Update footprint
        self.F = self.F + Rk_star
        self.F.data = cp.minimum(self.F.data, 1.0)

        self.Rk = Rk_star
        return Rk_star, self.power


class GpuDenseStormIterator:
    """GPU-accelerated dense STORM iterator using cuBLAS.

    All operations on dense GPU arrays. Best for small/dense graphs
    where cuBLAS matmul throughput dominates.

    Args:
        A: Adjacency matrix (numpy array or scipy sparse, densified on GPU).
        k: Maximum reachability order (-1 for full convergence).
    """

    def __init__(self, A, k=-1):
        if sp.issparse(A):
            A = A.toarray()
        A = np.asarray(A, dtype=np.float32)

        self.n = A.shape[0]

        # Transfer to GPU as dense
        A_gpu = cp.asarray(A)
        self.A = cp.heaviside(A_gpu, 0).astype(cp.float32)

        # R^(1)* = H(A)
        self.Rk = self.A.copy()

        # Footprint V = I + H(A)
        self.V = cp.heaviside(cp.eye(self.n, dtype=cp.float32) + self.A, 0).astype(cp.float32)

        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        # cuBLAS dense matmul
        temp = self.A.dot(self.Rk)

        # Path pruning: H(H(temp) - V)
        temp = cp.heaviside(cp.heaviside(temp, 0) - self.V, 0).astype(cp.float32)

        # Convergence check
        if float(temp.sum()) < 0.5:
            raise StopIteration

        # Update
        self.V = temp + self.V
        self.Rk = temp

        return self.Rk, self.power


class GpuFusedStormIterator:
    """GPU STORM with custom CUDA kernel for fused prune+update.

    Uses a dense Boolean footprint on GPU (uint8 matrix) with a custom
    ElementwiseKernel that fuses the prune-and-update step into a single
    GPU kernel launch, eliminating intermediate sparse temporaries.

    The SpMM step still uses cuSPARSE, but the prune+footprint update
    is a single fused kernel over the candidate COO entries.

    Args:
        A_csr: Adjacency matrix (scipy.sparse.csr_matrix).
        k: Maximum reachability order (-1 for full convergence).
    """

    # Custom CUDA kernel: for each candidate (i,j), check footprint,
    # mark new entries, update footprint in-place
    _prune_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void fused_prune(unsigned char* F, const int* rows, const int* cols,
                     int* is_new, int nnz, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= nnz) return;
        int r = rows[tid];
        int c = cols[tid];
        int idx = r * n + c;
        if (F[idx] == 0) {
            F[idx] = 1;
            is_new[tid] = 1;
        } else {
            is_new[tid] = 0;
        }
    }
    ''', 'fused_prune')

    def __init__(self, A_csr, k=-1):
        if not sp.issparse(A_csr):
            A_csr = sp.csr_matrix(A_csr)
        A_csr = A_csr.astype(np.float32)

        self.n = A_csr.shape[0]
        self.A = cusp.csr_matrix(A_csr)

        # Dense Boolean footprint on GPU (uint8 for kernel compatibility)
        self.F_dense = cp.zeros((self.n, self.n), dtype=cp.uint8)
        # Set diagonal (self-reachability)
        cp.fill_diagonal(self.F_dense, 1)

        # Initialize R^(1)* and mark in footprint
        self.Rk = self.A.copy()
        self.Rk.data = cp.ones_like(self.Rk.data)

        # Mark 1-hop pairs in footprint
        Rk_coo = self.Rk.tocoo()
        self.F_dense[Rk_coo.row, Rk_coo.col] = 1

        self.power = 1
        self.maxpower = k

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration

        self.power += 1

        # Step 1: cuSPARSE SpMM
        new_R = self.A.dot(self.Rk).tocsr()
        new_R.data = cp.ones_like(new_R.data)

        # Step 2: Extract candidate COO entries
        coo = new_R.tocoo()
        if coo.nnz == 0:
            raise StopIteration

        rows_gpu = coo.row.astype(cp.int32)
        cols_gpu = coo.col.astype(cp.int32)

        # Step 3: Fused prune + footprint update (single kernel launch)
        nnz = len(rows_gpu)
        is_new = cp.zeros(nnz, dtype=cp.int32)
        block_size = 256
        grid_size = (nnz + block_size - 1) // block_size
        self._prune_kernel(
            (grid_size,), (block_size,),
            (self.F_dense, rows_gpu, cols_gpu, is_new, nnz, self.n)
        )

        # Step 4: Filter to keep only new entries
        mask = is_new.astype(cp.bool_)
        new_count = int(mask.sum())

        if new_count == 0:
            raise StopIteration

        new_rows = rows_gpu[mask]
        new_cols = cols_gpu[mask]
        new_data = cp.ones(new_count, dtype=cp.float32)

        # Build new frontier as sparse
        Rk_star = cusp.coo_matrix(
            (new_data, (new_rows, new_cols)),
            shape=(self.n, self.n)
        ).tocsr()

        self.Rk = Rk_star
        return Rk_star, self.power
