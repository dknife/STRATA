"""TG2: GPU-DirectExpand — CUDA direct CSR frontier expansion.

D-STORM algebra without SpMM: a custom CUDA kernel directly expands
each frontier entry (i, j) by traversing j's CSR neighbors, checking
the dense footprint with guard+CAS (race-free 0->1 transitions).

Eliminates three SpMM bottlenecks:
  1. SpMM computes all products including already-visited pairs
  2. Sparse intermediate matrix creation/conversion overhead
  3. Separate prune step after SpMM

When compute_sigma=True, a second kernel accumulates shortest-path
counts sigma(s,t) via atomicAdd after each expansion step.
Two-pass approach: expand kernel (unchanged) then sigma accumulation.
"""

import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp


_expand_kernel = cp.RawKernel(r'''
extern "C" __global__
void direct_expand(
    const int* __restrict__ f_row,
    const int* __restrict__ f_col,
    const int   frontier_nnz,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    int* __restrict__ F,
    int* __restrict__ out_row,
    int* __restrict__ out_col,
    int* __restrict__ out_count,
    const int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_nnz) return;
    int i = f_row[tid];
    int j = f_col[tid];
    int start = indptr[j];
    int end   = indptr[j + 1];
    for (int e = start; e < end; e++) {
        int k = indices[e];
        long long idx = (long long)i * n + k;
        if (F[idx] == 0) {
            if (atomicCAS(&F[idx], 0, 1) == 0) {
                int pos = atomicAdd(out_count, 1);
                out_row[pos] = i;
                out_col[pos] = k;
            }
        }
    }
}
''', 'direct_expand')


_sigma_kernel = cp.RawKernel(r'''
extern "C" __global__
void sigma_accumulate(
    const int* __restrict__ f_row,
    const int* __restrict__ f_col,
    const double* __restrict__ f_sigma,
    const int   frontier_nnz,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const int* __restrict__ D,
    double* __restrict__ sigma,
    const int level,
    const int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_nnz) return;
    int i = f_row[tid];
    int j = f_col[tid];
    double sig = f_sigma[tid];
    int start = indptr[j];
    int end   = indptr[j + 1];
    for (int e = start; e < end; e++) {
        int k = indices[e];
        long long idx = (long long)i * n + k;
        if (D[idx] == level) {
            atomicAdd(&sigma[idx], sig);
        }
    }
}
''', 'sigma_accumulate')


def run_apsp(A_csr, k=-1, verbose=True, compute_sigma=False):
    """APSP via GPU direct frontier expansion (no SpMM).

    Args:
        A_csr: Adjacency matrix (sparse or dense).
        k: Hop constraint (-1 for full APSP).
        verbose: Show progress.
        compute_sigma: If True, also compute sigma(s,t) matrix.
            Returns (D, sigma) tuple instead of just D.
            sigma[i,j] = number of shortest paths from i to j.
            Uses two-pass approach: expand then accumulate.
    """
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    if verbose:
        print(f"  TG2 GPU-DirectExpand: n={n}" +
              (" (sigma mode)" if compute_sigma else ""))

    indptr_gpu = cp.asarray(A_csr.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_csr.indices.astype(np.int32))

    F_gpu = cp.zeros((n, n), dtype=cp.int32)
    cp.fill_diagonal(F_gpu, 1)
    D_gpu = cp.zeros((n, n), dtype=cp.int32)

    A_coo = A_csr.tocoo()
    f_rows = cp.asarray(A_coo.row.astype(np.int32))
    f_cols = cp.asarray(A_coo.col.astype(np.int32))
    frontier_nnz = len(f_rows)
    D_gpu[f_rows, f_cols] = 1
    F_gpu[f_rows, f_cols] = 1

    # Sigma mode: allocate sigma matrix and per-frontier sigma values
    sigma_gpu = None
    f_sigma = None
    if compute_sigma:
        sigma_gpu = cp.zeros((n, n), dtype=cp.float64)
        cp.fill_diagonal(sigma_gpu, 1.0)          # sigma(i,i) = 1
        sigma_gpu[f_rows, f_cols] = 1.0            # sigma=1 for direct neighbors
        f_sigma = cp.ones(frontier_nnz, dtype=cp.float64)

    out_row = cp.empty(n * n, dtype=cp.int32)
    out_col = cp.empty(n * n, dtype=cp.int32)
    out_count = cp.zeros(1, dtype=cp.int32)

    level = 2
    max_level = n if k < 0 else k

    while frontier_nnz > 0 and level <= max_level:
        out_count[0] = 0
        grid_size = (frontier_nnz + 255) // 256

        # Pass 1: expand frontier (unchanged from boolean mode)
        _expand_kernel(
            (grid_size,), (256,),
            (f_rows, f_cols, np.int32(frontier_nnz),
             indptr_gpu, indices_gpu, F_gpu,
             out_row, out_col, out_count, np.int32(n))
        )
        cp.cuda.Stream.null.synchronize()

        new_nnz = int(out_count[0])
        if verbose:
            print(f"    hop {level}: nnz(frontier)={new_nnz}")
        if new_nnz == 0:
            break

        new_rows = out_row[:new_nnz]
        new_cols = out_col[:new_nnz]
        D_gpu[new_rows, new_cols] = level

        # Pass 2: accumulate sigma (only in sigma mode)
        if compute_sigma:
            _sigma_kernel(
                (grid_size,), (256,),
                (f_rows, f_cols, f_sigma, np.int32(frontier_nnz),
                 indptr_gpu, indices_gpu, D_gpu, sigma_gpu,
                 np.int32(level), np.int32(n))
            )
            cp.cuda.Stream.null.synchronize()
            # Read accumulated sigma values for new frontier entries
            f_sigma = sigma_gpu[new_rows, new_cols].copy()

        f_rows = new_rows.copy()
        f_cols = new_cols.copy()
        frontier_nnz = new_nnz
        level += 1

    if compute_sigma:
        return cp.asnumpy(D_gpu), cp.asnumpy(sigma_gpu)
    return cp.asnumpy(D_gpu)
