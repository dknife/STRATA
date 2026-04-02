"""BG1: GPU Per-Source BFS — CUDA per-source level-synchronous BFS.

True GPU parallelization of SciPy's sequential per-source BFS.
Each source vertex gets one CUDA thread block; threads within
the block cooperate on frontier expansion via CSR traversal.

NOT based on D-STORM — pure graph BFS with no matrix algebra.
"""

import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp


_bfs_kernel = cp.RawKernel(r'''
extern "C" __global__
void bfs_apsp(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    int* __restrict__ D,
    const int n,
    const int max_level
) {
    const int src = blockIdx.x;
    if (src >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    int* dist = D + (long long)src * n;

    for (int i = tid; i < n; i += nthreads)
        dist[i] = -1;
    __syncthreads();
    if (tid == 0) dist[src] = 0;
    __syncthreads();

    const int limit = (max_level > 0) ? max_level : n - 1;
    for (int level = 0; level < limit; level++) {
        __shared__ int found;
        if (tid == 0) found = 0;
        __syncthreads();

        for (int u = tid; u < n; u += nthreads) {
            if (dist[u] == level) {
                const int row_start = indptr[u];
                const int row_end   = indptr[u + 1];
                for (int e = row_start; e < row_end; e++) {
                    const int v = indices[e];
                    if (dist[v] == -1) {
                        dist[v] = level + 1;
                        found = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (found == 0) break;
        __syncthreads();
    }

    for (int i = tid; i < n; i += nthreads) {
        if (dist[i] < 0) dist[i] = 0;
    }
}
''', 'bfs_apsp')


def run_apsp(A_csr, k=-1, block_size=256, verbose=True):
    """APSP via GPU per-source BFS (one CUDA block per source)."""
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    if verbose:
        print(f"  BG1 GPU-PerSrc-BFS: n={n}, blocks={n}, threads={block_size}")

    indptr_gpu = cp.asarray(A_csr.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_csr.indices.astype(np.int32))
    D_gpu = cp.empty((n, n), dtype=cp.int32)

    max_level = k if k > 0 else -1
    _bfs_kernel(
        (n,), (block_size,),
        (indptr_gpu, indices_gpu, D_gpu, np.int32(n), np.int32(max_level))
    )
    cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(D_gpu)
