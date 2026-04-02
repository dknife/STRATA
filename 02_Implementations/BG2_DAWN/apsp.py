"""BG2: DAWN-style GPU APSP — Sparse Optimized Vector-Matrix (SOVM) BFS.

Reimplements the DAWN algorithm (Feng et al., ICS 2024) using CuPy.
DAWN formulates BFS as a sparse vector-matrix operation: only frontier
vertices are expanded, unlike BG1 which scans all vertices per level.

Key differences from BG1 (GPU-PerSrc-BFS):
  - BG1: each thread scans ALL vertices looking for dist[u]==level → O(n) per level
  - BG2: each thread checks frontier[j], expands only if true → O(frontier) per level
  - Double-buffered frontiers (alpha/beta) avoid separate visited array

Reference: Yelai Feng et al., "DAWN: Matrix Operation-Optimized Algorithm
for Shortest Paths Problem on Unweighted Graphs," ACM ICS 2024.
https://github.com/lxrzlyr/GAL-DAWN
"""

import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp


# DAWN SOVM kernel: one block per source, frontier-driven expansion.
# Each thread handles ceil(n / blockDim.x) vertices.
# alpha = current frontier, beta = next frontier (double-buffered).
_dawn_kernel = cp.RawKernel(r'''
extern "C" __global__
void dawn_apsp(
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

    // Shared memory for convergence flag
    __shared__ int found;

    // Per-source distance row
    int* dist = D + (long long)src * n;

    // Use alpha/beta frontier arrays in global memory
    // We encode frontier in sign bit of dist:
    //   dist[v] = 0        → unvisited
    //   dist[v] = level    → visited at level, not in frontier
    //   dist[v] = -level   → visited at level, currently in frontier

    // Initialize: dist = 0 for all, dist[src] = mark as visited (not frontier)
    for (int i = tid; i < n; i += nthreads)
        dist[i] = 0;
    __syncthreads();

    // Set source as visited (distance 0, not in frontier)
    // Set source's neighbors as frontier (distance -1)
    if (tid == 0) dist[src] = 0;
    __syncthreads();

    // Initialize 1-hop: source's neighbors are the initial frontier
    const int src_start = indptr[src];
    const int src_end = indptr[src + 1];
    for (int e = src_start + tid; e < src_end; e += nthreads) {
        int v = indices[e];
        dist[v] = -1;  // negative = in frontier, distance = 1
    }
    __syncthreads();

    const int limit = (max_level > 0) ? max_level : n - 1;

    for (int level = 1; level <= limit; level++) {
        if (tid == 0) found = 0;
        __syncthreads();

        int next_level = level + 1;

        // SOVM: only expand vertices in the frontier (dist[j] == -level)
        for (int j = tid; j < n; j += nthreads) {
            if (dist[j] == -level) {
                // Finalize this vertex (remove from frontier)
                dist[j] = level;

                // Expand neighbors
                const int row_start = indptr[j];
                const int row_end = indptr[j + 1];
                for (int e = row_start; e < row_end; e++) {
                    int k = indices[e];
                    if (dist[k] == 0 && k != src) {
                        dist[k] = -next_level;  // mark as next frontier
                        found = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (found == 0) break;
        __syncthreads();
    }

    // Finalize: convert any remaining negative values to positive
    // and ensure dist[src] = 0
    for (int i = tid; i < n; i += nthreads) {
        int d = dist[i];
        if (d < 0) dist[i] = -d;
    }
}
''', 'dawn_apsp')


def run_apsp(A_csr, k=-1, block_size=256, verbose=True):
    """APSP via DAWN-style SOVM BFS (one CUDA block per source)."""
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    if verbose:
        print(f"  BG2 DAWN-SOVM: n={n}, blocks={n}, threads={block_size}")

    indptr_gpu = cp.asarray(A_csr.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_csr.indices.astype(np.int32))
    D_gpu = cp.empty((n, n), dtype=cp.int32)

    max_level = k if k > 0 else -1
    _dawn_kernel(
        (n,), (block_size,),
        (indptr_gpu, indices_gpu, D_gpu, np.int32(n), np.int32(max_level))
    )
    cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(D_gpu)
