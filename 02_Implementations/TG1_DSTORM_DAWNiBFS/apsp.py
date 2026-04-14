"""TG1: D-STORM-DAWNiBFS -- Bitwise frontier-sharing D-STORM with DAWN + iBFS.

Combines D-STORM's all-source simultaneous processing with DAWN's
frontier-driven expansion and iBFS's bitwise packing (SIGMOD 2016).

Key ideas:
  1. Pack BATCH_SIZE sources into W uint32 words (W = BATCH_SIZE/32)
  2. Frontier-driven: only expand vertices in someone's frontier
  3. Frontier sharing: traverse vertex j's neighbors ONCE, bitwise-OR
     distributes results to all BATCH_SIZE sources simultaneously
  4. D serves as footprint: no separate F matrix
  5. No COO output buffer: dense boolean frontier via bitmask
  6. Multi-word bitmask reduces Python batch count (n/BATCH_SIZE iterations)

Memory vs TG2:
  TG2: 16n^2 bytes (D + F + out_row + out_col)
  TG1: 4n^2 + 2*(n*W*4) bytes per batch, W = BATCH_SIZE/32

When compute_sigma=True, a second kernel accumulates shortest-path
counts sigma(s,t) by iterating over the old frontier bitmask and
using atomicAdd on a dense sigma_batch matrix (float64).

References:
  - DAWN: Feng et al., ICS 2024 (frontier-driven SOVM expansion)
  - iBFS: Liu & Huang, SIGMOD 2016 (bitwise multi-source BFS)
"""

import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp


_init_kernel = cp.RawKernel(r'''
extern "C" __global__
void init_batch(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    unsigned int* __restrict__ frontier,  // n * W words
    unsigned int* __restrict__ visited,   // n * W words
    int* __restrict__ D_batch,            // batch_size * n
    const int n,
    const int W,
    const int batch_start,
    const int batch_size
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // Mark sources themselves as visited
    {
        int s_local = j - batch_start;
        if (s_local >= 0 && s_local < batch_size) {
            int word = s_local >> 5;
            unsigned int bit = 1u << (s_local & 31);
            visited[j * W + word] |= bit;
        }
    }

    // Check if j is a 1-hop neighbor of any source in batch
    int start = indptr[j];
    int end = indptr[j + 1];
    for (int e = start; e < end; e++) {
        int u = indices[e];
        int s_local = u - batch_start;
        if (s_local >= 0 && s_local < batch_size) {
            int word = s_local >> 5;
            unsigned int bit = 1u << (s_local & 31);
            frontier[j * W + word] |= bit;
            visited[j * W + word] |= bit;
            D_batch[s_local * n + j] = 1;
        }
    }
}
''', 'init_batch')

_expand_kernel = cp.RawKernel(r'''
extern "C" __global__
void bitwise_expand(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const unsigned int* __restrict__ frontier,  // n * W
    unsigned int* __restrict__ next_frontier,    // n * W
    unsigned int* __restrict__ visited,          // n * W
    int* __restrict__ D_batch,                   // batch_size * n
    const int n,
    const int W,
    const int level,
    int* __restrict__ found
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // Quick check: is j in ANY source's frontier?
    int base = j * W;
    bool has_frontier = false;
    for (int w = 0; w < W; w++) {
        if (frontier[base + w] != 0u) { has_frontier = true; break; }
    }
    if (!has_frontier) return;

    int start = indptr[j];
    int end   = indptr[j + 1];
    for (int e = start; e < end; e++) {
        int k = indices[e];
        int k_base = k * W;

        for (int w = 0; w < W; w++) {
            unsigned int new_bits = frontier[base + w] & ~visited[k_base + w];
            if (new_bits != 0u) {
                atomicOr(&next_frontier[k_base + w], new_bits);
                atomicOr(&visited[k_base + w], new_bits);

                unsigned int bits = new_bits;
                while (bits) {
                    int b = __ffs(bits) - 1;
                    int s_local = w * 32 + b;
                    D_batch[s_local * n + k] = level;
                    bits &= bits - 1;
                }
                found[0] = 1;
            }
        }
    }
}
''', 'bitwise_expand')


_sigma_kernel = cp.RawKernel(r'''
extern "C" __global__
void sigma_accumulate_batch(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const unsigned int* __restrict__ frontier,
    double* __restrict__ sigma_batch,
    const int* __restrict__ D_batch,
    const int n,
    const int W,
    const int level,
    const int batch_size
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int base = j * W;
    bool has_frontier = false;
    for (int w = 0; w < W; w++) {
        if (frontier[base + w] != 0u) { has_frontier = true; break; }
    }
    if (!has_frontier) return;

    int start = indptr[j];
    int end   = indptr[j + 1];

    // For each source s in the old frontier at vertex j,
    // accumulate sigma[s,j] into sigma[s,k] for neighbors k
    // discovered at the current level.
    for (int w = 0; w < W; w++) {
        unsigned int bits = frontier[base + w];
        if (bits == 0u) continue;
        while (bits) {
            int b = __ffs(bits) - 1;
            int s_local = w * 32 + b;
            bits &= bits - 1;
            if (s_local >= batch_size) continue;

            double sig = sigma_batch[s_local * n + j];

            for (int e = start; e < end; e++) {
                int k = indices[e];
                if (D_batch[s_local * n + k] == level) {
                    atomicAdd(&sigma_batch[s_local * n + k], sig);
                }
            }
        }
    }
}
''', 'sigma_accumulate_batch')


def run_apsp(A_csr, k=-1, batch_size=512, verbose=True, compute_sigma=False):
    """APSP via bitwise frontier-sharing D-STORM.

    Args:
        A_csr: Adjacency matrix.
        k: Hop constraint (-1 for full APSP).
        batch_size: Sources per batch (multiple of 32, default 512).
        verbose: Show progress.
        compute_sigma: If True, also compute sigma(s,t) matrix.
            Returns (D, sigma) tuple instead of just D.
            sigma[i,j] = number of shortest paths from i to j.
    """
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    batch_size = min(batch_size, n)
    batch_size = ((batch_size + 31) // 32) * 32
    W = batch_size // 32
    num_batches = (n + batch_size - 1) // batch_size
    if verbose:
        print(f"  TG1 D-STORM-DAWNiBFS: n={n}, batch={batch_size}, W={W}, "
              f"batches={num_batches}" +
              (" (sigma mode)" if compute_sigma else ""))

    indptr_gpu = cp.asarray(A_csr.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_csr.indices.astype(np.int32))

    D_full = cp.zeros((n, n), dtype=cp.int32)

    # Sigma: full n x n matrix (accumulated across batches)
    sigma_full = None
    if compute_sigma:
        sigma_full = cp.zeros((n, n), dtype=cp.float64)
        cp.fill_diagonal(sigma_full, 1.0)

    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    max_level = n if k < 0 else k

    # Reusable buffers
    D_batch = cp.zeros((batch_size, n), dtype=cp.int32)
    frontier = cp.zeros(n * W, dtype=cp.uint32)
    next_frontier = cp.zeros(n * W, dtype=cp.uint32)
    visited = cp.zeros(n * W, dtype=cp.uint32)
    found = cp.zeros(1, dtype=cp.int32)

    # Sigma per-batch buffer
    sigma_batch = None
    if compute_sigma:
        sigma_batch = cp.zeros((batch_size, n), dtype=cp.float64)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        actual = batch_end - batch_start

        D_batch[:] = 0
        frontier[:] = 0
        next_frontier[:] = 0
        visited[:] = 0

        _init_kernel(
            (grid_size,), (block_size,),
            (indptr_gpu, indices_gpu, frontier, visited,
             D_batch, np.int32(n), np.int32(W),
             np.int32(batch_start), np.int32(actual))
        )

        if compute_sigma:
            sigma_batch[:] = 0
            # sigma=1 for direct neighbors (hop 1)
            sigma_batch[D_batch == 1] = 1.0
            # sigma(s,s) = 1 for self-distance
            src_idx = cp.arange(actual, dtype=cp.int32)
            sigma_batch[src_idx, batch_start + src_idx] = 1.0

        level = 2
        while level <= max_level:
            next_frontier[:] = 0
            found[0] = 0

            _expand_kernel(
                (grid_size,), (block_size,),
                (indptr_gpu, indices_gpu, frontier, next_frontier,
                 visited, D_batch, np.int32(n), np.int32(W),
                 np.int32(level), found)
            )
            cp.cuda.Stream.null.synchronize()

            if int(found[0]) == 0:
                break

            # Sigma pass: accumulate using old frontier (before swap)
            # D_batch already has current level values set by expand kernel
            if compute_sigma:
                _sigma_kernel(
                    (grid_size,), (block_size,),
                    (indptr_gpu, indices_gpu, frontier,
                     sigma_batch, D_batch,
                     np.int32(n), np.int32(W),
                     np.int32(level), np.int32(actual))
                )
                cp.cuda.Stream.null.synchronize()

            frontier, next_frontier = next_frontier, frontier
            level += 1

        D_full[batch_start:batch_end, :] = D_batch[:actual, :]
        if compute_sigma:
            sigma_full[batch_start:batch_end, :] = sigma_batch[:actual, :]

    if compute_sigma:
        return cp.asnumpy(D_full), cp.asnumpy(sigma_full)
    return cp.asnumpy(D_full)
