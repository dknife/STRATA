"""
EdgeMan_BFS_G: GPU source-based edge insertion/deletion.

- Insertion: O(n^2) vectorized on GPU via CuPy.
- Deletion: GPU kernel identifies affected sources (one thread per source),
  then GPU per-source BFS recomputes affected rows.
"""

import numpy as np
import scipy.sparse as sp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp

# ─── CUDA Kernels ─────────────────────────────────────────────────────

# Kernel 1: Identify affected sources (one thread per source)
_identify_kernel = cp.RawKernel(r'''
extern "C" __global__
void identify_affected_bfs(
    const int* __restrict__ D,      // n x n distance matrix (row-major)
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    int* __restrict__ affected,     // output: 1 if source s is affected
    const int n,
    const int a,
    const int b
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;

    // Check direction a->b: D[s,a]+1 == D[s,b]?
    int Dsa = D[(long long)s * n + a];
    int Dsb = D[(long long)s * n + b];

    int is_affected = 0;

    if (Dsa + 1 == Dsb && Dsb > 0) {
        int target = Dsb - 1;
        int has_alt = 0;
        for (int e = indptr[b]; e < indptr[b + 1]; e++) {
            int u = indices[e];
            if (u != a && D[(long long)s * n + u] == target) {
                has_alt = 1;
                break;
            }
        }
        if (!has_alt) is_affected = 1;
    }

    // Check direction b->a: D[s,b]+1 == D[s,a]?
    if (!is_affected && Dsb + 1 == Dsa && Dsa > 0) {
        int target = Dsa - 1;
        int has_alt = 0;
        for (int e = indptr[a]; e < indptr[a + 1]; e++) {
            int u = indices[e];
            if (u != b && D[(long long)s * n + u] == target) {
                has_alt = 1;
                break;
            }
        }
        if (!has_alt) is_affected = 1;
    }

    affected[s] = is_affected;
}
''', 'identify_affected_bfs')


# Kernel 2: Per-source BFS (one block per source, same as BG1)
_bfs_kernel = cp.RawKernel(r'''
extern "C" __global__
void bfs_multi(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    int* __restrict__ D,            // n_sources x n output
    const int* __restrict__ sources, // source vertex list
    const int n,
    const int n_sources
) {
    const int si = blockIdx.x;
    if (si >= n_sources) return;
    const int src = sources[si];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    int* dist = D + (long long)si * n;

    for (int i = tid; i < n; i += nthreads)
        dist[i] = -1;
    __syncthreads();
    if (tid == 0) dist[src] = 0;
    __syncthreads();

    for (int level = 0; level < n - 1; level++) {
        __shared__ int found;
        if (tid == 0) found = 0;
        __syncthreads();

        for (int u = tid; u < n; u += nthreads) {
            if (dist[u] == level) {
                for (int e = indptr[u]; e < indptr[u + 1]; e++) {
                    int v = indices[e];
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
''', 'bfs_multi')


# ─── Edge Insertion ───────────────────────────────────────────────────

def insert_edge(D, A_csr, a, b):
    """Insert undirected edge {a,b}. GPU-accelerated O(n^2) update."""
    n = D.shape[0]

    A_lil = A_csr.tolil()
    A_lil[a, b] = 1
    A_lil[b, a] = 1
    A_new = A_lil.tocsr()

    D_gpu = cp.asarray(D)
    Da = D_gpu[a, :].copy()
    Db = D_gpu[b, :].copy()
    Di_a = D_gpu[:, a].copy()
    Di_b = D_gpu[:, b].copy()

    term1 = Di_a[:, cp.newaxis] + 1 + Db[cp.newaxis, :]
    term2 = Di_b[:, cp.newaxis] + 1 + Da[cp.newaxis, :]

    can_reach_a = (Di_a > 0)
    can_reach_a[a] = True
    can_b_reach = (Db > 0)
    can_b_reach[b] = True
    can_reach_b = (Di_b > 0)
    can_reach_b[b] = True
    can_a_reach = (Da > 0)
    can_a_reach[a] = True

    valid1 = can_reach_a[:, cp.newaxis] & can_b_reach[cp.newaxis, :]
    valid2 = can_reach_b[:, cp.newaxis] & can_a_reach[cp.newaxis, :]

    INF = np.int32(n + 1)
    mask_d = D_gpu > 0
    mask_self = cp.eye(n, dtype=cp.bool_)
    D_work = cp.where(mask_d | mask_self, D_gpu, INF).astype(cp.int32)
    t1 = cp.where(valid1, term1, INF).astype(cp.int32)
    t2 = cp.where(valid2, term2, INF).astype(cp.int32)

    D_new = cp.minimum(D_work, cp.minimum(t1, t2))
    D_new[D_new >= INF] = 0
    cp.fill_diagonal(D_new, 0)

    np.copyto(D, cp.asnumpy(D_new))
    return A_new


def insert_edge_directed(D, A_csr, a, b):
    """Insert directed edge a->b. GPU-accelerated."""
    n = D.shape[0]

    A_lil = A_csr.tolil()
    A_lil[a, b] = 1
    A_new = A_lil.tocsr()

    D_gpu = cp.asarray(D)
    Di_a = D_gpu[:, a].copy()
    Db = D_gpu[b, :].copy()

    term1 = Di_a[:, cp.newaxis] + 1 + Db[cp.newaxis, :]

    can_reach_a = (Di_a > 0)
    can_reach_a[a] = True
    can_b_reach = (Db > 0)
    can_b_reach[b] = True
    valid1 = can_reach_a[:, cp.newaxis] & can_b_reach[cp.newaxis, :]

    INF = np.int32(n + 1)
    mask_d = D_gpu > 0
    mask_self = cp.eye(n, dtype=cp.bool_)
    D_work = cp.where(mask_d | mask_self, D_gpu, INF).astype(cp.int32)
    t1 = cp.where(valid1, term1, INF).astype(cp.int32)

    D_new = cp.minimum(D_work, t1)
    D_new[D_new >= INF] = 0
    cp.fill_diagonal(D_new, 0)

    np.copyto(D, cp.asnumpy(D_new))
    return A_new


# ─── Edge Deletion ────────────────────────────────────────────────────

def delete_edge(D, A_csr, a, b):
    """
    Delete undirected edge {a,b}. GPU source-based approach.
    Returns (A_new, n_affected).
    """
    n = D.shape[0]

    A_lil = A_csr.tolil()
    A_lil[a, b] = 0
    A_lil[b, a] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    indptr_gpu = cp.asarray(A_new.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_new.indices.astype(np.int32))
    D_gpu = cp.asarray(D.astype(np.int32))
    affected_gpu = cp.zeros(n, dtype=cp.int32)

    # Step 1: GPU identify affected sources
    threads = 256
    blocks = (n + threads - 1) // threads
    _identify_kernel(
        (blocks,), (threads,),
        (D_gpu, indptr_gpu, indices_gpu, affected_gpu,
         np.int32(n), np.int32(a), np.int32(b))
    )
    cp.cuda.Stream.null.synchronize()

    aff_sources = cp.where(affected_gpu > 0)[0].astype(cp.int32)
    n_affected = int(aff_sources.shape[0])

    # Step 2: GPU BFS recompute for affected sources
    if n_affected > 0:
        D_out_gpu = cp.empty((n_affected, n), dtype=cp.int32)
        block_size = 256
        _bfs_kernel(
            (n_affected,), (block_size,),
            (indptr_gpu, indices_gpu, D_out_gpu, aff_sources,
             np.int32(n), np.int32(n_affected))
        )
        cp.cuda.Stream.null.synchronize()

        D_out = cp.asnumpy(D_out_gpu)
        aff_list = cp.asnumpy(aff_sources)
        for k, s in enumerate(aff_list):
            D[s, :] = D_out[k]

    return A_new, n_affected
