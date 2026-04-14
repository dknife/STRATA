"""
EdgeMan_DLevel_G: GPU level-based edge insertion/deletion with cascade tracking.

- Insertion: O(n^2) vectorized on GPU via CuPy.
- Deletion: GPU kernel performs level-by-level cascade identification.
  Each level: one thread per (i,j) pair checks alternative parent.
  Sync barrier between levels (kernel launches). Early termination.
  Then GPU BFS recomputes affected rows.
"""

import numpy as np
import scipy.sparse as sp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from cuda_env import setup_cuda_env
setup_cuda_env()

import cupy as cp

# ─── CUDA Kernels ─────────────────────────────────────────────────────

# Kernel 1: Level-based affected pair identification (one thread per pair at level t)
_level_check_kernel = cp.RawKernel(r'''
extern "C" __global__
void level_check(
    const int* __restrict__ D,        // n x n distance matrix
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const unsigned char* __restrict__ aff,  // n x n affected flags (input)
    unsigned char* __restrict__ aff_out,    // n x n affected flags (output, copy of aff + new)
    const int* __restrict__ pair_rows,      // indices of pairs at this level
    const int* __restrict__ pair_cols,
    int* __restrict__ found_flag,           // global: set to 1 if any new affected
    const int n_pairs,
    const int target_dist,                  // t - 1
    const int n
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;

    const int i = pair_rows[k];
    const int j = pair_cols[k];

    // Check: does j have alt parent m with D[i,m]==target_dist and aff[i,m]==0?
    int has_alt = 0;
    for (int e = indptr[j]; e < indptr[j + 1]; e++) {
        int m = indices[e];
        if (D[(long long)i * n + m] == target_dist && aff[(long long)i * n + m] == 0) {
            has_alt = 1;
            break;
        }
    }

    if (!has_alt) {
        aff_out[(long long)i * n + j] = 1;
        atomicExch(found_flag, 1);
    }
}
''', 'level_check')


# Kernel 2: Per-source BFS (same as BFS_G)
_bfs_kernel = cp.RawKernel(r'''
extern "C" __global__
void bfs_multi(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    int* __restrict__ D,
    const int* __restrict__ sources,
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


# ─── Edge Deletion (Phase 1: Direct Check + Phase 2: Forward Propagation) ──

# Kernel: Phase 1 — identify directly affected pairs (one thread per source)
_direct_check_kernel = cp.RawKernel(r'''
extern "C" __global__
void direct_check(
    const int* __restrict__ D,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    unsigned char* __restrict__ aff,   // n x n, write directly affected pairs
    const int n,
    const int a,
    const int b
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int Dia = D[(long long)i * n + a];
    int Dib = D[(long long)i * n + b];

    // Check (i, b): was 'a' the only parent of b from source i?
    // a is removed from N(b), so check remaining neighbors of b
    if (Dia + 1 == Dib && Dib > 0) {
        int target = Dib - 1;
        int has_alt = 0;
        for (int e = indptr[b]; e < indptr[b + 1]; e++) {
            int m = indices[e];
            if (D[(long long)i * n + m] == target) {
                has_alt = 1;
                break;
            }
        }
        if (!has_alt) {
            aff[(long long)i * n + b] = 1;
        }
    }

    // Check (i, a): was 'b' the only parent of a from source i?
    if (Dib + 1 == Dia && Dia > 0) {
        int target = Dia - 1;
        int has_alt = 0;
        for (int e = indptr[a]; e < indptr[a + 1]; e++) {
            int m = indices[e];
            if (D[(long long)i * n + m] == target) {
                has_alt = 1;
                break;
            }
        }
        if (!has_alt) {
            aff[(long long)i * n + a] = 1;
        }
    }
}
''', 'direct_check')

# Kernel for checking alt parents of candidate pairs
_check_candidates_kernel = cp.RawKernel(r'''
extern "C" __global__
void check_candidates(
    const int* __restrict__ D,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const unsigned char* __restrict__ aff,
    unsigned char* __restrict__ aff_out,       // write new affected here
    const int* __restrict__ cand_rows,
    const int* __restrict__ cand_cols,
    int* __restrict__ new_aff_rows,            // output: newly affected pairs
    int* __restrict__ new_aff_cols,
    int* __restrict__ new_count,               // atomic counter
    const int n_cands,
    const int target_dist,
    const int n
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_cands) return;

    const int i = cand_rows[k];
    const int j = cand_cols[k];

    int has_alt = 0;
    for (int e = indptr[j]; e < indptr[j + 1]; e++) {
        int m = indices[e];
        if (D[(long long)i * n + m] == target_dist && aff[(long long)i * n + m] == 0) {
            has_alt = 1;
            break;
        }
    }

    if (!has_alt) {
        aff_out[(long long)i * n + j] = 1;
        int pos = atomicAdd(new_count, 1);
        new_aff_rows[pos] = i;
        new_aff_cols[pos] = j;
    }
}
''', 'check_candidates')

# Kernel to expand newly affected pairs into candidates for next level
_expand_candidates_kernel = cp.RawKernel(r'''
extern "C" __global__
void expand_candidates(
    const int* __restrict__ D,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const int* __restrict__ na_rows,     // newly affected (i, m) pairs
    const int* __restrict__ na_cols,
    int* __restrict__ cand_flag,            // n x n dedup flag (int for atomicExch)
    int* __restrict__ cand_rows,
    int* __restrict__ cand_cols,
    int* __restrict__ cand_count,        // atomic counter
    const int na_size,
    const int target_level,              // t (next level)
    const int n
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= na_size) return;

    const int i = na_rows[k];
    const int m = na_cols[k];

    // Expand: for each neighbor j of m, if D[i,j] == target_level
    for (int e = indptr[m]; e < indptr[m + 1]; e++) {
        int j = indices[e];
        if (D[(long long)i * n + j] == target_level) {
            // Dedup via atomicExch on cand_flag
            int old = atomicExch(&cand_flag[(long long)i * n + j], 1);
            if (old == 0) {
                int pos = atomicAdd(cand_count, 1);
                cand_rows[pos] = i;
                cand_cols[pos] = j;
            }
        }
    }
}
''', 'expand_candidates')


def delete_edge(D, A_csr, a, b):
    """
    Delete undirected edge {a,b}. GPU two-phase approach:
    Phase 1: GPU kernel identifies directly affected pairs (one thread per source).
    Phase 2: Forward propagation cascade from affected pairs.
    Returns (A_new, n_affected, n_levels_checked).
    """
    n = D.shape[0]
    diameter = int(D.max())

    A_lil = A_csr.tolil()
    A_lil[a, b] = 0
    A_lil[b, a] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    indptr_gpu = cp.asarray(A_new.indptr.astype(np.int32))
    indices_gpu = cp.asarray(A_new.indices.astype(np.int32))
    D_gpu = cp.asarray(D.astype(np.int32))

    aff_gpu = cp.zeros((n, n), dtype=cp.uint8)

    # ── Phase 1: Direct check (GPU, one thread per source) ─────────
    # For each source i, check if (i,b) or (i,a) lost their only parent
    direct_aff_gpu = cp.zeros(n, dtype=cp.int32)  # reuse BFS_G kernel
    threads = 256
    blocks = (n + threads - 1) // threads
    _direct_check_kernel(
        (blocks,), (threads,),
        (D_gpu, indptr_gpu, indices_gpu, aff_gpu,
         np.int32(n), np.int32(a), np.int32(b))
    )
    cp.cuda.Stream.null.synchronize()

    # Collect initial affected pairs and group by level
    init_rows, init_cols = cp.where(aff_gpu > 0)
    n_init = int(init_rows.shape[0])

    if n_init == 0:
        return A_new, 0, 1

    # Get levels for initial affected pairs
    init_levels = D_gpu[init_rows, init_cols]
    max_level = int(init_levels.max())
    levels_checked = max_level

    # ── Phase 2: Forward propagation (cascade) ─────────────────────
    max_buf = min(n * n, 1000000)
    cand_rows_gpu = cp.empty(max_buf, dtype=cp.int32)
    cand_cols_gpu = cp.empty(max_buf, dtype=cp.int32)
    new_rows_gpu = cp.empty(max_buf, dtype=cp.int32)
    new_cols_gpu = cp.empty(max_buf, dtype=cp.int32)
    cand_flag_gpu = cp.zeros((n, n), dtype=cp.int32)

    for t in range(1, diameter + 1):
        # Get newly affected at this level
        mask_t = (init_levels == t) if t <= max_level else cp.zeros(0, dtype=cp.bool_)
        if isinstance(mask_t, cp.ndarray) and mask_t.any():
            na_rows_t = init_rows[mask_t].astype(cp.int32)
            na_cols_t = init_cols[mask_t].astype(cp.int32)
        else:
            na_rows_t = cp.empty(0, dtype=cp.int32)
            na_cols_t = cp.empty(0, dtype=cp.int32)

        na_size = int(na_rows_t.shape[0])
        if na_size == 0:
            if t > max_level:
                break
            continue

        # Forward: expand to candidates at t+1
        next_t = t + 1
        if next_t > diameter:
            break

        cand_count_gpu = cp.zeros(1, dtype=cp.int32)
        blk = (na_size + threads - 1) // threads
        _expand_candidates_kernel(
            (blk,), (threads,),
            (D_gpu, indptr_gpu, indices_gpu,
             na_rows_t, na_cols_t,
             cand_flag_gpu, cand_rows_gpu, cand_cols_gpu, cand_count_gpu,
             np.int32(na_size), np.int32(next_t), np.int32(n))
        )
        cp.cuda.Stream.null.synchronize()

        n_cands = int(cand_count_gpu[0])

        # Reset cand flags
        if n_cands > 0:
            cand_flag_gpu[cand_rows_gpu[:n_cands], cand_cols_gpu[:n_cands]] = 0

        if n_cands == 0:
            continue

        # Check candidates for alt parent
        new_count_gpu = cp.zeros(1, dtype=cp.int32)
        blk = (n_cands + threads - 1) // threads
        _check_candidates_kernel(
            (blk,), (threads,),
            (D_gpu, indptr_gpu, indices_gpu, aff_gpu, aff_gpu,
             cand_rows_gpu, cand_cols_gpu,
             new_rows_gpu, new_cols_gpu, new_count_gpu,
             np.int32(n_cands), np.int32(next_t - 1), np.int32(n))
        )
        cp.cuda.Stream.null.synchronize()

        n_new = int(new_count_gpu[0])
        if n_new > 0:
            # Add cascade-discovered pairs as newly affected at next_t
            new_r = new_rows_gpu[:n_new]
            new_c = new_cols_gpu[:n_new]
            init_rows = cp.concatenate([init_rows, new_r])
            init_cols = cp.concatenate([init_cols, new_c])
            new_levels = cp.full(n_new, next_t, dtype=cp.int32)
            init_levels = cp.concatenate([init_levels, new_levels])
            if next_t > max_level:
                max_level = next_t

        levels_checked = max(levels_checked, next_t)

    # Collect affected sources
    src_affected = cp.any(aff_gpu, axis=1)
    aff_sources = cp.where(src_affected)[0].astype(cp.int32)
    n_affected = int(aff_sources.shape[0])

    # GPU BFS recompute
    if n_affected > 0:
        D_out_gpu = cp.empty((n_affected, n), dtype=cp.int32)
        _bfs_kernel(
            (n_affected,), (256,),
            (indptr_gpu, indices_gpu, D_out_gpu, aff_sources,
             np.int32(n), np.int32(n_affected))
        )
        cp.cuda.Stream.null.synchronize()

        D_out = cp.asnumpy(D_out_gpu)
        aff_list = cp.asnumpy(aff_sources)
        for k, s in enumerate(aff_list):
            D[s, :] = D_out[k]

    return A_new, n_affected, levels_checked
