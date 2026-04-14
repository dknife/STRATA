# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython core: BFS + direct-check + forward-propagation affected identification.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef np.int32_t INT32
ctypedef np.uint8_t UINT8


def bfs_single_source(const int[:] indptr, const int[:] indices, int src, int n):
    cdef np.ndarray[INT32, ndim=1] dist = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[UINT8, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef int head = 0, tail = 0, u, v, d_next, idx

    if queue == NULL: raise MemoryError()
    visited[src] = 1; queue[tail] = src; tail += 1

    while head < tail:
        u = queue[head]; head += 1; d_next = dist[u] + 1
        for idx in range(indptr[u], indptr[u + 1]):
            v = indices[idx]
            if visited[v] == 0:
                visited[v] = 1; dist[v] = d_next; queue[tail] = v; tail += 1
    free(queue)
    return dist


def bfs_multi_source(const int[:] indptr, const int[:] indices, const int[:] sources, int n):
    cdef int n_src = sources.shape[0]
    cdef np.ndarray[INT32, ndim=2] D_out = np.zeros((n_src, n), dtype=np.int32)
    cdef np.ndarray[UINT8, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef int head, tail, s, si, u, v, d_next, idx

    if queue == NULL: raise MemoryError()
    for si in range(n_src):
        s = sources[si]
        for v in range(n): visited[v] = 0; D_out[si, v] = 0
        visited[s] = 1; head = 0; tail = 0; queue[tail] = s; tail += 1
        while head < tail:
            u = queue[head]; head += 1; d_next = D_out[si, u] + 1
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                if visited[v] == 0:
                    visited[v] = 1; D_out[si, v] = d_next; queue[tail] = v; tail += 1
    free(queue)
    return D_out


def identify_affected_forward(
    const int[:,:] D,
    const int[:] indptr,
    const int[:] indices,
    int a, int b, int n, int diameter
):
    """
    Two-phase affected identification:
    Phase 1: Direct check — for each source i, check if (i,b)/(i,a) lost
             their only parent due to edge removal. O(n * d_avg).
    Phase 2: Forward propagation — cascade from directly affected pairs.
             O(|affected| * d_avg^2) per level.
    """
    cdef np.ndarray[UINT8, ndim=2] aff = np.zeros((n, n), dtype=np.uint8)
    cdef int i, j, m, k, idx, target, lev, has_alt
    cdef int levels_checked = 1

    # Buffers for newly affected at each level
    # na_i/na_m[level] stores pairs; use flat arrays with level offsets
    cdef int max_buf = n * 100  # generous buffer
    cdef int* na_i_buf = <int*>malloc(max_buf * sizeof(int))
    cdef int* na_m_buf = <int*>malloc(max_buf * sizeof(int))
    # Per-level start/end in buffer
    cdef int* lev_start = <int*>malloc((diameter + 2) * sizeof(int))
    cdef int* lev_end = <int*>malloc((diameter + 2) * sizeof(int))

    cdef int* cand_i = <int*>malloc(max_buf * sizeof(int))
    cdef int* cand_j = <int*>malloc(max_buf * sizeof(int))
    cdef np.ndarray[UINT8, ndim=2] cand_flag = np.zeros((n, n), dtype=np.uint8)

    if (na_i_buf == NULL or na_m_buf == NULL or lev_start == NULL or
        lev_end == NULL or cand_i == NULL or cand_j == NULL):
        if na_i_buf != NULL: free(na_i_buf)
        if na_m_buf != NULL: free(na_m_buf)
        if lev_start != NULL: free(lev_start)
        if lev_end != NULL: free(lev_end)
        if cand_i != NULL: free(cand_i)
        if cand_j != NULL: free(cand_j)
        raise MemoryError()

    cdef int buf_pos = 0
    cdef int min_level = diameter + 1
    cdef int max_level = 0

    for lev in range(diameter + 2):
        lev_start[lev] = 0
        lev_end[lev] = 0

    # ── Phase 1: Direct effect ─────────────────────────────────────
    # Check all sources: did removing (a,b) orphan (i,b) or (i,a)?
    for i in range(n):
        # Check (i, b): a was parent of b from source i?
        if D[i, a] + 1 == D[i, b] and D[i, b] > 0:
            has_alt = 0
            target = D[i, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                m = indices[idx]
                if D[i, m] == target:
                    has_alt = 1
                    break
            if has_alt == 0:
                lev = D[i, b]
                aff[i, b] = 1
                if lev_end[lev] == lev_start[lev]:
                    lev_start[lev] = buf_pos
                na_i_buf[buf_pos] = i
                na_m_buf[buf_pos] = b
                buf_pos += 1
                lev_end[lev] = buf_pos
                if lev < min_level: min_level = lev
                if lev > max_level: max_level = lev

        # Check (i, a): b was parent of a from source i?
        if aff[i, a] == 0 and D[i, b] + 1 == D[i, a] and D[i, a] > 0:
            has_alt = 0
            target = D[i, a] - 1
            for idx in range(indptr[a], indptr[a + 1]):
                m = indices[idx]
                if D[i, m] == target:
                    has_alt = 1
                    break
            if has_alt == 0:
                lev = D[i, a]
                aff[i, a] = 1
                if lev_end[lev] == lev_start[lev]:
                    lev_start[lev] = buf_pos
                na_i_buf[buf_pos] = i
                na_m_buf[buf_pos] = a
                buf_pos += 1
                lev_end[lev] = buf_pos
                if lev < min_level: min_level = lev
                if lev > max_level: max_level = lev

    if buf_pos == 0:
        free(na_i_buf); free(na_m_buf); free(lev_start); free(lev_end)
        free(cand_i); free(cand_j)
        result = np.array([], dtype=np.int32)
        return result, 1

    levels_checked = max_level

    # ── Phase 2: Forward propagation ───────────────────────────────
    cdef int t, cand_count, next_t
    for t in range(min_level, diameter + 1):
        if lev_start[t] == lev_end[t]:
            if t > max_level:
                break
            continue

        next_t = t + 1
        if next_t > diameter:
            break

        # Expand newly affected at level t → candidates at level t+1
        cand_count = 0
        for k in range(lev_start[t], lev_end[t]):
            i = na_i_buf[k]
            m = na_m_buf[k]
            for idx in range(indptr[m], indptr[m + 1]):
                j = indices[idx]
                if D[i, j] == next_t and aff[i, j] == 0 and cand_flag[i, j] == 0:
                    cand_flag[i, j] = 1
                    cand_i[cand_count] = i
                    cand_j[cand_count] = j
                    cand_count += 1

        # Check candidates for alt parent
        if lev_end[next_t] == lev_start[next_t]:
            lev_start[next_t] = buf_pos

        for k in range(cand_count):
            i = cand_i[k]
            j = cand_j[k]
            cand_flag[i, j] = 0  # reset for next use
            has_alt = 0
            target = next_t - 1
            for idx in range(indptr[j], indptr[j + 1]):
                m = indices[idx]
                if D[i, m] == target and aff[i, m] == 0:
                    has_alt = 1
                    break
            if has_alt == 0:
                aff[i, j] = 1
                na_i_buf[buf_pos] = i
                na_m_buf[buf_pos] = j
                buf_pos += 1
                lev_end[next_t] = buf_pos
                if next_t > max_level:
                    max_level = next_t

        levels_checked = max(levels_checked, next_t)

    free(na_i_buf); free(na_m_buf); free(lev_start); free(lev_end)
    free(cand_i); free(cand_j)

    # Collect affected sources
    cdef np.ndarray[UINT8, ndim=1] src_aff = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if aff[i, j] == 1:
                src_aff[i] = 1
                break

    result = np.where(np.asarray(src_aff) > 0)[0].astype(np.int32)
    return result, levels_checked
