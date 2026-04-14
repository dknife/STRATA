# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython core routines for EdgeMan_BFS: BFS + source-based affected identification.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef np.int32_t INT32
ctypedef np.uint8_t UINT8


def bfs_single_source(
    const int[:] indptr,
    const int[:] indices,
    int src,
    int n
):
    """BFS from single source on CSR graph. Returns int32 distance array."""
    cdef np.ndarray[INT32, ndim=1] dist = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[UINT8, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef int head = 0, tail = 0
    cdef int u, v, d_next, idx

    if queue == NULL:
        raise MemoryError()

    visited[src] = 1
    queue[tail] = src
    tail += 1

    while head < tail:
        u = queue[head]
        head += 1
        d_next = dist[u] + 1
        for idx in range(indptr[u], indptr[u + 1]):
            v = indices[idx]
            if visited[v] == 0:
                visited[v] = 1
                dist[v] = d_next
                queue[tail] = v
                tail += 1

    free(queue)
    return dist


def bfs_multi_source(
    const int[:] indptr,
    const int[:] indices,
    const int[:] sources,
    int n
):
    """BFS from multiple sources. Returns (n_sources, n) int32 distance matrix rows."""
    cdef int n_src = sources.shape[0]
    cdef np.ndarray[INT32, ndim=2] D_out = np.zeros((n_src, n), dtype=np.int32)
    cdef np.ndarray[UINT8, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    cdef int* queue = <int*>malloc(n * sizeof(int))
    cdef int head, tail, s, si, u, v, d_next, idx

    if queue == NULL:
        raise MemoryError()

    for si in range(n_src):
        s = sources[si]
        # Reset
        for v in range(n):
            visited[v] = 0
            D_out[si, v] = 0
        visited[s] = 1
        head = 0
        tail = 0
        queue[tail] = s
        tail += 1

        while head < tail:
            u = queue[head]
            head += 1
            d_next = D_out[si, u] + 1
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                if visited[v] == 0:
                    visited[v] = 1
                    D_out[si, v] = d_next
                    queue[tail] = v
                    tail += 1

    free(queue)
    return D_out


def identify_affected_deletion(
    const int[:,:] D,
    const int[:] indptr,
    const int[:] indices,
    int a,
    int b,
    int n
):
    """
    Source-based affected identification for undirected edge {a,b} deletion.
    Returns array of affected source indices.
    """
    cdef np.ndarray[UINT8, ndim=1] affected = np.zeros(n, dtype=np.uint8)
    cdef int s, idx, u, target_dist
    cdef int has_alt

    for s in range(n):
        # Check direction a->b: was (a,b) on shortest path s->b?
        if D[s, a] + 1 == D[s, b] and D[s, b] > 0:
            has_alt = 0
            target_dist = D[s, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                u = indices[idx]
                if u != a and D[s, u] == target_dist:
                    has_alt = 1
                    break
            if has_alt == 0:
                affected[s] = 1

        # Check direction b->a: was (b,a) on shortest path s->a?
        if affected[s] == 0 and D[s, b] + 1 == D[s, a] and D[s, a] > 0:
            has_alt = 0
            target_dist = D[s, a] - 1
            for idx in range(indptr[a], indptr[a + 1]):
                u = indices[idx]
                if u != b and D[s, u] == target_dist:
                    has_alt = 1
                    break
            if has_alt == 0:
                affected[s] = 1

    return np.where(np.asarray(affected) > 0)[0].astype(np.int32)
