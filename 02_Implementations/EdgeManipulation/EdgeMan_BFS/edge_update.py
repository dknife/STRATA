"""
EdgeMan_BFS: Source-based edge insertion/deletion with BFS recomputation.

- Insertion: O(n^2) closed-form update (undirected: both direction terms).
- Deletion: Per-source alternative parent check + BFS fallback for affected sources.
  No cascade tracking — any source without alt parent triggers full BFS rerun.

Uses Cython _core if available, falls back to pure Python.
"""

import numpy as np
import scipy.sparse as sp
from collections import deque

# ─── Cython / Python backend selection ────────────────────────────────

try:
    from EdgeManipulation.EdgeMan_BFS._core import (
        bfs_single_source as _cy_bfs,
        bfs_multi_source as _cy_bfs_multi,
        identify_affected_deletion as _cy_identify,
    )
    _USE_CYTHON = True
except ImportError:
    try:
        from ._core import (
            bfs_single_source as _cy_bfs,
            bfs_multi_source as _cy_bfs_multi,
            identify_affected_deletion as _cy_identify,
        )
        _USE_CYTHON = True
    except ImportError:
        _USE_CYTHON = False


def _py_bfs(A_csr, src, n):
    """Pure Python BFS fallback."""
    dist = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    visited[src] = True
    queue = deque([src])
    indptr = A_csr.indptr
    indices = A_csr.indices
    while queue:
        u = queue.popleft()
        d_next = dist[u] + 1
        for idx in range(indptr[u], indptr[u + 1]):
            v = indices[idx]
            if not visited[v]:
                visited[v] = True
                dist[v] = d_next
                queue.append(v)
    return dist


# ─── Edge Insertion ───────────────────────────────────────────────────

def _apply_insertion_formula(D, a, b, n, undirected=True):
    """Vectorized insertion update. Shared by both directed/undirected."""
    Da = D[a, :].copy()
    Db = D[b, :].copy()
    Di_a = D[:, a].copy()
    Di_b = D[:, b].copy()

    # term1[i,j] = D[i,a] + 1 + D[b,j]
    term1 = Di_a[:, np.newaxis] + 1 + Db[np.newaxis, :]

    can_reach_a = (Di_a > 0)
    can_reach_a[a] = True
    can_b_reach = (Db > 0)
    can_b_reach[b] = True
    valid1 = can_reach_a[:, np.newaxis] & can_b_reach[np.newaxis, :]

    INF = n + 1
    mask_d = D > 0
    mask_self = np.eye(n, dtype=np.bool_)
    D_work = np.where(mask_d | mask_self, D, INF).astype(np.int32)
    t1 = np.where(valid1, term1, INF).astype(np.int32)

    if undirected:
        # term2[i,j] = D[i,b] + 1 + D[a,j]
        term2 = Di_b[:, np.newaxis] + 1 + Da[np.newaxis, :]
        can_reach_b = (Di_b > 0)
        can_reach_b[b] = True
        can_a_reach = (Da > 0)
        can_a_reach[a] = True
        valid2 = can_reach_b[:, np.newaxis] & can_a_reach[np.newaxis, :]
        t2 = np.where(valid2, term2, INF).astype(np.int32)
        D_new = np.minimum(D_work, np.minimum(t1, t2))
    else:
        D_new = np.minimum(D_work, t1)

    D_new[D_new >= INF] = 0
    np.fill_diagonal(D_new, 0)
    np.copyto(D, D_new)


def insert_edge(D, A_csr, a, b):
    """Insert undirected edge {a, b}. Updates D in-place, returns modified A_csr."""
    n = D.shape[0]
    A_lil = A_csr.tolil()
    A_lil[a, b] = 1
    A_lil[b, a] = 1
    A_new = A_lil.tocsr()
    _apply_insertion_formula(D, a, b, n, undirected=True)
    return A_new


def insert_edge_directed(D, A_csr, a, b):
    """Insert directed edge a -> b. Updates D in-place, returns modified A_csr."""
    n = D.shape[0]
    A_lil = A_csr.tolil()
    A_lil[a, b] = 1
    A_new = A_lil.tocsr()
    _apply_insertion_formula(D, a, b, n, undirected=False)
    return A_new


# ─── Edge Deletion ────────────────────────────────────────────────────

def delete_edge(D, A_csr, a, b):
    """
    Delete undirected edge {a, b}. Updates D in-place, returns modified A_csr.
    Returns (A_new, n_affected).
    """
    n = D.shape[0]

    A_lil = A_csr.tolil()
    A_lil[a, b] = 0
    A_lil[b, a] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    indptr = A_new.indptr.astype(np.int32)
    indices = A_new.indices.astype(np.int32)

    if _USE_CYTHON:
        D_i32 = np.ascontiguousarray(D, dtype=np.int32)
        aff_sources = _cy_identify(D_i32, indptr, indices, a, b, n)
        if len(aff_sources) > 0:
            D_rows = _cy_bfs_multi(indptr, indices, aff_sources, n)
            for k, s in enumerate(aff_sources):
                D[s, :] = D_rows[k]
    else:
        aff_sources = _py_identify_affected(D, indptr, indices, a, b, n)
        for s in aff_sources:
            D[s, :] = _py_bfs(A_new, s, n)

    return A_new, len(aff_sources)


def delete_edge_directed(D, A_csr, a, b):
    """Delete directed edge a -> b. Updates D in-place, returns modified A_csr."""
    n = D.shape[0]

    A_lil = A_csr.tolil()
    A_lil[a, b] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    indptr = A_new.indptr.astype(np.int32)
    indices = A_new.indices.astype(np.int32)

    # Directed: only check a->b direction
    affected_sources = []
    for s in range(n):
        if D[s, a] + 1 == D[s, b] and D[s, b] > 0:
            has_alt = False
            target_dist = D[s, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                u = indices[idx]
                if u != a and D[s, u] == target_dist:
                    has_alt = True
                    break
            if not has_alt:
                affected_sources.append(s)

    if _USE_CYTHON and len(affected_sources) > 0:
        src_arr = np.array(affected_sources, dtype=np.int32)
        D_rows = _cy_bfs_multi(indptr, indices, src_arr, n)
        for k, s in enumerate(affected_sources):
            D[s, :] = D_rows[k]
    else:
        for s in affected_sources:
            D[s, :] = _py_bfs(A_new, s, n)

    return A_new, len(affected_sources)


def _py_identify_affected(D, indptr, indices, a, b, n):
    """Pure Python fallback for source-based affected identification."""
    affected = []
    for s in range(n):
        found = False
        if D[s, a] + 1 == D[s, b] and D[s, b] > 0:
            has_alt = False
            target_dist = D[s, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                u = indices[idx]
                if u != a and D[s, u] == target_dist:
                    has_alt = True
                    break
            if not has_alt:
                found = True

        if not found and D[s, b] + 1 == D[s, a] and D[s, a] > 0:
            has_alt = False
            target_dist = D[s, a] - 1
            for idx in range(indptr[a], indptr[a + 1]):
                u = indices[idx]
                if u != b and D[s, u] == target_dist:
                    has_alt = True
                    break
            if not has_alt:
                found = True

        if found:
            affected.append(s)
    return affected
