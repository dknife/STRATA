"""
EdgeMan_DLevel: Level-based edge insertion/deletion with cascade tracking.

- Insertion: O(n^2) closed-form update (same formula as BFS version).
- Deletion: Two-phase approach:
  Phase 1 — Identify directly affected pairs: same as BFS source check,
    O(n * d_avg). Edge removal orphans (i,b)/(i,a) when a/b was the only parent.
  Phase 2 — Forward propagation: cascade from directly affected pairs through
    edges. Only pairs reachable from affected are candidates.
    Cost: O(|affected| * d_avg^2) per level vs O(P_t * d_avg) in full-scan.

Uses Cython _core if available, falls back to pure Python.
"""

import numpy as np
import scipy.sparse as sp
from collections import deque, defaultdict

# ─── Cython / Python backend selection ────────────────────────────────

try:
    from EdgeManipulation.EdgeMan_DLevel._core import (
        bfs_single_source as _cy_bfs,
        bfs_multi_source as _cy_bfs_multi,
        identify_affected_forward as _cy_identify_fwd,
    )
    _USE_CYTHON = True
except ImportError:
    try:
        from ._core import (
            bfs_single_source as _cy_bfs,
            bfs_multi_source as _cy_bfs_multi,
            identify_affected_forward as _cy_identify_fwd,
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
    """Vectorized insertion update."""
    Da = D[a, :].copy()
    Db = D[b, :].copy()
    Di_a = D[:, a].copy()
    Di_b = D[:, b].copy()

    term1 = Di_a[:, np.newaxis] + 1 + Db[np.newaxis, :]
    can_reach_a = (Di_a > 0); can_reach_a[a] = True
    can_b_reach = (Db > 0);   can_b_reach[b] = True
    valid1 = can_reach_a[:, np.newaxis] & can_b_reach[np.newaxis, :]

    INF = n + 1
    mask_d = D > 0
    mask_self = np.eye(n, dtype=np.bool_)
    D_work = np.where(mask_d | mask_self, D, INF).astype(np.int32)
    t1 = np.where(valid1, term1, INF).astype(np.int32)

    if undirected:
        term2 = Di_b[:, np.newaxis] + 1 + Da[np.newaxis, :]
        can_reach_b = (Di_b > 0); can_reach_b[b] = True
        can_a_reach = (Da > 0);   can_a_reach[a] = True
        valid2 = can_reach_b[:, np.newaxis] & can_a_reach[np.newaxis, :]
        t2 = np.where(valid2, term2, INF).astype(np.int32)
        D_new = np.minimum(D_work, np.minimum(t1, t2))
    else:
        D_new = np.minimum(D_work, t1)

    D_new[D_new >= INF] = 0
    np.fill_diagonal(D_new, 0)
    np.copyto(D, D_new)


def insert_edge(D, A_csr, a, b):
    n = D.shape[0]
    A_lil = A_csr.tolil(); A_lil[a, b] = 1; A_lil[b, a] = 1
    A_new = A_lil.tocsr()
    _apply_insertion_formula(D, a, b, n, undirected=True)
    return A_new


def insert_edge_directed(D, A_csr, a, b):
    n = D.shape[0]
    A_lil = A_csr.tolil(); A_lil[a, b] = 1
    A_new = A_lil.tocsr()
    _apply_insertion_formula(D, a, b, n, undirected=False)
    return A_new


# ─── Edge Deletion (Direct Check + Forward Propagation) ──────────────

def delete_edge(D, A_csr, a, b):
    """
    Delete undirected edge {a, b}. Updates D in-place, returns modified A_csr.
    Returns (A_new, n_affected, n_levels_checked).
    """
    n = D.shape[0]
    diameter = int(D.max())

    A_lil = A_csr.tolil()
    A_lil[a, b] = 0
    A_lil[b, a] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    indptr = A_new.indptr.astype(np.int32)
    indices = A_new.indices.astype(np.int32)

    if _USE_CYTHON:
        D_i32 = np.ascontiguousarray(D, dtype=np.int32)
        aff_sources, levels_checked = _cy_identify_fwd(
            D_i32, indptr, indices, a, b, n, diameter
        )
        if len(aff_sources) > 0:
            D_rows = _cy_bfs_multi(indptr, indices, aff_sources, n)
            for k, s in enumerate(aff_sources):
                D[s, :] = D_rows[k]
    else:
        aff_sources, levels_checked = _py_identify_forward(
            D, indptr, indices, a, b, n, diameter
        )
        for s in aff_sources:
            D[s, :] = _py_bfs(A_new, s, n)

    return A_new, len(aff_sources), levels_checked


def delete_edge_directed(D, A_csr, a, b):
    n = D.shape[0]
    diameter = int(D.max())
    A_lil = A_csr.tolil(); A_lil[a, b] = 0
    A_new = A_lil.tocsr(); A_new.eliminate_zeros()
    indptr = A_new.indptr.astype(np.int32)
    indices = A_new.indices.astype(np.int32)

    aff_sources, levels_checked = _py_identify_forward_directed(
        D, indptr, indices, a, b, n, diameter
    )
    for s in aff_sources:
        D[s, :] = _py_bfs(A_new, s, n)
    return A_new, len(aff_sources), levels_checked


def _py_identify_forward(D, indptr, indices, a, b, n, diameter):
    """
    Phase 1: Identify directly affected pairs — for each source i, check if
    (i,b) or (i,a) lost their only parent due to edge removal. O(n * d_avg).

    Phase 2: Forward propagation — cascade from directly affected pairs.
    """
    aff = np.zeros((n, n), dtype=np.bool_)

    # Phase 1: Direct effect — check all sources
    # Group directly affected pairs by level for ordered forward propagation
    newly_affected_by_level = defaultdict(list)

    for i in range(n):
        # Check (i, b): was a a parent of b from source i?
        # a is no longer in N(b), so if a was the only parent → affected
        if D[i, a] + 1 == D[i, b] and D[i, b] > 0:
            # Check remaining neighbors of b in A' for alt parent
            has_alt = False
            target = D[i, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                m = indices[idx]
                if D[i, m] == target:
                    has_alt = True
                    break
            if not has_alt:
                lev = int(D[i, b])
                aff[i, b] = True
                newly_affected_by_level[lev].append((i, b))

        # Check (i, a): was b a parent of a from source i?
        if not aff[i, a] and D[i, b] + 1 == D[i, a] and D[i, a] > 0:
            has_alt = False
            target = D[i, a] - 1
            for idx in range(indptr[a], indptr[a + 1]):
                m = indices[idx]
                if D[i, m] == target:
                    has_alt = True
                    break
            if not has_alt:
                lev = int(D[i, a])
                aff[i, a] = True
                newly_affected_by_level[lev].append((i, a))

    if not newly_affected_by_level:
        return [], 1

    min_level = min(newly_affected_by_level.keys())
    max_level = max(newly_affected_by_level.keys())
    levels_checked = max_level

    # Phase 2: Forward propagation — cascade from directly affected
    for t in range(min_level, diameter + 1):
        # Combine direct affected at this level + cascade from previous
        newly_affected = newly_affected_by_level.get(t, [])

        if not newly_affected:
            # No new affected at this level, but might have some at higher levels
            # from direct effect
            if t > max_level:
                break
            continue

        # Forward: find candidates at level t+1
        next_t = t + 1
        if next_t > diameter:
            break

        candidates = set()
        for (i, m) in newly_affected:
            for idx in range(indptr[m], indptr[m + 1]):
                j = indices[idx]
                if D[i, j] == next_t and not aff[i, j]:
                    candidates.add((i, j))

        for (i, j) in candidates:
            has_alt = False
            target = next_t - 1
            for idx in range(indptr[j], indptr[j + 1]):
                m = indices[idx]
                if D[i, m] == target and not aff[i, m]:
                    has_alt = True
                    break
            if not has_alt:
                aff[i, j] = True
                newly_affected_by_level[next_t].append((i, j))
                if next_t > max_level:
                    max_level = next_t

        levels_checked = max(levels_checked, next_t)

    affected_sources = list(np.where(np.any(aff, axis=1))[0])
    return affected_sources, levels_checked


def _py_identify_forward_directed(D, indptr, indices, a, b, n, diameter):
    """Forward propagation for directed edge a->b deletion."""
    aff = np.zeros((n, n), dtype=np.bool_)
    newly_affected_by_level = defaultdict(list)

    for i in range(n):
        if D[i, a] + 1 == D[i, b] and D[i, b] > 0:
            has_alt = False
            target = D[i, b] - 1
            for idx in range(indptr[b], indptr[b + 1]):
                m = indices[idx]
                if D[i, m] == target:
                    has_alt = True
                    break
            if not has_alt:
                lev = int(D[i, b])
                aff[i, b] = True
                newly_affected_by_level[lev].append((i, b))

    if not newly_affected_by_level:
        return [], 1

    min_level = min(newly_affected_by_level.keys())
    max_level = max(newly_affected_by_level.keys())
    levels_checked = max_level

    for t in range(min_level, diameter + 1):
        newly_affected = newly_affected_by_level.get(t, [])
        if not newly_affected:
            if t > max_level:
                break
            continue

        next_t = t + 1
        if next_t > diameter:
            break

        candidates = set()
        for (i, m) in newly_affected:
            for idx in range(indptr[m], indptr[m + 1]):
                j = indices[idx]
                if D[i, j] == next_t and not aff[i, j]:
                    candidates.add((i, j))

        for (i, j) in candidates:
            has_alt = False
            target = next_t - 1
            for idx in range(indptr[j], indptr[j + 1]):
                m = indices[idx]
                if D[i, m] == target and not aff[i, m]:
                    has_alt = True
                    break
            if not has_alt:
                aff[i, j] = True
                newly_affected_by_level[next_t].append((i, j))
                if next_t > max_level:
                    max_level = next_t

        levels_checked = max(levels_checked, next_t)

    affected_sources = list(np.where(np.any(aff, axis=1))[0])
    return affected_sources, levels_checked
