"""BB1: Pure Python Brandes Algorithm for Betweenness Centrality.

Standard Brandes (2001): per-source BFS forward pass (sigma accumulation)
+ backward delta accumulation.  n Python loops, O(nm) total.
"""

import numpy as np
import scipy.sparse as sp
from collections import deque


def run_brandes(A_csr, verbose=True):
    """Compute betweenness centrality via standard Brandes algorithm.

    Args:
        A_csr: Adjacency matrix (scipy sparse or dense).
        verbose: Print progress.

    Returns:
        cb: Betweenness centrality vector (n,), unnormalized, undirected.
    """
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float32)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()

    n = A_csr.shape[0]
    indptr = A_csr.indptr
    indices = A_csr.indices

    if verbose:
        print(f"  BB1 Brandes-Python: n={n}")

    cb = np.zeros(n, dtype=np.float64)

    for s in range(n):
        # ── Forward BFS ──
        S = []                              # stack (BFS order)
        P = [[] for _ in range(n)]          # predecessors
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        Q = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            for idx in range(indptr[v], indptr[v + 1]):
                w = indices[idx]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # ── Backward accumulation ──
        delta = np.zeros(n, dtype=np.float64)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # Undirected normalization
    cb /= 2.0
    return cb
