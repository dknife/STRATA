"""
Shared graph loading utilities for Mac APSP implementations.
"""

import numpy as np
import scipy.sparse as sp


def load_graph(filepath, fmt=None, directed=False):
    """Load a graph and return (A_csr, n, m)."""
    if fmt is None:
        if filepath.endswith('.npz'):
            fmt = 'npz'
        else:
            fmt = 'edgelist'

    if fmt == 'npz':
        A = sp.load_npz(filepath).tocsr()
    else:
        A = _load_edgelist(filepath)

    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0

    if not directed:
        A = A + A.T
        A.data[:] = 1.0
        A.eliminate_zeros()

    A = A.tocsr()
    return A, A.shape[0], A.nnz


def _load_edgelist(filepath):
    edges, max_node = [], 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node = max(max_node, u, v)
    n = max_node + 1
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    return sp.csr_matrix(([1.0] * len(edges), (rows, cols)), shape=(n, n))


def make_undirected(A):
    A = A + A.T
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A.tocsr()


def is_connected(A):
    from scipy.sparse.csgraph import connected_components
    return connected_components(A, directed=False)[0] == 1
