"""Graph loading utilities (shared with CPU STORM)."""

import numpy as np
import scipy.sparse as sp


def load_graph(filepath, fmt=None, directed=False):
    """Load graph, return (A_csr, n, m)."""
    if fmt is None:
        if filepath.endswith('.gpickle'):
            fmt = 'gpickle'
        elif filepath.endswith('.npz'):
            fmt = 'npz'
        else:
            fmt = 'edgelist'

    if fmt == 'gpickle':
        import networkx as nx
        G = nx.read_gpickle(filepath)
        A = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    elif fmt == 'npz':
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
    edges = []
    max_node = 0
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
    n_components, _ = connected_components(A, directed=False)
    return n_components == 1
