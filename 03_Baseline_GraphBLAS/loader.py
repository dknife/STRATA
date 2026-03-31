"""
Graph loading utilities for the GraphBLAS baseline.

Compatible with D-STORM's loader interface.
"""

import numpy as np
import scipy.sparse as sp


def load_graph(filepath, fmt=None, directed=False):
    """
    Load a graph and return (A_csr, n, m).

    Parameters
    ----------
    filepath : str
        Path to graph file.
    fmt : str or None
        Format override: 'edgelist', 'gpickle', 'npz'.
        If None, auto-detected from extension.
    directed : bool
        If True, keep directed edges. Otherwise symmetrize.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Boolean adjacency matrix.
    n : int
        Number of vertices.
    m : int
        Number of (directed) edges.
    """
    if fmt is None:
        if filepath.endswith('.gpickle'):
            fmt = 'gpickle'
        elif filepath.endswith('.npz'):
            fmt = 'npz'
        else:
            fmt = 'edgelist'

    if fmt == 'gpickle':
        A = _load_gpickle(filepath)
    elif fmt == 'npz':
        A = sp.load_npz(filepath).tocsr()
    else:
        A = _load_edgelist(filepath)

    # Remove self-loops
    A.setdiag(0)
    A.eliminate_zeros()

    # Booleanize
    A.data[:] = 1.0

    if not directed:
        A = A + A.T
        A.data[:] = 1.0
        A.eliminate_zeros()

    A = A.tocsr()
    n = A.shape[0]
    m = A.nnz

    return A, n, m


def _load_edgelist(filepath):
    """Load edge-list format (whitespace-separated pairs per line)."""
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
    data = [1.0] * len(edges)

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _load_gpickle(filepath):
    """Load NetworkX gpickle format."""
    import networkx as nx
    G = nx.read_gpickle(filepath)
    return nx.to_scipy_sparse_array(G, format='csr').astype(float)


def make_undirected(A):
    """Symmetrize adjacency matrix."""
    A = A + A.T
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A.tocsr()


def is_connected(A):
    """Check if the graph is connected."""
    from scipy.sparse.csgraph import connected_components
    n_components, _ = connected_components(A, directed=False)
    return n_components == 1
