"""
STORM Loader: Graph loading and preprocessing.

Modernized from AORM's loader.py:
  - Returns scipy.sparse matrices (not dense)
  - No deprecated APIs (nx.read_gpickle, np.bool)
  - Supports multiple formats: edge-list, gpickle, npz, mtx

Functions:
    load_graph        - Unified graph loading interface
    load_edgelist     - Load from edge-list text files
    load_gpickle      - Load from NetworkX gpickle files
    load_npz          - Load from scipy npz sparse files
    is_connected      - Check graph connectivity
    make_undirected   - Convert directed to undirected graph
"""

import sys
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def load_graph(filepath, fmt=None, directed=True):
    """Unified graph loading interface.

    Automatically detects format from file extension, or uses
    the fmt parameter to override.

    Args:
        filepath: Path to graph file.
        fmt: Format override ('edgelist', 'gpickle', 'npz', 'mtx').
            If None, auto-detected from extension.
        directed: If True, treat as directed graph.

    Returns:
        A: Adjacency matrix as scipy.sparse.csr_matrix.
        n: Number of nodes.
        m: Number of edges (nonzeros).
    """
    if fmt is None:
        fmt = _detect_format(filepath)

    if fmt == 'gpickle':
        A = load_gpickle(filepath)
    elif fmt in ('edgelist', 'txt', 'csv', 'tsv', 'edges'):
        A = load_edgelist(filepath)
    elif fmt == 'npz':
        A = load_npz(filepath)
    elif fmt == 'mtx':
        from scipy.io import mmread
        A = mmread(filepath).tocsr().astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Ensure CSR format
    A = sp.csr_matrix(A, dtype=np.float32)

    # Remove self-loops
    A = A - sp.diags(A.diagonal())
    A.eliminate_zeros()

    # Booleanize
    A.data[:] = 1.0

    if not directed:
        A = make_undirected(A)

    n = A.shape[0]
    m = A.nnz

    return A, n, m


def load_edgelist(filepath):
    """Load graph from edge-list text file.

    Supports formats: .txt, .csv, .tsv, .edges
    Lines starting with #, %, @ are treated as comments.

    Returns:
        A: Adjacency matrix as scipy.sparse.csr_matrix.
    """
    delimiter = _get_delimiter(filepath)

    edges = []
    max_id = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in '#%@':
                continue
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            max_id = max(max_id, u, v)

    n = max_id + 1
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    data = np.ones(len(edges), dtype=np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A


def load_gpickle(filepath):
    """Load graph from NetworkX gpickle file.

    Uses pickle directly (nx.read_gpickle is removed in NetworkX 3.x).

    Returns:
        A: Adjacency matrix as scipy.sparse.csr_matrix.
    """
    import networkx as nx

    with open(filepath, 'rb') as f:
        graph = pickle.load(f)

    A = nx.adjacency_matrix(graph).astype(np.float32)
    return A


def load_npz(filepath):
    """Load graph from scipy sparse npz file.

    Returns:
        A: Adjacency matrix as scipy.sparse.csr_matrix.
    """
    return sp.load_npz(filepath).astype(np.float32).tocsr()


def is_connected(A):
    """Check if graph is connected.

    Args:
        A: Adjacency matrix (sparse or dense).

    Returns:
        True if connected (single component), False otherwise.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    n_components, _ = connected_components(A, directed=False)
    return n_components == 1


def make_undirected(A):
    """Convert directed graph to undirected by symmetrizing.

    A_undirected = H(max(A, A^T))

    Args:
        A: Adjacency matrix (sparse).

    Returns:
        A_sym: Symmetric adjacency matrix.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A_sym = A + A.T
    A_sym = A_sym.tocsr()
    A_sym.data = np.minimum(A_sym.data, 1.0)
    return A_sym


def _detect_format(filepath):
    """Auto-detect file format from extension."""
    filepath = filepath.lower()
    if filepath.endswith('.gpickle'):
        return 'gpickle'
    elif filepath.endswith('.npz'):
        return 'npz'
    elif filepath.endswith('.mtx'):
        return 'mtx'
    elif filepath.endswith('.csv'):
        return 'csv'
    elif filepath.endswith('.tsv'):
        return 'tsv'
    elif filepath.endswith('.edges'):
        return 'edges'
    else:
        return 'edgelist'


def _get_delimiter(filepath):
    """Determine delimiter from file extension."""
    if filepath.endswith('.csv'):
        return ','
    elif filepath.endswith('.tsv'):
        return '\t'
    else:
        return None  # whitespace splitting
