"""BB2: igraph C-level Brandes Algorithm for Betweenness Centrality.

Thin wrapper around igraph's C implementation.
igraph internally uses optimised C code for the Brandes algorithm,
making it orders of magnitude faster than pure-Python equivalents.
"""

import numpy as np
import scipy.sparse as sp
import igraph as ig


def run_brandes(A_csr, verbose=True):
    """Compute betweenness centrality via igraph C-Brandes.

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
    if verbose:
        print(f"  BB2 Brandes-C (igraph): n={n}")

    # Build igraph Graph from upper-triangle edges (undirected)
    coo = A_csr.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)

    cb = np.array(G.betweenness(directed=False), dtype=np.float64)
    return cb
