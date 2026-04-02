"""BC1: NetworkX BFS APSP — Python-level per-source BFS baseline."""

import numpy as np
import networkx as nx
import scipy.sparse as sp


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via NetworkX all_pairs_shortest_path_length (pure Python BFS)."""
    G = nx.from_scipy_sparse_array(A_csr)
    n = A_csr.shape[0]

    if verbose:
        print(f"  BC1 NetworkX BFS: n={n}")

    if k > 0:
        lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(G))

    D = np.zeros((n, n), dtype=np.int32)
    for src, targets in lengths.items():
        for tgt, dist in targets.items():
            D[src, tgt] = dist

    return D
