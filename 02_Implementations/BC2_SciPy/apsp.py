"""BC2: SciPy shortest_path — C-level per-source BFS baseline."""

import numpy as np
from scipy.sparse.csgraph import shortest_path


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via scipy.sparse.csgraph.shortest_path (C BFS, unweighted)."""
    n = A_csr.shape[0]
    if verbose:
        print(f"  BC2 SciPy: n={n}")

    D = shortest_path(A_csr, directed=False, unweighted=True)
    D = D.astype(np.int32)
    D[np.isinf(D.astype(np.float64))] = 0

    if k > 0:
        D[D > k] = 0

    return D
