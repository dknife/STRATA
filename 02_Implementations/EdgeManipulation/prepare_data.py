"""
Prepare Facebook graph data for edge manipulation experiments.

Loads the Facebook dataset, computes APSP via SciPy (BC2),
and saves adjacency matrix (A) and distance matrix (D) to files.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.prepare_data
"""

import sys, os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.loader import load_graph

FACEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '04_Datasets',
    'real-world', 'facebook_combined.txt'
)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading Facebook graph...")
    A, n, m = load_graph(FACEBOOK_PATH, fmt='edgelist', directed=False)
    print(f"  n={n}, m={m} (undirected edges={m//2})")

    print("Computing APSP via SciPy shortest_path...")
    D_float = shortest_path(A, directed=False, unweighted=True)
    D = D_float.astype(np.int32)
    D[np.isinf(D_float)] = 0  # disconnected -> 0
    print(f"  D shape={D.shape}, dtype={D.dtype}")
    print(f"  diameter={D.max()}, mean distance={D[D>0].mean():.2f}")

    # Save adjacency (sparse CSR -> npz)
    a_path = os.path.join(DATA_DIR, 'facebook_A.npz')
    sp.save_npz(a_path, A.tocsr())
    print(f"  Saved A -> {a_path}")

    # Save distance matrix (dense int32 -> npy)
    d_path = os.path.join(DATA_DIR, 'facebook_D.npy')
    np.save(d_path, D)
    print(f"  Saved D -> {d_path}")

    print("Done.")


if __name__ == '__main__':
    main()
