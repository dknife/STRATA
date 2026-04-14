"""TB2: Matrix Brandes — Pure C + OpenMP, Python ctypes wrapper.

The entire algorithm (forward SpMM + backward delta) runs in compiled C
with OpenMP parallelism.  Zero Python overhead in the hot path.
"""

import ctypes
import numpy as np
import scipy.sparse as sp
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_DLL_PATH = os.path.join(_DIR, 'brandes_core.dll')
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib

    # Auto-build if DLL missing
    if not os.path.exists(_DLL_PATH):
        from . import build as _build
        if _build.build(verbose=True) is None:
            raise RuntimeError("Cannot build brandes_core.dll")

    _lib = ctypes.CDLL(_DLL_PATH)
    _lib.matrix_brandes.argtypes = [
        ctypes.POINTER(ctypes.c_int),     # A_indptr
        ctypes.POINTER(ctypes.c_int),     # A_indices
        ctypes.c_int,                     # n
        ctypes.c_int,                     # nnz
        ctypes.POINTER(ctypes.c_double),  # cb_out
        ctypes.c_int,                     # verbose
    ]
    _lib.matrix_brandes.restype = ctypes.c_int
    return _lib


def run_brandes(A_csr, verbose=True):
    """Compute betweenness centrality via C Matrix Brandes.

    Args:
        A_csr: Adjacency matrix (scipy sparse or dense).
        verbose: Print progress (forwarded to C code).

    Returns:
        cb: Betweenness centrality vector (n,), unnormalized, undirected.
    """
    if not sp.issparse(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    A_csr = A_csr.astype(np.float64)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()
    A_csr = A_csr.tocsr()

    n = A_csr.shape[0]
    if verbose:
        print(f"  TB2 Matrix-Brandes-C (D-STORM): n={n}")

    indptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    nnz = len(indices)
    cb = np.zeros(n, dtype=np.float64)

    lib = _load()
    rc = lib.matrix_brandes(
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n),
        ctypes.c_int(nnz),
        cb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1 if verbose else 0),
    )

    if rc != 0:
        raise RuntimeError(f"matrix_brandes returned error code {rc}")

    return cb
