"""BC5: GraphBLAS per-source BFS APSP (GrB_vxm).

Standard LAGraph-style BFS: for each source vertex, runs a
level-synchronous masked BFS using GraphBLAS vector-matrix multiply.
"""

import numpy as np
import scipy.sparse as sp
from suitesparse_graphblas import ffi, lib


_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        rc = lib.GrB_init(lib.GrB_NONBLOCKING)
        if rc not in (0, 6):
            raise RuntimeError(f"GrB_init failed with rc={rc}")
        _initialized = True


def _check(rc, msg="GraphBLAS"):
    if rc != 0:
        raise RuntimeError(f"{msg} failed with GrB_Info={rc}")


def _scipy_to_grb_int32(A_csr):
    _ensure_init()
    A_csr = A_csr.copy()
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()
    A_coo = A_csr.tocoo()
    n = A_coo.shape[0]
    rows = A_coo.row.astype(np.uint64)
    cols = A_coo.col.astype(np.uint64)
    nnz = len(rows)

    M = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(M, lib.GrB_INT32, n, n))
    rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows))
    cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols))
    vals = np.ones(nnz, dtype=np.int32)
    vals_c = ffi.cast('int32_t*', ffi.from_buffer(vals))
    _check(lib.GrB_Matrix_build_INT32(M[0], rows_c, cols_c, vals_c, nnz,
                                       lib.GrB_PLUS_INT32))
    return M


def _bfs_single_source(A_grb, n, src):
    d = ffi.new('GrB_Vector*')
    _check(lib.GrB_Vector_new(d, lib.GrB_INT32, n))
    _check(lib.GrB_Vector_setElement_INT32(d[0], 0, src))

    q = ffi.new('GrB_Vector*')
    _check(lib.GrB_Vector_new(q, lib.GrB_INT32, n))
    _check(lib.GrB_Vector_setElement_INT32(q[0], 1, src))

    desc = ffi.new('GrB_Descriptor*')
    _check(lib.GrB_Descriptor_new(desc))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_MASK, lib.GrB_COMP))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_OUTP, lib.GrB_REPLACE))

    level = 1
    nvals = ffi.new('GrB_Index*')
    while True:
        _check(lib.GrB_vxm(q[0], d[0], ffi.NULL,
                            lib.GrB_MIN_FIRST_SEMIRING_INT32,
                            q[0], A_grb, desc[0]))
        _check(lib.GrB_Vector_nvals(nvals, q[0]))
        if nvals[0] == 0:
            break
        _check(lib.GrB_Vector_assign_INT32(d[0], q[0], ffi.NULL,
                                            level, lib.GrB_ALL, n, ffi.NULL))
        level += 1

    lib.GrB_Descriptor_free(desc)
    lib.GrB_Vector_free(q)
    return d


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via GraphBLAS per-source masked BFS (GrB_vxm)."""
    _ensure_init()
    n = A_csr.shape[0]

    if verbose:
        print(f"  BC5 GB-bfs: n={n}")

    A_grb = _scipy_to_grb_int32(A_csr)
    D = np.zeros((n, n), dtype=np.int32)
    val = ffi.new('int32_t*')

    iterator = range(n)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="GB-bfs APSP", unit="src")

    for src in iterator:
        d = _bfs_single_source(A_grb[0], n, src)
        for j in range(n):
            rc = lib.GrB_Vector_extractElement_INT32(val, d[0], j)
            if rc == 0:
                if k < 0 or val[0] <= k:
                    D[src, j] = val[0]
        lib.GrB_Vector_free(d)

    np.fill_diagonal(D, 0)
    lib.GrB_Matrix_free(A_grb)
    return D
