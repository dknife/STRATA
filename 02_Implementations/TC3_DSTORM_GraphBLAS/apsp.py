"""TC3: GB-frontier — D-STORM algorithm via GraphBLAS primitives.

Reimplements D-STORM's cumulative-footprint frontier algebra using
SuiteSparse:GraphBLAS C kernels (GrB_mxm with LOR_LAND semiring +
structural complement mask).
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


def _scipy_to_grb_bool(A_csr):
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
    _check(lib.GrB_Matrix_new(M, lib.GrB_BOOL, n, n))
    rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows))
    cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols))
    vals = ffi.new('bool[]', [True] * nnz)
    _check(lib.GrB_Matrix_build_BOOL(M[0], rows_c, cols_c, vals, nnz,
                                      lib.GrB_LOR))
    return M


def _grb_nvals(M):
    nvals = ffi.new('GrB_Index*')
    _check(lib.GrB_Matrix_nvals(nvals, M))
    return nvals[0]


def run_apsp(A_csr, k=-1, verbose=True):
    """APSP via D-STORM algorithm on GraphBLAS (GrB_mxm + complement mask)."""
    _ensure_init()
    n = A_csr.shape[0]

    if verbose:
        print(f"  TC3 GB-frontier: n={n}")

    A_grb = _scipy_to_grb_bool(A_csr)

    # R = A (1-hop frontier)
    R = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(R, lib.GrB_BOOL, n, n))
    _check(lib.GrB_Matrix_assign(R[0], ffi.NULL, ffi.NULL, A_grb[0],
                                  lib.GrB_ALL, n, lib.GrB_ALL, n, ffi.NULL))

    # F = I + A (cumulative footprint)
    F = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(F, lib.GrB_BOOL, n, n))
    for i in range(n):
        _check(lib.GrB_Matrix_setElement_BOOL(F[0], True, i, i))
    _check(lib.GrB_Matrix_assign(F[0], ffi.NULL, lib.GrB_LOR, A_grb[0],
                                  lib.GrB_ALL, n, lib.GrB_ALL, n, ffi.NULL))

    # Distance matrix
    D = np.zeros((n, n), dtype=np.int32)
    A_coo = A_csr.tocoo()
    for i, j in zip(A_coo.row, A_coo.col):
        D[i, j] = 1

    # Temp + descriptor
    T = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(T, lib.GrB_BOOL, n, n))
    desc = ffi.new('GrB_Descriptor*')
    _check(lib.GrB_Descriptor_new(desc))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_MASK, lib.GrB_COMP))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_OUTP, lib.GrB_REPLACE))

    level = 2
    max_level = n if k < 0 else k

    while level <= max_level:
        # T<!F> = R * A
        _check(lib.GrB_mxm(T[0], F[0], ffi.NULL,
                            lib.GrB_LOR_LAND_SEMIRING_BOOL,
                            R[0], A_grb[0], desc[0]))

        nnz = _grb_nvals(T[0])
        if verbose:
            print(f"    hop {level}: nnz={nnz}")
        if nnz == 0:
            break

        # Extract and record distances
        rows_buf = np.empty(nnz, dtype=np.uint64)
        cols_buf = np.empty(nnz, dtype=np.uint64)
        vals_buf = np.empty(nnz, dtype=np.bool_)
        nvals_p = ffi.new('GrB_Index*', nnz)
        _check(lib.GrB_Matrix_extractTuples_BOOL(
            ffi.cast('GrB_Index*', ffi.from_buffer(rows_buf)),
            ffi.cast('GrB_Index*', ffi.from_buffer(cols_buf)),
            ffi.cast('bool*', ffi.from_buffer(vals_buf)),
            nvals_p, T[0]))

        for idx in range(nvals_p[0]):
            D[rows_buf[idx], cols_buf[idx]] = level

        # F = F | T
        _check(lib.GrB_Matrix_assign(F[0], ffi.NULL, lib.GrB_LOR,
                                      T[0], lib.GrB_ALL, n,
                                      lib.GrB_ALL, n, ffi.NULL))
        # R = T
        _check(lib.GrB_Matrix_assign(R[0], ffi.NULL, ffi.NULL, T[0],
                                      lib.GrB_ALL, n, lib.GrB_ALL, n,
                                      ffi.NULL))
        level += 1

    lib.GrB_Matrix_free(A_grb)
    lib.GrB_Matrix_free(R)
    lib.GrB_Matrix_free(F)
    lib.GrB_Matrix_free(T)
    lib.GrB_Descriptor_free(desc)

    return D
