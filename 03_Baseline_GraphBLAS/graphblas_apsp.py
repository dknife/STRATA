"""
GraphBLAS-based All-Pairs Shortest Paths (APSP) for unweighted graphs.

Implements BFS-based APSP using the SuiteSparse:GraphBLAS C library
via its CFFI bindings. This serves as a strong baseline against D-STORM,
representing the state-of-the-art in sparse linear algebra graph processing.

Three APSP strategies are provided:
  1. graphblas_bfs_apsp       — source-wise masked BFS via SpMV (standard)
  2. graphblas_level_apsp     — level-synchronous BFS via SpMM (batch)
  3. graphblas_frontier_apsp  — frontier-masking SpMM closest to D-STORM logic
"""

import numpy as np
import scipy.sparse as sp
from suitesparse_graphblas import ffi, lib

# ---------------------------------------------------------------------------
# GraphBLAS lifecycle helpers
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        rc = lib.GrB_init(lib.GrB_NONBLOCKING)
        if rc not in (0, 6):  # GrB_SUCCESS or GrB_INVALID_OBJECT (already init)
            raise RuntimeError(f"GrB_init failed with rc={rc}")
        _initialized = True


def _check(rc, msg="GraphBLAS"):
    if rc != 0:
        raise RuntimeError(f"{msg} failed with GrB_Info={rc}")


# ---------------------------------------------------------------------------
# Conversion helpers: scipy.sparse <-> GrB_Matrix
# ---------------------------------------------------------------------------

def scipy_to_grb_bool(A_csr):
    """Convert a scipy CSR matrix to a GrB_Matrix (BOOL). Self-loops removed."""
    _ensure_init()
    A_csr = A_csr.copy()
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()
    A_csr = A_csr.tocoo()
    n = A_csr.shape[0]
    rows = A_csr.row.astype(np.uint64)
    cols = A_csr.col.astype(np.uint64)
    nnz = len(rows)

    M = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(M, lib.GrB_BOOL, n, n), "Matrix_new")

    # Build from COO arrays
    rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows))
    cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols))
    vals = ffi.new('bool[]', [True] * nnz)

    _check(lib.GrB_Matrix_build_BOOL(M[0], rows_c, cols_c, vals, nnz,
                                      lib.GrB_LOR), "Matrix_build")
    return M


def scipy_to_grb_int32(A_csr):
    """Convert a scipy CSR matrix to a GrB_Matrix (INT32), values = 1. Self-loops removed."""
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
    _check(lib.GrB_Matrix_new(M, lib.GrB_INT32, n, n), "Matrix_new")

    rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows))
    cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols))
    vals = np.ones(nnz, dtype=np.int32)
    vals_c = ffi.cast('int32_t*', ffi.from_buffer(vals))

    _check(lib.GrB_Matrix_build_INT32(M[0], rows_c, cols_c, vals_c, nnz,
                                       lib.GrB_PLUS_INT32), "Matrix_build")
    return M


def grb_matrix_to_dense(M, n, dtype=np.int32):
    """Extract a GrB_Matrix to a dense numpy array."""
    result = np.zeros((n, n), dtype=dtype)
    val = ffi.new('int32_t*')
    for i in range(n):
        for j in range(n):
            rc = lib.GrB_Matrix_extractElement_INT32(val, M, i, j)
            if rc == 0:  # GrB_SUCCESS
                result[i, j] = val[0]
    return result


def grb_matrix_nvals(M):
    """Get number of stored values in a GrB_Matrix."""
    nvals = ffi.new('GrB_Index*')
    _check(lib.GrB_Matrix_nvals(nvals, M), "nvals")
    return nvals[0]


def grb_vector_nvals(v):
    """Get number of stored values in a GrB_Vector."""
    nvals = ffi.new('GrB_Index*')
    _check(lib.GrB_Vector_nvals(nvals, v), "nvals")
    return nvals[0]


# ---------------------------------------------------------------------------
# Method 1: Source-wise masked BFS via SpMV
#
# This is the standard GraphBLAS BFS pattern from the LAGraph reference.
# For each source vertex s, we run a level-synchronous BFS using
# vxm (vector-matrix multiply) with structural complement masking.
# ---------------------------------------------------------------------------

def _bfs_single_source(A_grb, n, src):
    """
    Single-source BFS using GraphBLAS masked vxm.

    Returns a GrB_Vector of distances from src.
    Uses MIN_FIRST semiring on INT32 with complement masking.
    """
    # Distance vector d: d[src] = 0
    d = ffi.new('GrB_Vector*')
    _check(lib.GrB_Vector_new(d, lib.GrB_INT32, n))
    _check(lib.GrB_Vector_setElement_INT32(d[0], 0, src))

    # Frontier vector q: q[src] = 1
    q = ffi.new('GrB_Vector*')
    _check(lib.GrB_Vector_new(q, lib.GrB_INT32, n))
    _check(lib.GrB_Vector_setElement_INT32(q[0], 1, src))

    # Descriptor: complement mask + replace
    desc = ffi.new('GrB_Descriptor*')
    _check(lib.GrB_Descriptor_new(desc))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_MASK, lib.GrB_COMP))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_OUTP, lib.GrB_REPLACE))

    level = 1
    while True:
        # q<!d> = q * A  (vxm with complement mask d, replace output)
        # Only discover vertices NOT already in d
        _check(lib.GrB_vxm(q[0], d[0], ffi.NULL,
                            lib.GrB_MIN_FIRST_SEMIRING_INT32,
                            q[0], A_grb, desc[0]), "vxm")

        nq = grb_vector_nvals(q[0])
        if nq == 0:
            break

        # d<q> = level  (assign level to all newly discovered vertices)
        _check(lib.GrB_Vector_assign_INT32(d[0], q[0], ffi.NULL,
                                            level, lib.GrB_ALL, n, ffi.NULL),
               "assign")
        level += 1

    lib.GrB_Descriptor_free(desc)
    lib.GrB_Vector_free(q)
    return d, level - 1


def graphblas_bfs_apsp(A_csr, k=-1, verbose=True):
    """
    APSP via per-source masked BFS (GraphBLAS vxm).

    This is the standard LAGraph-style BFS-APSP baseline.

    Parameters
    ----------
    A_csr : scipy.sparse.csr_matrix
        Boolean adjacency matrix (unweighted).
    k : int
        Hop constraint (-1 for full APSP).
    verbose : bool
        Print progress.

    Returns
    -------
    D : np.ndarray (n x n, int32)
        Distance matrix. 0 = self or unreachable.
    """
    _ensure_init()
    n = A_csr.shape[0]

    # Build GrB adjacency (INT32 with all values = 1)
    A_grb = scipy_to_grb_int32(A_csr)

    D = np.zeros((n, n), dtype=np.int32)
    val = ffi.new('int32_t*')

    iterator = range(n)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="GraphBLAS BFS-APSP", unit="src")

    for src in iterator:
        d, max_level = _bfs_single_source(A_grb[0], n, src)

        # Extract distances for this source
        for j in range(n):
            rc = lib.GrB_Vector_extractElement_INT32(val, d[0], j)
            if rc == 0:
                if k < 0 or val[0] <= k:
                    D[src, j] = val[0]

        lib.GrB_Vector_free(d)

    np.fill_diagonal(D, 0)
    lib.GrB_Matrix_free(A_grb)
    return D


# ---------------------------------------------------------------------------
# Method 2: Level-synchronous BFS via SpMM (batch)
#
# Instead of per-source BFS, we do a batch approach:
# expand all frontiers simultaneously using Boolean SpMM.
# This is closer to D-STORM's approach but uses GraphBLAS kernels.
# ---------------------------------------------------------------------------

def graphblas_level_apsp(A_csr, k=-1, verbose=True):
    """
    APSP via level-synchronous Boolean SpMM with complement masking.

    At each hop level, we compute: Frontier = (Frontier * A) AND NOT Visited
    This mirrors D-STORM's cumulative footprint pruning but uses
    GraphBLAS mxm with structural complement masking.

    Parameters
    ----------
    A_csr : scipy.sparse.csr_matrix
        Boolean adjacency matrix.
    k : int
        Hop constraint (-1 for full).
    verbose : bool
        Print progress.

    Returns
    -------
    D : np.ndarray (n x n, int32)
        Distance matrix.
    """
    _ensure_init()
    n = A_csr.shape[0]

    # Build GrB adjacency (BOOL)
    A_grb = scipy_to_grb_bool(A_csr)

    # Frontier F = A (1-hop shell)
    F = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(F, lib.GrB_BOOL, n, n))
    _check(lib.GrB_Matrix_assign(F[0], A_grb[0], ffi.NULL, A_grb[0],
                                  lib.GrB_ALL, n, lib.GrB_ALL, n, ffi.NULL),
           "assign F=A")

    # Visited V = I + A (identity + 1-hop)
    V = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(V, lib.GrB_BOOL, n, n))
    # Set identity
    for i in range(n):
        _check(lib.GrB_Matrix_setElement_BOOL(V[0], True, i, i))
    # V = V | A
    _check(lib.GrB_Matrix_assign(V[0], ffi.NULL, lib.GrB_LOR,
                                  A_grb[0], lib.GrB_ALL, n,
                                  lib.GrB_ALL, n, ffi.NULL), "V|=A")

    # Distance matrix (dense numpy, accumulated on CPU)
    D = np.zeros((n, n), dtype=np.int32)

    # Record 1-hop distances
    A_coo = A_csr.tocoo()
    for i, j in zip(A_coo.row, A_coo.col):
        D[i, j] = 1

    # Descriptor for complement mask + replace
    desc = ffi.new('GrB_Descriptor*')
    _check(lib.GrB_Descriptor_new(desc))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_MASK, lib.GrB_COMP))
    _check(lib.GrB_Descriptor_set(desc[0], lib.GrB_OUTP, lib.GrB_REPLACE))

    # Temp matrix for candidates
    T = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(T, lib.GrB_BOOL, n, n))

    level = 2
    max_level = n if k < 0 else k

    while level <= max_level:
        # T<!V> = F * A  (candidates masked by complement of visited)
        _check(lib.GrB_mxm(T[0], V[0], ffi.NULL,
                            lib.GrB_LOR_LAND_SEMIRING_BOOL,
                            F[0], A_grb[0], desc[0]), "mxm")

        nnz = grb_matrix_nvals(T[0])
        if verbose:
            print(f"  hop {level}: nnz(frontier) = {nnz}")
        if nnz == 0:
            break

        # Extract new pairs and record distances
        rows_buf = np.empty(nnz, dtype=np.uint64)
        cols_buf = np.empty(nnz, dtype=np.uint64)
        vals_buf = np.empty(nnz, dtype=np.bool_)
        nvals_p = ffi.new('GrB_Index*', nnz)

        rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows_buf))
        cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols_buf))
        vals_c = ffi.cast('bool*', ffi.from_buffer(vals_buf))

        _check(lib.GrB_Matrix_extractTuples_BOOL(
            rows_c, cols_c, vals_c, nvals_p, T[0]), "extractTuples")

        actual_nnz = nvals_p[0]
        for idx in range(actual_nnz):
            D[rows_buf[idx], cols_buf[idx]] = level

        # Update visited: V = V | T
        _check(lib.GrB_Matrix_assign(V[0], ffi.NULL, lib.GrB_LOR,
                                      T[0], lib.GrB_ALL, n,
                                      lib.GrB_ALL, n, ffi.NULL), "V|=T")

        # Swap: F = T for next iteration
        _check(lib.GrB_Matrix_assign(F[0], T[0], ffi.NULL, T[0],
                                      lib.GrB_ALL, n, lib.GrB_ALL, n,
                                      ffi.NULL), "F=T")
        # Clear F entries not in T
        _check(lib.GrB_Matrix_assign(F[0], ffi.NULL, ffi.NULL, T[0],
                                      lib.GrB_ALL, n, lib.GrB_ALL, n,
                                      ffi.NULL), "F=T")

        level += 1

    # Cleanup
    lib.GrB_Matrix_free(A_grb)
    lib.GrB_Matrix_free(F)
    lib.GrB_Matrix_free(V)
    lib.GrB_Matrix_free(T)
    lib.GrB_Descriptor_free(desc)

    return D


# ---------------------------------------------------------------------------
# Method 3: Frontier-masking SpMM (closest to D-STORM logic)
#
# Reimplements D-STORM's exact algorithm using GraphBLAS primitives.
# This is the fairest comparison: same algorithm, GraphBLAS kernels.
# ---------------------------------------------------------------------------

def graphblas_frontier_apsp(A_csr, k=-1, verbose=True):
    """
    D-STORM algorithm reimplemented with GraphBLAS primitives.

    Uses the exact same cumulative-footprint frontier algebra as D-STORM,
    but delegates SpMM and element-wise masking to SuiteSparse:GraphBLAS
    C kernels instead of scipy/Python.

    Parameters
    ----------
    A_csr : scipy.sparse.csr_matrix
        Boolean adjacency matrix.
    k : int
        Hop constraint (-1 for full).
    verbose : bool
        Print progress.

    Returns
    -------
    D : np.ndarray (n x n, int32)
        Distance matrix.
    """
    _ensure_init()
    n = A_csr.shape[0]

    A_grb = scipy_to_grb_bool(A_csr)

    # R = A (current frontier = 1-hop shell)
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

    # Temp candidate matrix
    T = ffi.new('GrB_Matrix*')
    _check(lib.GrB_Matrix_new(T, lib.GrB_BOOL, n, n))

    # Descriptor: complement mask + replace
    desc_cr = ffi.new('GrB_Descriptor*')
    _check(lib.GrB_Descriptor_new(desc_cr))
    _check(lib.GrB_Descriptor_set(desc_cr[0], lib.GrB_MASK, lib.GrB_COMP))
    _check(lib.GrB_Descriptor_set(desc_cr[0], lib.GrB_OUTP, lib.GrB_REPLACE))

    level = 2
    max_level = n if k < 0 else k

    while level <= max_level:
        # T<!F> = R * A  (SpMM with complement-mask by footprint)
        _check(lib.GrB_mxm(T[0], F[0], ffi.NULL,
                            lib.GrB_LOR_LAND_SEMIRING_BOOL,
                            R[0], A_grb[0], desc_cr[0]), "mxm")

        nnz = grb_matrix_nvals(T[0])
        if verbose:
            print(f"  hop {level}: nnz(R*) = {nnz}")
        if nnz == 0:
            break

        # Extract new shell and record distances
        rows_buf = np.empty(nnz, dtype=np.uint64)
        cols_buf = np.empty(nnz, dtype=np.uint64)
        vals_buf = np.empty(nnz, dtype=np.bool_)
        nvals_p = ffi.new('GrB_Index*', nnz)

        rows_c = ffi.cast('GrB_Index*', ffi.from_buffer(rows_buf))
        cols_c = ffi.cast('GrB_Index*', ffi.from_buffer(cols_buf))
        vals_c = ffi.cast('bool*', ffi.from_buffer(vals_buf))

        _check(lib.GrB_Matrix_extractTuples_BOOL(
            rows_c, cols_c, vals_c, nvals_p, T[0]), "extractTuples")

        for idx in range(nvals_p[0]):
            D[rows_buf[idx], cols_buf[idx]] = level

        # Update footprint: F = F | T
        _check(lib.GrB_Matrix_assign(F[0], ffi.NULL, lib.GrB_LOR,
                                      T[0], lib.GrB_ALL, n,
                                      lib.GrB_ALL, n, ffi.NULL), "F|=T")

        # R = T (next frontier)
        _check(lib.GrB_Matrix_assign(R[0], ffi.NULL, ffi.NULL, T[0],
                                      lib.GrB_ALL, n, lib.GrB_ALL, n,
                                      ffi.NULL), "R=T")

        level += 1

    # Cleanup
    lib.GrB_Matrix_free(A_grb)
    lib.GrB_Matrix_free(R)
    lib.GrB_Matrix_free(F)
    lib.GrB_Matrix_free(T)
    lib.GrB_Descriptor_free(desc_cr)

    return D


# ---------------------------------------------------------------------------
# Convenience: select method by name
# ---------------------------------------------------------------------------

METHODS = {
    'bfs': graphblas_bfs_apsp,
    'level': graphblas_level_apsp,
    'frontier': graphblas_frontier_apsp,
}


def graphblas_apsp(A_csr, method='frontier', k=-1, verbose=True):
    """Run GraphBLAS APSP with the specified method."""
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")
    return fn(A_csr, k=k, verbose=verbose)
