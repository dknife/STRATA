/*
 * TB2: Matrix Brandes in pure C + OpenMP.
 *
 * Forward:  Fused SpMM + prune (parallel over result rows).
 * Backward: Per-source delta accumulation (parallel over sources).
 *
 * Compiled as a shared library (.dll / .so), called via ctypes.
 *
 * NOTE: MSVC OpenMP requires loop variable declared before
 *       #pragma omp parallel for, so we use C89-style declarations
 *       for all OpenMP loop indices.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int matrix_brandes(
    const int *A_indptr,
    const int *A_indices,
    int        n,
    int        nnz,
    double    *cb_out,
    int        verbose)
{
    long long nn = (long long)n * n;
    int i, e, s, w, k, pos, ae, re, cnt;
    int power, diam, Rk_nnz, total_new;
    long long base, idx;
    double v, coeff;

    /* ── Allocate ────────────────────────────────────────── */
    int      *D     = (int *)     calloc(nn, sizeof(int));
    double   *sigma = (double *)  calloc(nn, sizeof(double));
    uint8_t  *F     = (uint8_t *) calloc(nn, sizeof(uint8_t));
    double   *buf   = (double *)  calloc(nn, sizeof(double));
    double   *delta = (double *)  calloc(nn, sizeof(double));

    long long cap = nn;  /* safe upper bound for frontier size */

    int    *buf_ptr_A = (int *)    calloc(n + 1, sizeof(int));
    int    *buf_idx_A = (int *)    malloc(cap * sizeof(int));
    double *buf_val_A = (double *) malloc(cap * sizeof(double));
    int    *buf_ptr_B = (int *)    calloc(n + 1, sizeof(int));
    int    *buf_idx_B = (int *)    malloc(cap * sizeof(int));
    double *buf_val_B = (double *) malloc(cap * sizeof(double));
    int    *row_cnt = (int *)   malloc(n * sizeof(int));

    /* Current and next frontier pointers (swapped, not freed individually) */
    int    *Rk_ptr = buf_ptr_A;
    int    *Rk_idx = buf_idx_A;
    double *Rk_val = buf_val_A;
    int    *nf_ptr = buf_ptr_B;
    int    *nf_idx = buf_idx_B;
    double *nf_val = buf_val_B;

    if (!D || !sigma || !F || !buf || !delta ||
        !buf_ptr_A || !buf_idx_A || !buf_val_A ||
        !buf_ptr_B || !buf_idx_B || !buf_val_B || !row_cnt) {
        free(D); free(sigma); free(F); free(buf); free(delta);
        free(buf_ptr_A); free(buf_idx_A); free(buf_val_A);
        free(buf_ptr_B); free(buf_idx_B); free(buf_val_B); free(row_cnt);
        return -1;
    }

    /* ── Initialise diagonal ─────────────────────────────── */
    for (i = 0; i < n; i++) {
        F[(long long)i * n + i] = 1;
        sigma[(long long)i * n + i] = 1.0;
    }

    /* ── Hop 1: frontier = adjacency ─────────────────────── */
    memcpy(Rk_ptr, A_indptr, (n + 1) * sizeof(int));
    for (e = 0; e < nnz; e++) {
        Rk_idx[e] = A_indices[e];
        Rk_val[e] = 1.0;
    }
    for (i = 0; i < n; i++) {
        base = (long long)i * n;
        for (e = A_indptr[i]; e < A_indptr[i + 1]; e++) {
            k = A_indices[e];
            F[base + k]     = 1;
            D[base + k]     = 1;
            sigma[base + k] = 1.0;
        }
    }

    Rk_nnz = nnz;
    diam   = 1;
    power  = 1;

    /* ── Forward pass: fused SpMM + prune ────────────────── */
    while (Rk_nnz > 0) {
        power++;
        memset(buf, 0, nn * sizeof(double));

        /* Parallel SpMM + fused prune */
        #pragma omp parallel for schedule(dynamic, 64) private(ae, re, k, base)
        for (i = 0; i < n; i++) {
            base = (long long)i * n;
            for (ae = A_indptr[i]; ae < A_indptr[i + 1]; ae++) {
                int j = A_indices[ae];
                for (re = Rk_ptr[j]; re < Rk_ptr[j + 1]; re++) {
                    k = Rk_idx[re];
                    if (F[base + k] == 0)
                        buf[base + k] += Rk_val[re];
                }
            }
        }

        /* Count new entries per row */
        total_new = 0;
        #pragma omp parallel for reduction(+:total_new) private(cnt, base, k)
        for (i = 0; i < n; i++) {
            cnt = 0;
            base = (long long)i * n;
            for (k = 0; k < n; k++)
                if (buf[base + k] > 0.0) cnt++;
            row_cnt[i] = cnt;
            total_new += cnt;
        }

        if (total_new == 0) break;
        diam = power;

        if (verbose) {
            printf("    forward k=%d: nnz=%d\n", power, total_new);
            fflush(stdout);
        }

        /* Build new frontier CSR + update D / sigma / F */
        nf_ptr[0] = 0;
        for (i = 0; i < n; i++)
            nf_ptr[i + 1] = nf_ptr[i] + row_cnt[i];

        #pragma omp parallel for private(pos, base, k, v)
        for (i = 0; i < n; i++) {
            pos = nf_ptr[i];
            base = (long long)i * n;
            for (k = 0; k < n; k++) {
                v = buf[base + k];
                if (v > 0.0) {
                    F[base + k]     = 1;
                    D[base + k]     = power;
                    sigma[base + k] = v;
                    nf_idx[pos] = k;
                    nf_val[pos] = v;
                    pos++;
                }
            }
        }

        /* Swap frontiers */
        { int    *t = Rk_ptr; Rk_ptr = nf_ptr; nf_ptr = t; }
        { int    *t = Rk_idx; Rk_idx = nf_idx; nf_idx = t; }
        { double *t = Rk_val; Rk_val = nf_val; nf_val = t; }
        Rk_nnz = total_new;
    }

    if (verbose) {
        printf("    diameter=%d\n", diam);
        fflush(stdout);
    }

    /* ── Backward pass: per-source delta accumulation ────── */
    #pragma omp parallel for schedule(dynamic, 16) private(k, w, e, coeff, base)
    for (s = 0; s < n; s++) {
        base = (long long)s * n;
        for (k = diam; k >= 2; k--) {
            for (w = 0; w < n; w++) {
                if (D[base + w] != k) continue;
                coeff = (1.0 + delta[base + w]) / sigma[base + w];
                for (e = A_indptr[w]; e < A_indptr[w + 1]; e++) {
                    int vv = A_indices[e];
                    if (D[base + vv] == k - 1)
                        delta[base + vv] += sigma[base + vv] * coeff;
                }
            }
        }
    }

    /* ── Betweenness: sum columns / 2 ────────────────────── */
    memset(cb_out, 0, n * sizeof(double));
    for (s = 0; s < n; s++) {
        base = (long long)s * n;
        for (i = 0; i < n; i++)
            cb_out[i] += delta[base + i];
    }
    for (i = 0; i < n; i++)
        cb_out[i] /= 2.0;

    /* ── Free ────────────────────────────────────────────── */
    free(D); free(sigma); free(F); free(buf); free(delta);
    free(buf_ptr_A); free(buf_idx_A); free(buf_val_A);
    free(buf_ptr_B); free(buf_idx_B); free(buf_val_B); free(row_cnt);
    return 0;
}
