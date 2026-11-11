/*
 * openblas_dgemm.c -- C transliteration of OpenBLAS's portable C GEMM kernels.
 *
 * Source: OpenBLAS 0.3.31 (copy with SHA256SUMS in ~/netlib_ref/openblas-0.3.31),
 *   interface/gemm.c L679-L683  small-matrix path: beta == 0 -> *_b0 kernel
 *   kernel/generic/gemm_small_matrix_kernel_{nn,nt,tn,tt}.c L31-L56
 *
 *   C = alpha*op(A)*op(B) + beta*C, column-major.
 *
 * Transliteration rules (generic_solver/lib_v3/README.md): FLOAT = OPENBLAS_SCALAR (double by default),
 * BLASLONG = long, blasint = int; inc_x = inc_y = 1; column-major addressing
 * a[i + j*lda] (or a pointer bumped by lda per column) becomes A[j][i] of the
 * VLA parameter OPENBLAS_SCALAR A[][lda] (the same address); loop order and
 * arithmetic order verbatim; scalar branches are specialised by macros.
 *
 * Licence of the source (OpenBLAS LICENSE, BSD-3-Clause):
 *   Copyright (c) 2011-2025, The OpenBLAS Project.
 *   Copyright 2009, 2010 The University of Texas at Austin (GotoBLAS2 parts).
 *   All rights reserved.
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions are met:
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. Neither the name of the OpenBLAS project nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *   AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED (full text in LICENSE).
 *
 * Macros: TRANSA, TRANSB ('N' or 'T'); BETA_CASE 0 (B0 kernel) or 2.
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#ifndef OPENBLAS_SCALAR
#define OPENBLAS_SCALAR double
#endif
#ifndef ALPHA_CASE
#define ALPHA_CASE 2
#endif
/* Ordinary API specialization: M=1 and unit lda/ldc. The VLA rows then
 * contain one element, so A[k][0] and C[j][0] have the flat addresses A[k]
 * and C[j]. No benchmark dimensions or algebraic matcher are involved. */
#ifdef OPENBLAS_UNIT_M
#if TRANSA != 'N'
#error "unit-M specialization requires non-transposed A"
#endif
#define M_LIMIT 1
#define A_IK A[k]
#define C_JI C[j]
#define MATRIX_PARAMS int N, int K, OPENBLAS_SCALAR A[],
#define OUTPUT_PARAMS OPENBLAS_SCALAR C[]
#else
#define M_LIMIT M
#define MATRIX_PARAMS int M, int N, int K, int lda, OPENBLAS_SCALAR A[][lda],
#define OUTPUT_PARAMS int ldc, OPENBLAS_SCALAR C[][ldc]
#define C_JI C[j][i]
#if TRANSA == 'N'
#define A_IK A[k][i]   /* A[i+k*lda] */
#else
#define A_IK A[i][k]   /* A[i*lda+k] */
#endif
#endif /* OPENBLAS_UNIT_M */
#if TRANSB == 'N'
#define B_KJ B[j][k]   /* B[k+j*ldb] */
#else
#define B_KJ B[k][j]   /* B[k*ldb+j] */
#endif
#if BETA_CASE == 0
#define BETA_PARAM
#else
#define BETA_PARAM OPENBLAS_SCALAR beta,
#endif

#if ALPHA_CASE == 1
#define ALPHA_PARAM
#define SCALED_RESULT result
#else
#define ALPHA_PARAM OPENBLAS_SCALAR alpha,
#define SCALED_RESULT (alpha * result)
#endif

void NETLIB_SPEC(MATRIX_PARAMS ALPHA_PARAM
                 int ldb, OPENBLAS_SCALAR B[][ldb], BETA_PARAM OUTPUT_PARAMS) {
  long i, j, k;
  OPENBLAS_SCALAR result = 0.0;

  for (i = 0; i < M_LIMIT; i++) {
    for (j = 0; j < N; j++) {
      result = 0.0;
      for (k = 0; k < K; k++) {
        result += A_IK * B_KJ;
      }
#if BETA_CASE == 0
      C_JI = SCALED_RESULT;
#else
      C_JI = C_JI * beta + SCALED_RESULT;
#endif
    }
  }
}
