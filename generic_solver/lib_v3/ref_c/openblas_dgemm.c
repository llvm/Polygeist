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
 * Macros: TRANSA, TRANSB ('N' or 'T'); BETA_CASE 0 (B0), 1, or 2.
 * BETA_CASE=1 specializes the ordinary beta argument to one.
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
/* Ordinary dimension specializations of the same column-major GEMM.
 * Unit physical dimensions use flat pointers with identical addresses.
 * No scalar arithmetic or accumulation order is changed. */
#if defined(OPENBLAS_UNIT_K) && defined(OPENBLAS_UNIT_M)
#error "unit-K and unit-M specializations are separate"
#endif
#if defined(OPENBLAS_UNIT_N) && !defined(OPENBLAS_UNIT_M)
#error "unit-N specialization also requires unit-M"
#endif
#ifdef OPENBLAS_UNIT_K
#if TRANSA != 'N' || TRANSB != 'N'
#error "unit-K specialization requires non-transposed operands"
#endif
#define M_LIMIT M
#define N_LIMIT N
#define K_LIMIT 1
#define MATRIX_PARAMS int M, int N, OPENBLAS_SCALAR A[],
#define B_PARAMS OPENBLAS_SCALAR B[],
#define OUTPUT_PARAMS int ldc, OPENBLAS_SCALAR C[][ldc]
#define A_IK A[i]
#define B_KJ B[j]
#define C_JI C[j][i]
#else
#define K_LIMIT K
#ifdef OPENBLAS_UNIT_M
#if TRANSA != 'N'
#error "unit-M specialization requires non-transposed A"
#endif
#define M_LIMIT 1
#define A_IK A[k]
#define C_JI C[j]
#define OUTPUT_PARAMS OPENBLAS_SCALAR C[]
#ifdef OPENBLAS_UNIT_N
#if TRANSB != 'N'
#error "unit-N specialization requires non-transposed B"
#endif
#define N_LIMIT 1
#define MATRIX_PARAMS int K, OPENBLAS_SCALAR A[],
#define B_PARAMS OPENBLAS_SCALAR B[],
#define B_KJ B[k]
#else
#define N_LIMIT N
#define MATRIX_PARAMS int N, int K, OPENBLAS_SCALAR A[],
#endif
#else
#define M_LIMIT M
#define N_LIMIT N
#define MATRIX_PARAMS int M, int N, int K, int lda, OPENBLAS_SCALAR A[][lda],
#define OUTPUT_PARAMS int ldc, OPENBLAS_SCALAR C[][ldc]
#define C_JI C[j][i]
#if TRANSA == 'N'
#define A_IK A[k][i]
#else
#define A_IK A[i][k]
#endif
#endif
#ifndef OPENBLAS_UNIT_N
#define B_PARAMS int ldb, OPENBLAS_SCALAR B[][ldb],
#if TRANSB == 'N'
#define B_KJ B[j][k]
#else
#define B_KJ B[k][j]
#endif
#endif
#endif
#if BETA_CASE != 2
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
                 B_PARAMS BETA_PARAM OUTPUT_PARAMS) {
  long i, j, k;
  OPENBLAS_SCALAR result = 0.0;

  for (i = 0; i < M_LIMIT; i++) {
    for (j = 0; j < N_LIMIT; j++) {
      result = 0.0;
      for (k = 0; k < K_LIMIT; k++) {
        result += A_IK * B_KJ;
      }
#if BETA_CASE == 0
      C_JI = SCALED_RESULT;
#elif BETA_CASE == 1
      C_JI = C_JI + SCALED_RESULT;
#else
      C_JI = C_JI * beta + SCALED_RESULT;
#endif
    }
  }
}
