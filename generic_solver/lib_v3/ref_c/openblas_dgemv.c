/*
 * openblas_dgemv.c -- C transliteration of the OpenBLAS single-thread DGEMV
 * path with its portable C kernels.
 *
 * Source: OpenBLAS 0.3.31 (copy with SHA256SUMS in ~/netlib_ref/openblas-0.3.31),
 *   interface/gemv.c L269-L294  if (beta != ONE) SCAL_K(leny, ..., dummy2 = 0);
 *                               if (alpha == ZERO) return; gemv[trans](...)
 *   kernel/arm/scal.c L39-L79   (dummy2 == 0 branch, inlined)
 *   kernel/arm/gemv_n.c L41-L65 (trans = 0)   kernel/arm/gemv_t.c L40-L66 (trans = 1)
 *
 * Transliteration rules (generic_solver/lib_v3/README.md): FLOAT = double,
 * BLASLONG = long, blasint = int; inc_x = inc_y = 1; column-major addressing
 * a[i + j*lda] (or a pointer bumped by lda per column) becomes A[j][i] of the
 * VLA parameter double A[][lda] (the same address); loop order and
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
 * Macros: TRANS 'N' (interface trans = 0, gemv_n) or 'T' (trans = 1, gemv_t); BETA_CASE 0 (beta == 0),
 * 1 (beta == 1: SCAL_K skipped) or 2 (general). alpha != 0 is assumed.
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#if BETA_CASE == 2
#define BETA_PARAM double beta,
#else
#define BETA_PARAM
#endif

void NETLIB_SPEC(int m, int n, double alpha, int lda, double A[][lda],
                 double x[], BETA_PARAM double y[]) {
  long i, j, ix, iy;
  double temp;
#if BETA_CASE != 1
  { /* SCAL_K(leny, 0, 0, beta, y, 1, NULL, 0, NULL, 0): scal.c dummy2 == 0 */
    long si = 0, sj = 0;
    long leny = (TRANS == 'N') ? m : n;   /* gemv.c: if (trans) leny = n */
    while (sj < leny) {
#if BETA_CASE == 0
      y[si] = 0.0;
#else
      y[si] = beta * y[si];
#endif
      si += 1;
      sj++;
    }
  }
#endif
#if TRANS == 'N'
  /* gemv_n.c: a_ptr = a + j*lda -> A[j] */
  ix = 0;
  for (j = 0; j < n; j++) {
    temp = alpha * x[ix];
    iy = 0;
    for (i = 0; i < m; i++) {
      y[iy] += temp * A[j][i];
      iy += 1;
    }
    ix += 1;
  }
#else
  /* gemv_t.c */
  iy = 0;
  for (j = 0; j < n; j++) {
    temp = 0.0;
    ix = 0;
    for (i = 0; i < m; i++) {
      temp += A[j][i] * x[ix];
      ix += 1;
    }
    y[iy] += alpha * temp;
    iy += 1;
  }
#endif
}
