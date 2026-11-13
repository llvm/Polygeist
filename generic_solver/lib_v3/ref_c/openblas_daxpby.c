/*
 * openblas_daxpby.c -- C transliteration of the OpenBLAS DAXPBY extension.
 *
 * Source: OpenBLAS 0.3.31 (copy with SHA256SUMS in ~/netlib_ref/openblas-0.3.31),
 *   interface/axpby.c L57-L64   n <= 0 return; AXPBY_K(n, alpha, x, 1, beta, y, 1)
 *   kernel/arm/axpby.c L31-L94
 *
 *   y = alpha*x + beta*y.
 *
 * Transliteration rules (generic_solver/lib_v3/README.md): FLOAT = double,
 * BLASLONG = long, blasint = int; inc_x = inc_y = 1; loop order and
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
 * Macros: BETA_CASE 0 (beta == 0: y = alpha*x) or 2 (general); alpha != 0.
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#if BETA_CASE == 2
#define BETA_PARAM double beta,
#else
#define BETA_PARAM
#endif

void NETLIB_SPEC(int n, double alpha, double x[], BETA_PARAM double y[]) {
  long i = 0;
  long ix = 0, iy = 0;
  while (i < n) {
#if BETA_CASE == 0
    y[iy] = alpha * x[ix];
#else
    y[iy] = alpha * x[ix] + beta * y[iy];
#endif
    ix += 1;
    iy += 1;
    i++;
  }
}
