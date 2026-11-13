/*
 * openblas_dscal.c -- C transliteration of the OpenBLAS DSCAL path.
 *
 * Source: OpenBLAS 0.3.31 (copy with SHA256SUMS in ~/netlib_ref/openblas-0.3.31),
 *   interface/scal.c L70-L96   incx/n checks; alpha == ONE return; SCAL_K(..., dummy2 = 1)
 *   kernel/arm/scal.c L39-L79  (dummy2 == 1 branch, da != 0.0)
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
 * alpha != 0 and alpha != 1 are assumed (the zeroing / NaN branch and the
 * interface's quick return are not taken).
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif

void NETLIB_SPEC(int n, double alpha, double x[]) {
  long i = 0, j = 0;
  while (j < n) {
    x[i] = alpha * x[i];
    i += 1;
    j++;
  }
}
