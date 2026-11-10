/*
 * csc_matvec.c -- faithful C transliteration of SciPy sparsetools CSC_MATVEC.
 *
 * Source: scipy/scipy commit f0352fce59c631194fec178ca20c50cb345de114,
 *         scipy/sparse/sparsetools/csc.h, template csc_matvec at L28-L45.
 *
 *   Y += A*X   for a CSC matrix A (column pointers Ap, row indices Ai,
 *              values Ax) and dense vectors X, Y.  Every update is a
 *              scatter-add into Y at a row index read from Ai.
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - the template is instantiated with I = INDEX_T and T = double; loop
 *     order and arithmetic order are kept verbatim.  SciPy instantiates I as
 *     npy_int32 and npy_int64 (scipy/sparse/_generate_sparsetools.py,
 *     I_TYPES, same commit); INDEX_T = int (cuSPARSE CUSPARSE_INDEX_32I) or
 *     long long (CUSPARSE_INDEX_64I) gives the two instances;
 *   - BETA_CASE specialises how Y enters, as for the BLAS specs:
 *       1: the reference itself                            (Y += A*X)
 *       0: Y is zero-filled first, then the reference      (Y  = A*X)
 *     BETA_CASE 0 is listed as a deviation in the manifest (SciPy has no
 *     such branch).
 *
 * Licence of the source:
 *   Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *   3. Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Macros:
 *   NETLIB_SPEC  name of the emitted C function (= spec name)
 *   BETA_CASE    0 or 1 (see above)
 *   INDEX_T      the index type I (default int)
 */

#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#ifndef BETA_CASE
#error "BETA_CASE must be 0 or 1"
#endif
#ifndef INDEX_T
#define INDEX_T int
#endif

void NETLIB_SPEC(const INDEX_T n_row, const INDEX_T n_col, const INDEX_T Ap[],
                 const INDEX_T Ai[], const double Ax[], const double Xx[],
                 double Yx[]) {
#if BETA_CASE == 0
  for (INDEX_T i = 0; i < n_row; i++)
    Yx[i] = 0.0;
#else
  (void)n_row;
#endif
  /* L36-L44 */
  for (INDEX_T j = 0; j < n_col; j++) {
    INDEX_T col_start = Ap[j];
    INDEX_T col_end = Ap[j + 1];

    for (INDEX_T ii = col_start; ii < col_end; ii++) {
      INDEX_T i = Ai[ii];
      Yx[i] += Ax[ii] * Xx[j];
    }
  }
}
