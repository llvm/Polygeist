/*
 * csr_matvec.c -- faithful C transliteration of SciPy sparsetools CSR_MATVEC.
 *
 * Source: scipy/scipy commit f0352fce59c631194fec178ca20c50cb345de114,
 *         scipy/sparse/sparsetools/csr.h, template csr_matvec at L1131-L1146.
 *
 *   Y += A*X   for a CSR matrix A (row pointers Ap, column indices Aj,
 *              values Ax) and dense vectors X, Y.
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - the template is instantiated with I = INDEX_T and T = double; loop
 *     order and arithmetic order are kept verbatim.  SciPy instantiates I as
 *     npy_int32 and npy_int64 (scipy/sparse/_generate_sparsetools.py,
 *     I_TYPES, same commit); INDEX_T = int (cuSPARSE CUSPARSE_INDEX_32I) or
 *     long long (CUSPARSE_INDEX_64I) gives the two instances;
 *   - BETA_CASE specialises how Y enters, as for the BLAS specs:
 *       1: the reference itself, sum starts at Yx[i]       (Y += A*X)
 *       0: sum starts at zero and Yx is not read           (Y  = A*X)
 *     BETA_CASE 0 is the reference applied to a zero-filled Y (listed as a
 *     deviation in the manifest: SciPy has no such branch).
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
                 const INDEX_T Aj[], const double Ax[], const double Xx[],
                 double Yx[]) {
  (void)n_col;
  /* L1139-L1145 */
  for (INDEX_T i = 0; i < n_row; i++) {
#if BETA_CASE == 1
    double sum = Yx[i];
#else
    double sum = 0.0;
#endif
    for (INDEX_T jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      sum += Ax[jj] * Xx[Aj[jj]];
    }
    Yx[i] = sum;
  }
}
