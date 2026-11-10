/*
 * dger.c -- faithful C transliteration of Reference-BLAS DGER.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dger.f, SUBROUTINE at L129, executable part L163-L221.
 *
 *   A := alpha*x*y**T + A
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - column-major A(I,J) -> A[J][I]; index recurrences kept 0-based;
 *   - INCX.EQ.1 and the increment signs specialised by INC_CASE;
 *   - argument checking (L163-L178) dropped; the quick return L182
 *     (M.EQ.0 .OR. N.EQ.0 .OR. ALPHA.EQ.ZERO) dropped, so specs assume
 *     alpha != 0 (alpha == 0 makes DGER a no-op);
 *   - DEVIATION: the zero-skip guard `IF (Y(JY).NE.ZERO)` (L194 and L209) is
 *     removed; results differ from the reference only when y(j) == 0 and
 *     the column of x*0 would propagate NaN/Inf.
 *
 * Reference-LAPACK licence (BSD-3-Clause), reproduced from LICENSE at the
 * cited commit:
 *
 *   Copyright (c) 1992-2025 The University of Tennessee and The University
 *                           of Tennessee Research Foundation.  All rights
 *                           reserved.
 *   Copyright (c) 2000-2025 The University of California Berkeley. All
 *                           rights reserved.
 *   Copyright (c) 2006-2025 The University of Colorado Denver.  All rights
 *                           reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions are
 *   met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer listed
 *     in this license in the documentation and/or other materials
 *     provided with the distribution.
 *   - Neither the name of the copyright holders nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *   The copyright holders provide no reassurances that the source code
 *   provided does not infringe any patent, copyright, or any other
 *   intellectual property rights of third parties.  The copyright holders
 *   disclaim any liability to any recipient for claims brought against
 *   recipient by any third party for infringement of that parties
 *   intellectual property rights.
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
 */

/* ---- Specialisation macros (set with -D by build_lib_v2.py) -------------
 *   NETLIB_SPEC  name of the emitted C function (= spec name)
 *   ALPHA_CASE   0: alpha == 0 (alpha is not a parameter)
 *                2: general, alpha != 0 (default)
 *   BETA_CASE    0: beta == 0, 1: beta == 1 (beta is not a parameter)
 *                2: general, beta is a parameter, beta != 0 and beta != 1
 * LSAME is a case-sensitive compare: flags are passed upper case.
 * ------------------------------------------------------------------------ */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#define LSAME(CA, CB) ((CA) == (CB))
#ifndef ALPHA_CASE
#define ALPHA_CASE 2
#endif
#if ALPHA_CASE == 0
#define ALPHA_PARAM
#define ALPHA 0.0
#else
#define ALPHA_PARAM double alpha,
#define ALPHA alpha
#endif
#ifdef BETA_CASE
#if BETA_CASE == 0
#define BETA_PARAM
#define BETA 0.0
#elif BETA_CASE == 1
#define BETA_PARAM
#define BETA 1.0
#else
#define BETA_PARAM double beta,
#define BETA beta
#endif
#define BETA_EQ_ZERO (BETA_CASE == 0)
#define BETA_NE_ONE (BETA_CASE != 1)
#endif
#define ALPHA_EQ_ZERO (ALPHA_CASE == 0)

/* ---- Increment specialisation -------------------------------------------
 *   INC_CASE     1: incx == 1 and incy == 1 (default; not parameters)
 *                2: general increments as parameters, assumed > 0 and != 1
 * ------------------------------------------------------------------------ */
#ifndef INC_CASE
#define INC_CASE 1
#endif
#if INC_CASE == 1
#define INCX 1
#define INCY 1
#define INCX_PARAM
#define INCY_PARAM
#define INCX_EQ_ONE 1
#define INCY_EQ_ONE 1
#else
#define INCX incx
#define INCY incy
#define INCX_PARAM , int incx
#define INCY_PARAM , int incy
#define INCX_EQ_ONE 0
#define INCY_EQ_ONE 0
#endif
#define INCX_GT_ZERO 1
#define INCY_GT_ZERO 1

void NETLIB_SPEC(int m, int n, double alpha, double x[] INCX_PARAM,
                 double y[] INCY_PARAM, int lda, double A[][lda]) {
  double temp;
  int i, ix, j, jy, kx;

  /* L187-L191 (Fortran JY = 1 -> jy = 0) */
  if (INCY_GT_ZERO) {
    jy = 0;
  } else {
    jy = -(n - 1) * INCY;
  }
  if (INCX_EQ_ONE) {
    /* L193-L201 */
    for (j = 0; j < n; j++) {
      /* zero-skip guard IF (Y(JY).NE.ZERO) removed (deviation) */
      temp = alpha * y[jy];
      for (i = 0; i < m; i++) {
        A[j][i] = A[j][i] + x[i] * temp;
      }
      jy = jy + INCY;
    }
  } else {
    /* L203-L218 */
    if (INCX_GT_ZERO) {
      kx = 0;
    } else {
      kx = -(m - 1) * INCX;
    }
    for (j = 0; j < n; j++) {
      /* zero-skip guard IF (Y(JY).NE.ZERO) removed (deviation) */
      temp = alpha * y[jy];
      ix = kx;
      for (i = 0; i < m; i++) {
        A[j][i] = A[j][i] + x[ix] * temp;
        ix = ix + INCX;
      }
      jy = jy + INCY;
    }
  }
  /* L221 RETURN */
}
