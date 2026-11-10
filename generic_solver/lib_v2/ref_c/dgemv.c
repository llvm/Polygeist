/*
 * dgemv.c -- faithful C transliteration of Reference-BLAS DGEMV.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dgemv.f, SUBROUTINE at L157, executable part L196-L326.
 *
 *   y := alpha*A*x + beta*y   or   y := alpha*A**T*x + beta*y
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - column-major A(I,J) -> A[J][I]; X(*) / Y(*) -> double x[] / y[];
 *   - Fortran 1-based index variables KX, KY, JX, JY, IX, IY are kept as
 *     0-based recurrences (Fortran value - 1), same update order;
 *   - TRANS is a compile-time character macro ('N' or 'T');
 *   - ALPHA.EQ.ZERO (L273), BETA.EQ.ZERO, BETA.NE.ONE (L247-L272) are
 *     specialised by ALPHA_CASE / BETA_CASE; INCX.EQ.1 / INCY.EQ.1 and the
 *     sign of the increments by INC_CASE;
 *   - argument checking (L196-L214) and the quick return L218-L219 dropped.
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

#ifndef TRANS
#error "TRANS must be 'N' or 'T'"
#endif
#ifndef BETA_CASE
#error "BETA_CASE must be 0, 1 or 2"
#endif

void NETLIB_SPEC(int m, int n, ALPHA_PARAM int lda, double A[][lda],
                 double x[] INCX_PARAM, BETA_PARAM double y[] INCY_PARAM) {
  const double ONE = 1.0, ZERO = 0.0; /* L176-L177 */
  double temp;
  int i, ix, iy, j, jx, jy, kx, ky, lenx, leny;

  /* L224-L230: Set LENX and LENY, the lengths of the vectors x and y. */
  if (LSAME(TRANS, 'N')) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }
  /* L231-L240 (Fortran KX = 1 -> kx = 0) */
  if (INCX_GT_ZERO) {
    kx = 0;
  } else {
    kx = -(lenx - 1) * INCX;
  }
  if (INCY_GT_ZERO) {
    ky = 0;
  } else {
    ky = -(leny - 1) * INCY;
  }

  /* L247-L272: First form  y := beta*y. */
  if (BETA_NE_ONE) {
    if (INCY_EQ_ONE) {
      if (BETA_EQ_ZERO) {
        for (i = 0; i < leny; i++) {
          y[i] = ZERO;
        }
      } else {
        for (i = 0; i < leny; i++) {
          y[i] = BETA * y[i];
        }
      }
    } else {
      iy = ky;
      if (BETA_EQ_ZERO) {
        for (i = 0; i < leny; i++) {
          y[iy] = ZERO;
          iy = iy + INCY;
        }
      } else {
        for (i = 0; i < leny; i++) {
          y[iy] = BETA * y[iy];
          iy = iy + INCY;
        }
      }
    }
  }
  /* L273 */
  if (ALPHA_EQ_ZERO) return;
  if (LSAME(TRANS, 'N')) {
    /* L278-L297: Form  y := alpha*A*x + y. */
    jx = kx;
    if (INCY_EQ_ONE) {
      for (j = 0; j < n; j++) {
        temp = ALPHA * x[jx];
        for (i = 0; i < m; i++) {
          y[i] = y[i] + temp * A[j][i];
        }
        jx = jx + INCX;
      }
    } else {
      for (j = 0; j < n; j++) {
        temp = ALPHA * x[jx];
        iy = ky;
        for (i = 0; i < m; i++) {
          y[iy] = y[iy] + temp * A[j][i];
          iy = iy + INCY;
        }
        jx = jx + INCX;
      }
    }
  } else {
    /* L302-L323: Form  y := alpha*A**T*x + y. */
    jy = ky;
    if (INCX_EQ_ONE) {
      for (j = 0; j < n; j++) {
        temp = ZERO;
        for (i = 0; i < m; i++) {
          temp = temp + A[j][i] * x[i];
        }
        y[jy] = y[jy] + ALPHA * temp;
        jy = jy + INCY;
      }
    } else {
      for (j = 0; j < n; j++) {
        temp = ZERO;
        ix = kx;
        for (i = 0; i < m; i++) {
          temp = temp + A[j][i] * x[ix];
          ix = ix + INCX;
        }
        y[jy] = y[jy] + ALPHA * temp;
        jy = jy + INCY;
      }
    }
  }
  /* L326 RETURN */
}
