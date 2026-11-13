/*
 * dgemm.c -- faithful C transliteration of Reference-BLAS DGEMM.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dgemm.f, SUBROUTINE at L213, executable part L256-L404.
 *
 *   C := alpha*op( A )*op( B ) + beta*C
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - column-major storage kept: Fortran A(I,J) -> A[J][I] on `double A[][lda]`;
 *   - loop order / direction / arithmetic order of the reference kept;
 *     loops are re-based from 1..N to 0..N-1;
 *   - TRANSA, TRANSB are compile-time character macros ('N' or 'T');
 *   - ALPHA.EQ.ZERO, BETA.EQ.ZERO, BETA.NE.ONE are specialised by
 *     ALPHA_CASE / BETA_CASE;
 *   - argument checking (L271-L294, XERBLA) and the quick return L298-L299
 *     are dropped; NROWA/NROWB (L258-L267) only feed argument checking.
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

#ifndef TRANSA
#error "TRANSA must be 'N' or 'T'"
#endif
#ifndef TRANSB
#error "TRANSB must be 'N' or 'T'"
#endif
#ifndef BETA_CASE
#error "BETA_CASE must be 0, 1 or 2"
#endif

void NETLIB_SPEC(int m, int n, int k, ALPHA_PARAM int lda, double A[][lda],
                 int ldb, double B[][ldb], BETA_PARAM int ldc,
                 double C[][ldc]) {
  const double ONE = 1.0, ZERO = 0.0; /* L248-L249 */
  double temp;
  int i, j, l, nota, notb;

  /* L256-L257 */
  nota = LSAME(TRANSA, 'N');
  notb = LSAME(TRANSB, 'N');

  /* L303-L318: And if alpha.eq.zero. */
  if (ALPHA_EQ_ZERO) {
    if (BETA_EQ_ZERO) {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          C[j][i] = ZERO;
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          C[j][i] = BETA * C[j][i];
        }
      }
    }
    return;
  }

  /* L322: Start the operations. */
  if (notb) {
    if (nota) {
      /* L327-L343: Form  C := alpha*A*B + beta*C. */
      for (j = 0; j < n; j++) {
        if (BETA_EQ_ZERO) {
          for (i = 0; i < m; i++) {
            C[j][i] = ZERO;
          }
        } else if (BETA_NE_ONE) {
          for (i = 0; i < m; i++) {
            C[j][i] = BETA * C[j][i];
          }
        }
        for (l = 0; l < k; l++) {
          temp = ALPHA * B[j][l];
          for (i = 0; i < m; i++) {
            C[j][i] = C[j][i] + temp * A[l][i];
          }
        }
      }
    } else {
      /* L348-L360: Form  C := alpha*A**T*B + beta*C */
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          temp = ZERO;
          for (l = 0; l < k; l++) {
            temp = temp + A[i][l] * B[j][l];
          }
          if (BETA_EQ_ZERO) {
            C[j][i] = ALPHA * temp;
          } else {
            C[j][i] = ALPHA * temp + BETA * C[j][i];
          }
        }
      }
    }
  } else {
    if (nota) {
      /* L367-L383: Form  C := alpha*A*B**T + beta*C */
      for (j = 0; j < n; j++) {
        if (BETA_EQ_ZERO) {
          for (i = 0; i < m; i++) {
            C[j][i] = ZERO;
          }
        } else if (BETA_NE_ONE) {
          for (i = 0; i < m; i++) {
            C[j][i] = BETA * C[j][i];
          }
        }
        for (l = 0; l < k; l++) {
          temp = ALPHA * B[l][j];
          for (i = 0; i < m; i++) {
            C[j][i] = C[j][i] + temp * A[l][i];
          }
        }
      }
    } else {
      /* L388-L400: Form  C := alpha*A**T*B**T + beta*C */
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          temp = ZERO;
          for (l = 0; l < k; l++) {
            temp = temp + A[i][l] * B[l][j];
          }
          if (BETA_EQ_ZERO) {
            C[j][i] = ALPHA * temp;
          } else {
            C[j][i] = ALPHA * temp + BETA * C[j][i];
          }
        }
      }
    }
  }
  /* L404 RETURN */
}
