/*
 * dsymm.c -- faithful C transliteration of Reference-BLAS DSYMM.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dsymm.f, SUBROUTINE at L188, executable part L228-L362.
 *
 *   C := alpha*A*B + beta*C   (SIDE = 'L')   or
 *   C := alpha*B*A + beta*C   (SIDE = 'R'),  A symmetric
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - column-major A(I,J) -> A[J][I]; DO K = 1,I - 1 -> k = 0..i-1;
 *     DO I = M,1,-1 -> i = m-1..0; DO K = I + 1,M -> k = i+1..m-1;
 *   - SIDE ('L'/'R') and UPLO ('U'/'L') are compile-time character macros;
 *   - ALPHA.EQ.ZERO and BETA.EQ.ZERO specialised by ALPHA_CASE / BETA_CASE
 *     (DSYMM has no BETA.NE.ONE branch, so BETA_CASE is 0 or 2);
 *   - argument checking (L237-L257) and quick return L261-L262 dropped.
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

#ifndef SIDE
#error "SIDE must be 'L' or 'R'"
#endif
#ifndef UPLO
#error "UPLO must be 'U' or 'L'"
#endif
#ifndef BETA_CASE
#error "BETA_CASE must be 0 or 2"
#endif

void NETLIB_SPEC(int m, int n, ALPHA_PARAM int lda, double A[][lda], int ldb,
                 double B[][ldb], BETA_PARAM int ldc, double C[][ldc]) {
  const double ONE = 1.0, ZERO = 0.0; /* L222-L223 */
  double temp1, temp2;
  int i, j, k, upper;

  /* L233 */
  upper = LSAME(UPLO, 'U');

  /* L266-L281: And when  alpha.eq.zero. */
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

  /* L285: Start the operations. */
  if (LSAME(SIDE, 'L')) {
    /* Form  C := alpha*A*B + beta*C. */
    if (upper) {
      /* L290-L305 */
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          temp1 = ALPHA * B[j][i];
          temp2 = ZERO;
          for (k = 0; k < i; k++) {
            C[j][k] = C[j][k] + temp1 * A[i][k];
            temp2 = temp2 + B[j][k] * A[i][k];
          }
          if (BETA_EQ_ZERO) {
            C[j][i] = temp1 * A[i][i] + ALPHA * temp2;
          } else {
            C[j][i] = BETA * C[j][i] + temp1 * A[i][i] + ALPHA * temp2;
          }
        }
      }
    } else {
      /* L307-L322 */
      for (j = 0; j < n; j++) {
        for (i = m - 1; i >= 0; i--) {
          temp1 = ALPHA * B[j][i];
          temp2 = ZERO;
          for (k = i + 1; k < m; k++) {
            C[j][k] = C[j][k] + temp1 * A[i][k];
            temp2 = temp2 + B[j][k] * A[i][k];
          }
          if (BETA_EQ_ZERO) {
            C[j][i] = temp1 * A[i][i] + ALPHA * temp2;
          } else {
            C[j][i] = BETA * C[j][i] + temp1 * A[i][i] + ALPHA * temp2;
          }
        }
      }
    }
  } else {
    /* L328-L359: Form  C := alpha*B*A + beta*C. */
    for (j = 0; j < n; j++) {
      temp1 = ALPHA * A[j][j];
      if (BETA_EQ_ZERO) {
        for (i = 0; i < m; i++) {
          C[j][i] = temp1 * B[j][i];
        }
      } else {
        for (i = 0; i < m; i++) {
          C[j][i] = BETA * C[j][i] + temp1 * B[j][i];
        }
      }
      for (k = 0; k < j; k++) {
        if (upper) {
          temp1 = ALPHA * A[j][k];
        } else {
          temp1 = ALPHA * A[k][j];
        }
        for (i = 0; i < m; i++) {
          C[j][i] = C[j][i] + temp1 * B[k][i];
        }
      }
      for (k = j + 1; k < n; k++) {
        if (upper) {
          temp1 = ALPHA * A[k][j];
        } else {
          temp1 = ALPHA * A[j][k];
        }
        for (i = 0; i < m; i++) {
          C[j][i] = C[j][i] + temp1 * B[k][i];
        }
      }
    }
  }
  /* L362 RETURN */
}
