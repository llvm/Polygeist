/*
 * dtrmm.c -- faithful C transliteration of Reference-BLAS DTRMM.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dtrmm.f, SUBROUTINE at L176, executable part L216-L398.
 *
 *   B := alpha*op( A )*B   (SIDE = 'L')   or   B := alpha*B*op( A )
 *   (SIDE = 'R'), A triangular
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - column-major A(I,J) -> A[J][I]; reversed loops (DO K = M,1,-1 etc.)
 *     kept reversed; triangular bounds re-based to 0;
 *   - SIDE, UPLO, TRANSA ('N'/'T'), DIAG ('U'/'N') are compile-time
 *     character macros;
 *   - ALPHA.EQ.ZERO (L257-L264) specialised by ALPHA_CASE;
 *   - argument checking (L225-L249) and quick return L253 dropped;
 *   - DEVIATION (exact): the guards `IF (TEMP.NE.ONE)` (L372, L388) are
 *     removed; multiplying by 1.0 is the identity in IEEE arithmetic.
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
#ifndef TRANSA
#error "TRANSA must be 'N' or 'T'"
#endif
#ifndef DIAG
#error "DIAG must be 'U' or 'N'"
#endif

void NETLIB_SPEC(int m, int n, ALPHA_PARAM int lda, double A[][lda], int ldb,
                 double B[][ldb]) {
  const double ONE = 1.0, ZERO = 0.0; /* L210-L211 */
  double temp;
  int i, j, k, lside, nounit, upper;

  /* L216-L223 */
  lside = LSAME(SIDE, 'L');
  nounit = LSAME(DIAG, 'N');
  upper = LSAME(UPLO, 'U');

  /* L257-L264: And when  alpha.eq.zero. */
  if (ALPHA_EQ_ZERO) {
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        B[j][i] = ZERO;
      }
    }
    return;
  }

  /* L268: Start the operations. */
  if (lside) {
    if (LSAME(TRANSA, 'N')) {
      /* Form  B := alpha*A*B. */
      if (upper) {
        /* L274-L283 */
        for (j = 0; j < n; j++) {
          for (k = 0; k < m; k++) {
            temp = ALPHA * B[j][k];
            for (i = 0; i < k; i++) {
              B[j][i] = B[j][i] + temp * A[k][i];
            }
            if (nounit) temp = temp * A[k][k];
            B[j][k] = temp;
          }
        }
      } else {
        /* L285-L294 */
        for (j = 0; j < n; j++) {
          for (k = m - 1; k >= 0; k--) {
            temp = ALPHA * B[j][k];
            B[j][k] = temp;
            if (nounit) B[j][k] = B[j][k] * A[k][k];
            for (i = k + 1; i < m; i++) {
              B[j][i] = B[j][i] + temp * A[k][i];
            }
          }
        }
      }
    } else {
      /* Form  B := alpha*A**T*B. */
      if (upper) {
        /* L301-L310 */
        for (j = 0; j < n; j++) {
          for (i = m - 1; i >= 0; i--) {
            temp = B[j][i];
            if (nounit) temp = temp * A[i][i];
            for (k = 0; k < i; k++) {
              temp = temp + A[i][k] * B[j][k];
            }
            B[j][i] = ALPHA * temp;
          }
        }
      } else {
        /* L312-L321 */
        for (j = 0; j < n; j++) {
          for (i = 0; i < m; i++) {
            temp = B[j][i];
            if (nounit) temp = temp * A[i][i];
            for (k = i + 1; k < m; k++) {
              temp = temp + A[i][k] * B[j][k];
            }
            B[j][i] = ALPHA * temp;
          }
        }
      }
    }
  } else {
    if (LSAME(TRANSA, 'N')) {
      /* Form  B := alpha*B*A. */
      if (upper) {
        /* L330-L342 */
        for (j = n - 1; j >= 0; j--) {
          temp = ALPHA;
          if (nounit) temp = temp * A[j][j];
          for (i = 0; i < m; i++) {
            B[j][i] = temp * B[j][i];
          }
          for (k = 0; k < j; k++) {
            temp = ALPHA * A[j][k];
            for (i = 0; i < m; i++) {
              B[j][i] = B[j][i] + temp * B[k][i];
            }
          }
        }
      } else {
        /* L344-L356 */
        for (j = 0; j < n; j++) {
          temp = ALPHA;
          if (nounit) temp = temp * A[j][j];
          for (i = 0; i < m; i++) {
            B[j][i] = temp * B[j][i];
          }
          for (k = j + 1; k < n; k++) {
            temp = ALPHA * A[j][k];
            for (i = 0; i < m; i++) {
              B[j][i] = B[j][i] + temp * B[k][i];
            }
          }
        }
      }
    } else {
      /* Form  B := alpha*B*A**T. */
      if (upper) {
        /* L363-L377 */
        for (k = 0; k < n; k++) {
          for (j = 0; j < k; j++) {
            temp = ALPHA * A[k][j];
            for (i = 0; i < m; i++) {
              B[j][i] = B[j][i] + temp * B[k][i];
            }
          }
          temp = ALPHA;
          if (nounit) temp = temp * A[k][k];
          /* guard IF (TEMP.NE.ONE) removed (exact deviation) */
          for (i = 0; i < m; i++) {
            B[k][i] = temp * B[k][i];
          }
        }
      } else {
        /* L379-L393 */
        for (k = n - 1; k >= 0; k--) {
          for (j = k + 1; j < n; j++) {
            temp = ALPHA * A[k][j];
            for (i = 0; i < m; i++) {
              B[j][i] = B[j][i] + temp * B[k][i];
            }
          }
          temp = ALPHA;
          if (nounit) temp = temp * A[k][k];
          /* guard IF (TEMP.NE.ONE) removed (exact deviation) */
          for (i = 0; i < m; i++) {
            B[k][i] = temp * B[k][i];
          }
        }
      }
    }
  }
  /* L398 RETURN */
}
