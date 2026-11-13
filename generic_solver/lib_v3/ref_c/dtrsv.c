/*
 * netlib_dtrsv_<uplo><trans><diag> -- C transliteration of Reference BLAS DTRSV.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d,
 *         BLAS/SRC/dtrsv.f
 *           TRANS='N', UPLO='U', INCX=1 : lines 223-229
 *           TRANS='N', UPLO='L', INCX=1 : lines 245-251
 *           TRANS='T', UPLO='U', INCX=1 : lines 272-279
 *           TRANS='T', UPLO='L', INCX=1 : lines 296-303
 *         (TRANS='C' is identical to 'T' for real data, dtrsv.f:217/266.)
 *
 * Conventions (generic_solver/lib_v2/README.md):
 *   - Column-major storage is kept: Fortran A(I,J) with leading dimension LDA
 *     is the C99 VLA parameter `double A[][lda]` accessed as A[J-1][I-1].
 *   - Loop order, loop direction and arithmetic order follow the reference.
 *   - UPLO / TRANS / DIAG are compile-time macros, e.g.
 *       -DUPLO="'U'" -DTRANS="'T'" -DDIAG="'N'"   -> netlib_dtrsv_utn
 *     (or NETLIB_SPEC from the build_lib_v2.py wrapper names the function).
 *   - INCX = 1 specialisation (KX/JX/IX bookkeeping of the INCX != 1 paths
 *     is not transliterated).  XERBLA checks and the N = 0 quick return are
 *     dropped.  This version of DTRSV has no data-dependent zero-skip guards.
 *
 * ---------------------------------------------------------------------------
 * Copyright (c) 1992-2025 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2000-2025 The University of California Berkeley. All
 *                         rights reserved.
 * Copyright (c) 2006-2025 The University of Colorado Denver.  All rights
 *                         reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer listed
 *   in this license in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties.  The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------
 */

#ifndef UPLO
#error "define UPLO as 'U' or 'L'"
#endif
#ifndef TRANS
#error "define TRANS as 'N' or 'T'"
#endif
#ifndef DIAG
#error "define DIAG as 'N' or 'U'"
#endif

#if UPLO == 'U'
#define SPEC_U_ u
#else
#define SPEC_U_ l
#endif
#if TRANS == 'N'
#define SPEC_T_ n
#else
#define SPEC_T_ t
#endif
#if DIAG == 'N'
#define SPEC_D_ n
#else
#define SPEC_D_ u
#endif
#define SPEC_CAT_(a, b, c, d) a##b##c##d
#define SPEC_CAT(a, b, c, d) SPEC_CAT_(a, b, c, d)
#ifdef NETLIB_SPEC /* set by scripts/correctness/lib_v2/build_lib_v2.py wrappers */
#define SPEC_NAME NETLIB_SPEC
#else
#define SPEC_NAME SPEC_CAT(netlib_dtrsv_, SPEC_U_, SPEC_T_, SPEC_D_)
#endif

/* SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX) with INCX = 1 */
void SPEC_NAME(int n, int lda, double A[][lda], double X[]) {
#if TRANS == 'N'
  /* Form  x := inv( A )*x. */
#if UPLO == 'U'
  /* dtrsv.f:223-229
   *   DO 20 J = N,1,-1
   *       IF (NOUNIT) X(J) = X(J)/A(J,J)
   *       TEMP = X(J)
   *       DO 10 I = J - 1,1,-1
   *           X(I) = X(I) - TEMP*A(I,J)                                     */
  for (int j = n - 1; j >= 0; j--) {
#if DIAG == 'N'
    X[j] = X[j] / A[j][j];
#endif
    double temp = X[j];
    for (int i = j - 1; i >= 0; i--)
      X[i] = X[i] - temp * A[j][i];
  }
#else
  /* dtrsv.f:245-251
   *   DO 60 J = 1,N
   *       IF (NOUNIT) X(J) = X(J)/A(J,J)
   *       TEMP = X(J)
   *       DO 50 I = J + 1,N
   *           X(I) = X(I) - TEMP*A(I,J)                                     */
  for (int j = 0; j < n; j++) {
#if DIAG == 'N'
    X[j] = X[j] / A[j][j];
#endif
    double temp = X[j];
    for (int i = j + 1; i < n; i++)
      X[i] = X[i] - temp * A[j][i];
  }
#endif
#else
  /* Form  x := inv( A**T )*x. */
#if UPLO == 'U'
  /* dtrsv.f:272-279
   *   DO 100 J = 1,N
   *       TEMP = X(J)
   *       DO 90 I = 1,J - 1
   *           TEMP = TEMP - A(I,J)*X(I)
   *       IF (NOUNIT) TEMP = TEMP/A(J,J)
   *       X(J) = TEMP                                                       */
  for (int j = 0; j < n; j++) {
    double temp = X[j];
    for (int i = 0; i < j; i++)
      temp = temp - A[j][i] * X[i];
#if DIAG == 'N'
    temp = temp / A[j][j];
#endif
    X[j] = temp;
  }
#else
  /* dtrsv.f:296-303
   *   DO 140 J = N,1,-1
   *       TEMP = X(J)
   *       DO 130 I = N,J + 1,-1
   *           TEMP = TEMP - A(I,J)*X(I)
   *       IF (NOUNIT) TEMP = TEMP/A(J,J)
   *       X(J) = TEMP                                                       */
  for (int j = n - 1; j >= 0; j--) {
    double temp = X[j];
    for (int i = n - 1; i >= j + 1; i--)
      temp = temp - A[j][i] * X[i];
#if DIAG == 'N'
    temp = temp / A[j][j];
#endif
    X[j] = temp;
  }
#endif
#endif
}
