/*
 * netlib_dpotf2_<uplo> -- C transliteration of LAPACK DPOTF2 (unblocked
 * Cholesky), with the called Reference BLAS routines inlined.
 *
 * Source: Reference-LAPACK commit 6a1a8b8afd07ef4f6afc9fff320771fa26671f8d
 *   SRC/dpotf2.f        UPLO='U': lines 170-189, UPLO='L': lines 194-215
 *   BLAS/SRC/ddot.f     general-increment loop, lines 134-142
 *   BLAS/SRC/dgemv.f    TRANS='T', INCX=1 : lines 302-311 (BETA=ONE, so the
 *                       BETA scaling block at lines 247-272 folds away)
 *                       TRANS='N', INCY=1 : lines 278-286
 *   BLAS/SRC/dscal.f    general-increment loop, lines 131-134
 *
 * Conventions (generic_solver/lib_v2/README.md):
 *   - Column-major storage is kept: A(I,J) -> A[J-1][I-1] on `double A[][lda]`.
 *   - Loop order, direction and arithmetic order of the reference are kept,
 *     including DGEMV's ALPHA = -ONE multiply and DSCAL's DA = ONE/AJJ.
 *   - UPLO is a compile-time macro: -DUPLO="'U'" -> netlib_dpotf2_u.
 *   - XERBLA checks and the N = 0 quick return are dropped, as are the
 *     DGEMV / DSCAL / DDOT size quick returns (they guard zero-trip loops).
 *
 * Deviations (also listed in manifest/dpotf2.json):
 *   - The non-positive-definite exit (dpotf2.f:175-178 / 200-203,
 *     IF (AJJ.LE.ZERO .OR. DISNAN(AJJ)) ... GO TO 30, INFO = J) is removed:
 *     the spec describes the INFO = 0 path (A symmetric positive definite).
 *   - DDOT with INCX = INCY = 1 (UPLO='U') and DSCAL with INCX = 1
 *     (UPLO='L') use the general-increment loops instead of the 5-way
 *     unrolled loops.  Fortran evaluates DTEMP + p1 + p2 + ... left to right
 *     and DSCAL is element-wise, so the arithmetic is bitwise identical.
 *   - DSCAL's DA.EQ.ONE quick return is removed (exact: 1*x == x).
 *   - The IF( J.LT.N ) guard around DGEMV/DSCAL (dpotf2.f:184/209) is
 *     removed.  For J = N both calls have N-J = 0 iterations, i.e. it is a
 *     size quick return; dropping it is exact.  Keeping it produces an
 *     affine.if that blocks --linalg-debufferize (see RAISE_STATUS.md).
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
#include <math.h>

#ifndef UPLO
#error "define UPLO as 'U' or 'L'"
#endif

#ifdef NETLIB_SPEC /* set by scripts/correctness/lib_v2/build_lib_v2.py wrappers */
#define SPEC_NAME NETLIB_SPEC
#elif UPLO == 'U'
#define SPEC_NAME netlib_dpotf2_u
#else
#define SPEC_NAME netlib_dpotf2_l
#endif

/* SUBROUTINE DPOTF2( UPLO, N, A, LDA, INFO ), INFO = 0 path */
void SPEC_NAME(int n, int lda, double A[][lda]) {
#if UPLO == 'U'
  /* Compute the Cholesky factorization A = U**T *U.  (dpotf2.f:170-189) */
  for (int j = 0; j < n; j++) {
    /* AJJ = A( J, J ) - DDOT( J-1, A( 1, J ), 1, A( 1, J ), 1 ) */
    double dtemp = 0.0;
    for (int i = 0; i < j; i++)
      dtemp = dtemp + A[j][i] * A[j][i];
    double ajj = A[j][j] - dtemp;
    /* AJJ = SQRT( AJJ ); A( J, J ) = AJJ */
    ajj = sqrt(ajj);
    A[j][j] = ajj;
    /* Compute elements J+1:N of row J.
     * IF( J.LT.N ) is dropped: for J = N both calls have N-J = 0 trips
     * (size quick return), see the deviations list above. */
    /* DGEMV( 'Transpose', J-1, N-J, -ONE, A( 1, J+1 ), LDA, A( 1, J ), 1,
     *        ONE, A( J, J+1 ), LDA )
     *   DO 100 J = 1,N:  TEMP = ZERO
     *                    DO 90 I = 1,M: TEMP = TEMP + A(I,J)*X(I)
     *                    Y(JY) = Y(JY) + ALPHA*TEMP                         */
    for (int jj = 0; jj < n - j - 1; jj++) {
      double temp = 0.0;
      for (int i = 0; i < j; i++)
        temp = temp + A[j + 1 + jj][i] * A[j][i];
      A[j + 1 + jj][j] = A[j + 1 + jj][j] + (-1.0) * temp;
    }
    /* DSCAL( N-J, ONE / AJJ, A( J, J+1 ), LDA ):  DX(I) = DA*DX(I) */
    double da = 1.0 / ajj;
    for (int i = 0; i < n - j - 1; i++)
      A[j + 1 + i][j] = da * A[j + 1 + i][j];
  }
#else
  /* Compute the Cholesky factorization A = L*L**T.  (dpotf2.f:194-215) */
  for (int j = 0; j < n; j++) {
    /* AJJ = A( J, J ) - DDOT( J-1, A( J, 1 ), LDA, A( J, 1 ), LDA ) */
    double dtemp = 0.0;
    for (int i = 0; i < j; i++)
      dtemp = dtemp + A[i][j] * A[i][j];
    double ajj = A[j][j] - dtemp;
    /* AJJ = SQRT( AJJ ); A( J, J ) = AJJ */
    ajj = sqrt(ajj);
    A[j][j] = ajj;
    /* Compute elements J+1:N of column J.
     * IF( J.LT.N ) is dropped: for J = N both calls have N-J = 0 trips
     * (size quick return), see the deviations list above. */
    /* DGEMV( 'No transpose', N-J, J-1, -ONE, A( J+1, 1 ), LDA, A( J, 1 ),
     *        LDA, ONE, A( J+1, J ), 1 )
     *   DO 60 J = 1,N:  TEMP = ALPHA*X(JX)
     *                   DO 50 I = 1,M: Y(I) = Y(I) + TEMP*A(I,J)            */
    for (int jj = 0; jj < j; jj++) {
      double temp = (-1.0) * A[jj][j];
      for (int i = 0; i < n - j - 1; i++)
        A[j][j + 1 + i] = A[j][j + 1 + i] + temp * A[jj][j + 1 + i];
    }
    /* DSCAL( N-J, ONE / AJJ, A( J+1, J ), 1 ):  DX(I) = DA*DX(I) */
    double da = 1.0 / ajj;
    for (int i = 0; i < n - j - 1; i++)
      A[j][j + 1 + i] = da * A[j][j + 1 + i];
  }
#endif
}
