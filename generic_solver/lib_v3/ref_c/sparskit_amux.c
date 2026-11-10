/*
 * sparskit_amux.c -- faithful C transliteration of SPARSKIT2 AMUX.
 *
 * Source: SPARSKIT2 (Y. Saad), BLASSM/matvec.f, subroutine amux at L34-L76
 *         (SPARSKIT2.tar.gz from www-users.cse.umn.edu/~saad/software/SPARSKIT,
 *         matvec.f md5 8c9ee5d16411920e357df19a7e359b8e).
 *
 *   y = A*x   for an n-row CSR matrix A stored the Fortran way: row pointers
 *             ia, column indices ja and values a all hold 1-based positions
 *             (cuSPARSE CUSPARSE_INDEX_BASE_ONE).
 *
 * Transliteration rules:
 *   - every Fortran subscript X(S) becomes X[S - 1]; loop variables keep
 *     their Fortran values (i = 1..n, k = ia(i)..ia(i+1)-1), so the index
 *     data is used exactly as SPARSKIT uses it;
 *   - loop order and arithmetic order are kept verbatim; integer = int,
 *     real*8 = double; arguments are passed in Fortran order.
 *
 * Licence of the source:
 *   Copyright (C) 2005  the Regents of the University of Minnesota
 *
 *   SPARSKIT is free software; you can redistribute it and/or modify it
 *   under the terms of the GNU Lesser General Public License as published
 *   by the Free Software Foundation [version 2.1 of the License, or any
 *   later version.]  This file is a derivative of SPARSKIT and is
 *   distributed under the same licence.  It is distributed WITHOUT ANY
 *   WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *   License for more details.
 *
 * Macros:
 *   NETLIB_SPEC  name of the emitted C function (= spec name)
 */

#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif

void NETLIB_SPEC(int n, const double x[], double y[], const double a[],
                 const int ja[], const int ia[]) {
  double t;
  int i, k;
  /* L61-L73 */
  for (i = 1; i <= n; i++) {                  /* do 100 i = 1,n            */
    t = 0.0;                                  /*    t = 0.0d0              */
    for (k = ia[i - 1]; k <= ia[i] - 1; k++)  /*    do 99 k=ia(i),ia(i+1)-1 */
      t = t + a[k - 1] * x[ja[k - 1] - 1];    /*       t = t + a(k)*x(ja(k)) */
    y[i - 1] = t;                             /*    y(i) = t               */
  }
}
