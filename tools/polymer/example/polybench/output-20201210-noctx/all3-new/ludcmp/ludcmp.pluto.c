#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// TODO: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE w;

/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24;
 register int lbv, ubv;
/* Start of CLooG code */
if (_PB_N >= 1) {
  if (_PB_N >= 3) {
    w = A[0][0];;
    A[0][0] = w;;
    for (t23=1;t23<=_PB_N-1;t23++) {
      for (t24=0;t24<=t23-1;t24++) {
        w = A[t23][t24];;
      }
    }
    for (t20=2;t20<=_PB_N-1;t20++) {
      for (t21=1;t21<=t20-1;t21++) {
        for (t22=0;t22<=t21-1;t22++) {
          w -= A[t20][t22] * A[t22][t21];;
        }
      }
    }
    for (t18=1;t18<=_PB_N-1;t18++) {
      lbv=0;
      ubv=t18-1;
#pragma ivdep
#pragma vector always
      for (t19=lbv;t19<=ubv;t19++) {
        A[t18][t19] = w / A[t19][t19];;
      }
    }
    for (t17=1;t17<=_PB_N-1;t17++) {
      w = A[0][t17];;
    }
    for (t16=1;t16<=_PB_N-1;t16++) {
      for (t17=t16;t17<=_PB_N-1;t17++) {
        w = A[t16][t17];;
      }
    }
    for (t13=1;t13<=_PB_N-1;t13++) {
      for (t14=t13;t14<=_PB_N-1;t14++) {
        for (t15=0;t15<=t13-1;t15++) {
          w -= A[t13][t15] * A[t15][t14];;
        }
      }
    }
    lbv=1;
    ubv=_PB_N-1;
#pragma ivdep
#pragma vector always
    for (t12=lbv;t12<=ubv;t12++) {
      A[0][t12] = w;;
    }
    for (t11=1;t11<=_PB_N-1;t11++) {
      lbv=t11;
      ubv=_PB_N-1;
#pragma ivdep
#pragma vector always
      for (t12=lbv;t12<=ubv;t12++) {
        A[t11][t12] = w;;
      }
    }
  }
  if (_PB_N == 2) {
    w = A[0][0];;
    A[0][0] = w;;
    w = A[1][0];;
    A[1][0] = w / A[0][0];;
    w = A[0][1];;
    w = A[1][1];;
    w -= A[1][0] * A[0][1];;
    A[0][1] = w;;
    A[1][1] = w;;
  }
  if (_PB_N == 1) {
    w = A[0][0];;
    A[0][0] = w;;
  }
  if (_PB_N >= 2) {
    w = b[0];;
    y[0] = w;;
    for (t10=1;t10<=_PB_N-1;t10++) {
      w = b[t10];;
    }
    for (t8=1;t8<=_PB_N-1;t8++) {
      for (t9=0;t9<=t8-1;t9++) {
        w -= A[t8][t9] * y[t9];;
      }
    }
    lbv=1;
    ubv=_PB_N-1;
#pragma ivdep
#pragma vector always
    for (t7=lbv;t7<=ubv;t7++) {
      y[t7] = w;;
    }
  }
  if (_PB_N == 1) {
    w = b[0];;
    y[0] = w;;
  }
  if (_PB_N >= 2) {
    w = y[0];;
    x[0] = w / A[0][0];;
    for (t6=1;t6<=_PB_N-1;t6++) {
      w = y[t6];;
    }
    for (t4=0;t4<=_PB_N-2;t4++) {
      for (t5=t4+1;t5<=_PB_N-1;t5++) {
        w -= A[t4][t5] * x[t5];;
      }
    }
  }
  if (_PB_N == 1) {
    w = y[0];;
    x[0] = w / A[0][0];;
  }
  lbv=1;
  ubv=_PB_N-1;
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    x[t3] = w / A[t3][t3];;
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
// CHECK:   func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:      %c0 = constant 0 : index
// CHECK-NEXT:      %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      %1 = alloca() : memref<1xf64>
// CHECK-NEXT:      %2 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %3 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %4 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %5 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg6 = 0 to #map0(%arg5) {
// CHECK-NEXT:          %10 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:          affine.for %arg7 = 0 to #map0(%arg6) {
// CHECK-NEXT:            %13 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
// CHECK-NEXT:            %14 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:            %15 = mulf %13, %14 : f64
// CHECK-NEXT:            %16 = subf %2, %15 : f64
// CHECK-NEXT:            affine.store %16, %1[0] : memref<1xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          %11 = affine.load %arg1[%arg6, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %12 = divf %3, %11 : f64
// CHECK-NEXT:          affine.store %12, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg6 = #map0(%arg5) to %0 {
// CHECK-NEXT:          %10 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:          affine.for %arg7 = 0 to #map0(%arg5) {
// CHECK-NEXT:            %11 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
// CHECK-NEXT:            %12 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:            %13 = mulf %11, %12 : f64
// CHECK-NEXT:            %14 = subf %4, %13 : f64
// CHECK-NEXT:            affine.store %14, %1[0] : memref<1xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.store %5, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %7 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        %10 = affine.load %arg2[%arg5] : memref<2000xf64>
// CHECK-NEXT:        affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:        affine.for %arg6 = 0 to #map0(%arg5) {
// CHECK-NEXT:          %11 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %12 = affine.load %arg4[%arg6] : memref<2000xf64>
// CHECK-NEXT:          %13 = mulf %11, %12 : f64
// CHECK-NEXT:          %14 = subf %6, %13 : f64
// CHECK-NEXT:          affine.store %14, %1[0] : memref<1xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %7, %arg4[%arg5] : memref<2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %8 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %9 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        %10 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<2000xf64>
// CHECK-NEXT:        affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:        affine.for %arg6 = #map1(%arg5)[%0] to %0 {
// CHECK-NEXT:          %13 = affine.load %arg1[-%arg5 + symbol(%0) - 1, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %14 = affine.load %arg3[%arg6] : memref<2000xf64>
// CHECK-NEXT:          %15 = mulf %13, %14 : f64
// CHECK-NEXT:          %16 = subf %8, %15 : f64
// CHECK-NEXT:          affine.store %16, %1[0] : memref<1xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %11 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<2000x2000xf64>
// CHECK-NEXT:        %12 = divf %9, %11 : f64
// CHECK-NEXT:        affine.store %12, %arg3[-%arg5 + symbol(%0) - 1] : memref<2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
