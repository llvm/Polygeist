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
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;

//BLAS PARAMS
//SIDE = 'L'
//UPLO = 'L'
// =>  Form  C := alpha*A*B + beta*C
// A is MxM
// B is MxN
// C is MxN
//note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
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
  int t1, t2, t3, t4;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_M >= 1) && (_PB_N >= 1)) {
  for (t2=0;t2<=_PB_N-1;t2++) {
    temp2 = 0;;
    C[0][t2] = beta * C[0][t2] + alpha*B[0][t2] * A[0][0] + alpha * temp2;;
  }
  if (_PB_M >= 2) {
    for (t2=0;t2<=_PB_N-1;t2++) {
      C[0][t2] += alpha*B[1][t2] * A[1][0];;
      temp2 = 0;;
      temp2 += B[0][t2] * A[1][0];;
      C[1][t2] = beta * C[1][t2] + alpha*B[1][t2] * A[1][1] + alpha * temp2;;
    }
  }
  for (t1=2;t1<=_PB_M-1;t1++) {
    for (t2=0;t2<=_PB_N-1;t2++) {
      C[0][t2] += alpha*B[t1][t2] * A[t1][0];;
      temp2 = 0;;
      temp2 += B[0][t2] * A[t1][0];;
      for (t3=1;t3<=t1-1;t3++) {
        C[t3][t2] += alpha*B[t1][t2] * A[t1][t3];;
        temp2 += B[t3][t2] * A[t1][t3];;
      }
      C[t1][t2] = beta * C[t1][t2] + alpha*B[t1][t2] * A[t1][t1] + alpha * temp2;;
    }
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     %2 = index_cast %arg1 : i32 to index
// CHECK-NEXT:     %3 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %3, %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %4 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %5 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %6 = mulf %arg2, %5 : f64
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       %7 = affine.load %arg5[%arg7, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:       affine.for %arg8 = 0 to %2 {
// CHECK-NEXT:         %8 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:         %9 = mulf %arg2, %8 : f64
// CHECK-NEXT:         affine.for %arg9 = 0 to #map(%arg7) {
// CHECK-NEXT:           %17 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
// CHECK-NEXT:           %18 = mulf %9, %17 : f64
// CHECK-NEXT:           %19 = affine.load %arg4[%arg9, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:           %20 = addf %19, %18 : f64
// CHECK-NEXT:           affine.store %20, %arg4[%arg9, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:           %21 = affine.load %arg6[%arg9, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:           %22 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
// CHECK-NEXT:           %23 = mulf %21, %22 : f64
// CHECK-NEXT:           %24 = addf %4, %23 : f64
// CHECK-NEXT:           affine.store %24, %0[0] : memref<1xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         %10 = affine.load %arg4[%arg7, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:         %11 = mulf %arg3, %10 : f64
// CHECK-NEXT:         %12 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:         %13 = mulf %arg2, %12 : f64
// CHECK-NEXT:         %14 = mulf %13, %7 : f64
// CHECK-NEXT:         %15 = addf %11, %14 : f64
// CHECK-NEXT:         %16 = addf %15, %6 : f64
// CHECK-NEXT:         affine.store %16, %arg4[%arg7, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// EXEC: {{[0-9]\.[0-9]+}}
