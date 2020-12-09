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
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
 DATA_TYPE z[N];
 DATA_TYPE alpha;
 DATA_TYPE beta;
 DATA_TYPE sum;

 int i,k;

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
alpha = -r[0];;
beta = SCALAR_VAL(1.0);;
y[0] = -r[0];;
if (_PB_N >= 2) {
  sum = SCALAR_VAL(0.0);;
  sum += r[1 -0 -1]*y[0];;
  beta = (1-alpha*alpha)*beta;;
  alpha = - (r[1] + sum)/beta;;
  y[1] = alpha;;
  z[0] = y[0] + alpha*y[1 -0 -1];;
  y[0] = z[0];;
}
for (t2=2;t2<=_PB_N-1;t2++) {
  sum = SCALAR_VAL(0.0);;
  sum += r[t2-0 -1]*y[0];;
  beta = (1-alpha*alpha)*beta;;
  for (t3=1;t3<=t2-1;t3++) {
    sum += r[t2-t3-1]*y[t3];;
  }
  alpha = - (r[t2] + sum)/beta;;
  y[t2] = alpha;;
  for (t3=t2;t3<=2*t2-1;t3++) {
    z[(-t2+t3)] = y[(-t2+t3)] + alpha*y[t2-(-t2+t3)-1];;
  }
  for (t3=2*t2;t3<=3*t2-1;t3++) {
    y[(-2*t2+t3)] = z[(-2*t2+t3)];;
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK:  func @kernel_durbin(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
// CHECK-NEXT:      %c0 = constant 0 : index
// CHECK-NEXT:      %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:      %c1_i32 = constant 1 : i32
// CHECK-NEXT:      %cst_0 = constant 0.000000e+00 : f64
// CHECK-NEXT:      %0 = alloca() : memref<2000xf64>
// CHECK-NEXT:      %1 = alloca() : memref<1xf64>
// CHECK-NEXT:      %2 = alloca() : memref<1xf64>
// CHECK-NEXT:      %3 = alloca() : memref<1xf64>
// CHECK-NEXT:      %4 = load %arg1[%c0] : memref<2000xf64>
// CHECK-NEXT:      %5 = negf %4 : f64
// CHECK-NEXT:      store %5, %arg2[%c0] : memref<2000xf64>
// CHECK-NEXT:      store %cst, %2[%c0] : memref<1xf64>
// CHECK-NEXT:      %6 = load %arg1[%c0] : memref<2000xf64>
// CHECK-NEXT:      %7 = negf %6 : f64
// CHECK-NEXT:      store %7, %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %8 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      %9 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:      %10 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %11 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %12 = mulf %10, %11 : f64
// CHECK-NEXT:      %13 = subf %9, %12 : f64
// CHECK-NEXT:      %14 = load %2[%c0] : memref<1xf64>
// CHECK-NEXT:      %15 = mulf %13, %14 : f64
// CHECK-NEXT:      store %15, %2[%c0] : memref<1xf64>
// CHECK-NEXT:      store %cst_0, %3[%c0] : memref<1xf64>
// CHECK-NEXT:      %16 = load %3[%c0] : memref<1xf64>
// CHECK-NEXT:      %17 = load %3[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg3 = 1 to %8 {
// CHECK-NEXT:        affine.for %arg4 = 0 to #map(%arg3) {
// CHECK-NEXT:          %22 = affine.load %arg1[%arg3 - %arg4 - 1] : memref<2000xf64>
// CHECK-NEXT:          %23 = affine.load %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:          %24 = mulf %22, %23 : f64
// CHECK-NEXT:          %25 = addf %16, %24 : f64
// CHECK-NEXT:          affine.store %25, %3[0] : memref<1xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %18 = affine.load %arg1[%arg3] : memref<2000xf64>
// CHECK-NEXT:        %19 = addf %18, %17 : f64
// CHECK-NEXT:        %20 = negf %19 : f64
// CHECK-NEXT:        %21 = divf %20, %15 : f64
// CHECK-NEXT:        affine.store %21, %1[0] : memref<1xf64>
// CHECK-NEXT:        affine.for %arg4 = 0 to #map(%arg3) {
// CHECK-NEXT:          %22 = affine.load %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:          %23 = affine.load %arg2[%arg3 - %arg4 - 1] : memref<2000xf64>
// CHECK-NEXT:          %24 = mulf %21, %23 : f64
// CHECK-NEXT:          %25 = addf %22, %24 : f64
// CHECK-NEXT:          affine.store %25, %0[%arg4] : memref<2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg4 = 0 to #map(%arg3) {
// CHECK-NEXT:          %22 = affine.load %0[%arg4] : memref<2000xf64>
// CHECK-NEXT:          affine.store %22, %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %21, %arg2[%arg3] : memref<2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
