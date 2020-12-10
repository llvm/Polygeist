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
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

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
  int t1, t2, t3;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NY >= 1) && (_PB_TMAX >= 1)) {
  if ((_PB_NX >= 2) && (_PB_NY >= 2)) {
    for (t1=0;t1<=_PB_TMAX-1;t1++) {
      ey[0][0] = _fict_[t1];;
      for (t3=t1+1;t3<=t1+_PB_NY-1;t3++) {
        ey[0][(-t1+t3)] = _fict_[t1];;
        ex[0][(-t1+t3)] = ex[0][(-t1+t3)] - SCALAR_VAL(0.5)*(hz[0][(-t1+t3)]-hz[0][(-t1+t3)-1]);;
      }
      for (t2=t1+1;t2<=t1+_PB_NX-1;t2++) {
        ey[(-t1+t2)][0] = ey[(-t1+t2)][0] - SCALAR_VAL(0.5)*(hz[(-t1+t2)][0]-hz[(-t1+t2)-1][0]);;
        for (t3=t1+1;t3<=t1+_PB_NY-1;t3++) {
          ey[(-t1+t2)][(-t1+t3)] = ey[(-t1+t2)][(-t1+t3)] - SCALAR_VAL(0.5)*(hz[(-t1+t2)][(-t1+t3)]-hz[(-t1+t2)-1][(-t1+t3)]);;
          ex[(-t1+t2)][(-t1+t3)] = ex[(-t1+t2)][(-t1+t3)] - SCALAR_VAL(0.5)*(hz[(-t1+t2)][(-t1+t3)]-hz[(-t1+t2)][(-t1+t3)-1]);;
          hz[(-t1+t2-1)][(-t1+t3-1)] = hz[(-t1+t2-1)][(-t1+t3-1)] - SCALAR_VAL(0.7)* (ex[(-t1+t2-1)][(-t1+t3-1)+1] - ex[(-t1+t2-1)][(-t1+t3-1)] + ey[(-t1+t2-1)+1][(-t1+t3-1)] - ey[(-t1+t2-1)][(-t1+t3-1)]);;
        }
      }
    }
  }
  if ((_PB_NX >= 2) && (_PB_NY == 1)) {
    for (t1=0;t1<=_PB_TMAX-1;t1++) {
      ey[0][0] = _fict_[t1];;
      for (t2=t1+1;t2<=t1+_PB_NX-1;t2++) {
        ey[(-t1+t2)][0] = ey[(-t1+t2)][0] - SCALAR_VAL(0.5)*(hz[(-t1+t2)][0]-hz[(-t1+t2)-1][0]);;
      }
    }
  }
  if ((_PB_NX == 1) && (_PB_NY >= 2)) {
    for (t1=0;t1<=_PB_TMAX-1;t1++) {
      ey[0][0] = _fict_[t1];;
      for (t3=t1+1;t3<=t1+_PB_NY-1;t3++) {
        ey[0][(-t1+t3)] = _fict_[t1];;
        ex[0][(-t1+t3)] = ex[0][(-t1+t3)] - SCALAR_VAL(0.5)*(hz[0][(-t1+t3)]-hz[0][(-t1+t3)-1]);;
      }
    }
  }
  if ((_PB_NX <= 0) && (_PB_NY >= 2)) {
    for (t1=0;t1<=_PB_TMAX-1;t1++) {
      for (t3=t1;t3<=t1+_PB_NY-1;t3++) {
        ey[0][(-t1+t3)] = _fict_[t1];;
      }
    }
  }
  if ((_PB_NX <= 1) && (_PB_NY == 1)) {
    for (t1=0;t1<=_PB_TMAX-1;t1++) {
      ey[0][0] = _fict_[t1];;
    }
  }
}
/* End of CLooG code */
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}

// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>

// CHECK:    func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
// CHECK-NEXT:    %cst = constant 5.000000e-01 : f64
// CHECK-NEXT:    %cst_0 = constant 0.69999999999999996 : f64
// CHECK-NEXT:    %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    %1 = index_cast %arg2 : i32 to index
// CHECK-NEXT:    %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg7 = 0 to %2 {
// CHECK-NEXT:      %3 = affine.load %arg6[%arg7] : memref<500xf64>
// CHECK-NEXT:      affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:        affine.store %3, %arg4[0, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 1 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 0 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8 - 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 1 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8, %arg9 - 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to #map()[%0] {
// CHECK-NEXT:        affine.for %arg9 = 0 to #map()[%1] {
// CHECK-NEXT:          %4 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg3[%arg8, %arg9 + 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = affine.load %arg4[%arg8 + 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %9 = addf %7, %8 : f64
// CHECK-NEXT:          %10 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %11 = subf %9, %10 : f64
// CHECK-NEXT:          %12 = mulf %cst_0, %11 : f64
// CHECK-NEXT:          %13 = subf %4, %12 : f64
// CHECK-NEXT:          affine.store %13, %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


// EXEC: {{[0-9]\.[0-9]+}}
