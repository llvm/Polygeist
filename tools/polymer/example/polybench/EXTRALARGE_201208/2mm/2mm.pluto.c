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
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

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
  int t1, t2, t3, t4, t5, t6, t7, t8, t9;
 register int lbv, ubv;
/* Start of CLooG code */
if (_PB_NI >= 1) {
  for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
    if ((_PB_NJ >= 0) && (_PB_NL >= 0)) {
      for (t3=0;t3<=floord(_PB_NJ+_PB_NL-1,32);t3++) {
        if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NL-31,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=_PB_NL-1;
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
            lbv=_PB_NL;
            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
          }
        }
        if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-32,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
          }
        }
        if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NJ-31,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=_PB_NJ-1;
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
            lbv=_PB_NJ;
            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
            }
          }
        }
        if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-32,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
          }
        }
        if ((_PB_NJ == _PB_NL) && (t3 <= floord(_PB_NJ-1,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
          }
        }
        if ((t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NL,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              tmp[t4][t5] = SCALAR_VAL(0.0);;
            }
          }
        }
        if ((t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NJ,32))) {
          for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
            lbv=32*t3;
            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
            for (t5=lbv;t5<=ubv;t5++) {
              D[t4][t5] *= beta;;
            }
          }
        }
      }
    }
    if (_PB_NL <= -1) {
      for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
          lbv=32*t3;
          ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
          for (t5=lbv;t5<=ubv;t5++) {
            tmp[t4][t5] = SCALAR_VAL(0.0);;
          }
        }
      }
    }
    if (_PB_NJ <= -1) {
      for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
          lbv=32*t3;
          ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
          for (t5=lbv;t5<=ubv;t5++) {
            D[t4][t5] *= beta;;
          }
        }
      }
    }
  }
  if (_PB_NJ >= 1) {
    for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
        if (_PB_NK >= 1) {
          for (t5=0;t5<=floord(_PB_NK-1,32);t5++) {
            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
              for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                for (t9=32*t5;t9<=min(_PB_NK-1,32*t5+31);t9++) {
                  tmp[t6][t7] += alpha * A[t6][t9] * B[t9][t7];;
                }
              }
            }
          }
        }
        if (_PB_NL >= 1) {
          for (t5=0;t5<=floord(_PB_NL-1,32);t5++) {
            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
              for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                lbv=32*t5;
                ubv=min(_PB_NL-1,32*t5+31);
#pragma ivdep
#pragma vector always
                for (t9=lbv;t9<=ubv;t9++) {
                  D[t6][t9] += tmp[t6][t7] * C[t7][t9];;
                }
              }
            }
          }
        }
      }
    }
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
	      alpha, beta,
	      POLYBENCH_ARRAY(tmp),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}

// CHECK:   func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<800x900xf64>, %arg7: memref<800x1100xf64>, %arg8: memref<1100x900xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<800x1200xf64>) {
// CHECK-NEXT:  %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:  %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %2 = index_cast %arg2 : i32 to index
// CHECK-NEXT:  affine.for %arg11 = 0 to %0 {
// CHECK-NEXT:    affine.for %arg12 = 0 to %1 {
// CHECK-NEXT:      affine.store %cst, %arg6[%arg11, %arg12] : memref<800x900xf64>
// CHECK-NEXT:      %4 = affine.load %arg6[%arg11, %arg12] : memref<800x900xf64>
// CHECK-NEXT:      affine.for %arg13 = 0 to %2 {
// CHECK-NEXT:        %5 = affine.load %arg7[%arg11, %arg13] : memref<800x1100xf64>
// CHECK-NEXT:        %6 = mulf %arg4, %5 : f64
// CHECK-NEXT:        %7 = affine.load %arg8[%arg13, %arg12] : memref<1100x900xf64>
// CHECK-NEXT:        %8 = mulf %6, %7 : f64
// CHECK-NEXT:        %9 = addf %4, %8 : f64
// CHECK-NEXT:        affine.store %9, %arg6[%arg11, %arg12] : memref<800x900xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = index_cast %arg3 : i32 to index
// CHECK-NEXT:  affine.for %arg11 = 0 to %0 {
// CHECK-NEXT:    affine.for %arg12 = 0 to %3 {
// CHECK-NEXT:      %4 = affine.load %arg10[%arg11, %arg12] : memref<800x1200xf64>
// CHECK-NEXT:      %5 = mulf %4, %arg5 : f64
// CHECK-NEXT:      affine.store %5, %arg10[%arg11, %arg12] : memref<800x1200xf64>
// CHECK-NEXT:      %6 = affine.load %arg10[%arg11, %arg12] : memref<800x1200xf64>
// CHECK-NEXT:      affine.for %arg13 = 0 to %1 {
// CHECK-NEXT:      %7 = affine.load %arg6[%arg11, %arg13] : memref<800x900xf64>
// CHECK-NEXT:      %8 = affine.load %arg9[%arg13, %arg12] : memref<900x1200xf64>
// CHECK-NEXT:      %9 = mulf %7, %8 : f64
// CHECK-NEXT:      %10 = addf %6, %9 : f64
// CHECK-NEXT:      affine.store %10, %arg10[%arg11, %arg12] : memref<800x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}
