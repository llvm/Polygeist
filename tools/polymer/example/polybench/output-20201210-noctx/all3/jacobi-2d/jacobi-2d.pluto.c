#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// RUN: mlir-clang %s %stdinclude | FileCheck %s
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
/* jacobi-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_2d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
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
  int t1, t2, t3, t4, t5, t6;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_N >= 3) && (_PB_TSTEPS >= 1)) {
  for (t1=0;t1<=floord(_PB_TSTEPS-1,32);t1++) {
    for (t2=2*t1;t2<=min(floord(2*_PB_TSTEPS+_PB_N-3,32),floord(64*t1+_PB_N+61,32));t2++) {
      for (t3=max(ceild(32*t2-_PB_N-28,32),2*t1);t3<=min(min(floord(2*_PB_TSTEPS+_PB_N-3,32),floord(64*t1+_PB_N+61,32)),floord(32*t2+_PB_N+28,32));t3++) {
        if ((t1 <= floord(32*t3-_PB_N+1,64)) && (t2 <= t3-1)) {
          if ((_PB_N+1)%2 == 0) {
            for (t5=max(32*t2,32*t3-_PB_N+3);t5<=32*t2+31;t5++) {
              A[(-32*t3+t5+_PB_N-2)][(_PB_N-2)] = SCALAR_VAL(0.2) * (B[(-32*t3+t5+_PB_N-2)][(_PB_N-2)] + B[(-32*t3+t5+_PB_N-2)][(_PB_N-2)-1] + B[(-32*t3+t5+_PB_N-2)][1+(_PB_N-2)] + B[1+(-32*t3+t5+_PB_N-2)][(_PB_N-2)] + B[(-32*t3+t5+_PB_N-2)-1][(_PB_N-2)]);;
            }
          }
        }
        if ((t1 <= floord(32*t2-_PB_N+1,64)) && (t2 >= t3)) {
          if ((_PB_N+1)%2 == 0) {
            for (t6=max(32*t3,32*t2-_PB_N+3);t6<=min(32*t2,32*t3+31);t6++) {
              A[(_PB_N-2)][(-32*t2+t6+_PB_N-2)] = SCALAR_VAL(0.2) * (B[(_PB_N-2)][(-32*t2+t6+_PB_N-2)] + B[(_PB_N-2)][(-32*t2+t6+_PB_N-2)-1] + B[(_PB_N-2)][1+(-32*t2+t6+_PB_N-2)] + B[1+(_PB_N-2)][(-32*t2+t6+_PB_N-2)] + B[(_PB_N-2)-1][(-32*t2+t6+_PB_N-2)]);;
            }
          }
        }
        if ((_PB_N == 3) && (t2 == t3)) {
          for (t4=16*t2;t4<=min(min(_PB_TSTEPS-1,32*t1+31),16*t2+14);t4++) {
            B[1][1] = SCALAR_VAL(0.2) * (A[1][1] + A[1][1 -1] + A[1][1+1] + A[1+1][1] + A[1 -1][1]);;
            A[1][1] = SCALAR_VAL(0.2) * (B[1][1] + B[1][1 -1] + B[1][1+1] + B[1+1][1] + B[1 -1][1]);;
          }
        }
        if (t2 == t3) {
          for (t4=max(ceild(32*t2-_PB_N+2,2),32*t1);t4<=min(min(min(floord(32*t2-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31),16*t2-1);t4++) {
            for (t5=32*t2;t5<=2*t4+_PB_N-2;t5++) {
              for (t6=32*t2;t6<=2*t4+_PB_N-2;t6++) {
                B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
                A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
              }
              A[(-2*t4+t5-1)][(_PB_N-2)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)][(_PB_N-2)-1] + B[(-2*t4+t5-1)][1+(_PB_N-2)] + B[1+(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)-1][(_PB_N-2)]);;
            }
            for (t6=32*t2;t6<=2*t4+_PB_N-1;t6++) {
              A[(_PB_N-2)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)][(-2*t4+t6-1)-1] + B[(_PB_N-2)][1+(-2*t4+t6-1)] + B[1+(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)-1][(-2*t4+t6-1)]);;
            }
          }
        }
        for (t4=max(max(ceild(32*t2-_PB_N+33,2),ceild(32*t3-_PB_N+2,2)),32*t1);t4<=min(min(min(floord(32*t3-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31),16*t2-1);t4++) {
          for (t5=32*t2;t5<=32*t2+31;t5++) {
            for (t6=32*t3;t6<=2*t4+_PB_N-2;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
            A[(-2*t4+t5-1)][(_PB_N-2)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)][(_PB_N-2)-1] + B[(-2*t4+t5-1)][1+(_PB_N-2)] + B[1+(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)-1][(_PB_N-2)]);;
          }
        }
        for (t4=max(max(ceild(32*t2-_PB_N+2,2),ceild(32*t3-_PB_N+33,2)),32*t1);t4<=min(min(min(floord(32*t2-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31),16*t3-1);t4++) {
          for (t5=32*t2;t5<=2*t4+_PB_N-2;t5++) {
            for (t6=32*t3;t6<=32*t3+31;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
          }
          for (t6=32*t3;t6<=32*t3+31;t6++) {
            A[(_PB_N-2)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)][(-2*t4+t6-1)-1] + B[(_PB_N-2)][1+(-2*t4+t6-1)] + B[1+(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)-1][(-2*t4+t6-1)]);;
          }
        }
        for (t4=max(max(ceild(32*t2-_PB_N+33,2),ceild(32*t3-_PB_N+33,2)),32*t1);t4<=min(min(min(_PB_TSTEPS-1,32*t1+31),16*t2-1),16*t3-1);t4++) {
          for (t5=32*t2;t5<=32*t2+31;t5++) {
            for (t6=32*t3;t6<=32*t3+31;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
          }
        }
        if ((_PB_N >= 4) && (t2 == t3)) {
          for (t4=16*t2;t4<=min(min(floord(32*t2-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31);t4++) {
            for (t6=2*t4+1;t6<=2*t4+_PB_N-2;t6++) {
              B[1][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[1][(-2*t4+t6)] + A[1][(-2*t4+t6)-1] + A[1][1+(-2*t4+t6)] + A[1+1][(-2*t4+t6)] + A[1 -1][(-2*t4+t6)]);;
            }
            for (t5=2*t4+2;t5<=2*t4+_PB_N-2;t5++) {
              B[(-2*t4+t5)][1] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][1] + A[(-2*t4+t5)][1 -1] + A[(-2*t4+t5)][1+1] + A[1+(-2*t4+t5)][1] + A[(-2*t4+t5)-1][1]);;
              for (t6=2*t4+2;t6<=2*t4+_PB_N-2;t6++) {
                B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
                A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
              }
              A[(-2*t4+t5-1)][(_PB_N-2)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)][(_PB_N-2)-1] + B[(-2*t4+t5-1)][1+(_PB_N-2)] + B[1+(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)-1][(_PB_N-2)]);;
            }
            for (t6=2*t4+2;t6<=2*t4+_PB_N-1;t6++) {
              A[(_PB_N-2)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)][(-2*t4+t6-1)-1] + B[(_PB_N-2)][1+(-2*t4+t6-1)] + B[1+(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)-1][(-2*t4+t6-1)]);;
            }
          }
        }
        if (t2 == t3) {
          for (t4=max(ceild(32*t2-_PB_N+33,2),16*t2);t4<=min(min(_PB_TSTEPS-1,32*t1+31),16*t2+14);t4++) {
            for (t6=2*t4+1;t6<=32*t2+31;t6++) {
              B[1][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[1][(-2*t4+t6)] + A[1][(-2*t4+t6)-1] + A[1][1+(-2*t4+t6)] + A[1+1][(-2*t4+t6)] + A[1 -1][(-2*t4+t6)]);;
            }
            for (t5=2*t4+2;t5<=32*t2+31;t5++) {
              B[(-2*t4+t5)][1] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][1] + A[(-2*t4+t5)][1 -1] + A[(-2*t4+t5)][1+1] + A[1+(-2*t4+t5)][1] + A[(-2*t4+t5)-1][1]);;
              for (t6=2*t4+2;t6<=32*t2+31;t6++) {
                B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
                A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
              }
            }
          }
        }
        for (t4=max(ceild(32*t3-_PB_N+2,2),16*t2);t4<=min(min(min(min(floord(32*t3-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31),16*t2+14),16*t3-1);t4++) {
          for (t6=32*t3;t6<=2*t4+_PB_N-2;t6++) {
            B[1][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[1][(-2*t4+t6)] + A[1][(-2*t4+t6)-1] + A[1][1+(-2*t4+t6)] + A[1+1][(-2*t4+t6)] + A[1 -1][(-2*t4+t6)]);;
          }
          for (t5=2*t4+2;t5<=32*t2+31;t5++) {
            for (t6=32*t3;t6<=2*t4+_PB_N-2;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
            A[(-2*t4+t5-1)][(_PB_N-2)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)][(_PB_N-2)-1] + B[(-2*t4+t5-1)][1+(_PB_N-2)] + B[1+(-2*t4+t5-1)][(_PB_N-2)] + B[(-2*t4+t5-1)-1][(_PB_N-2)]);;
          }
        }
        for (t4=max(ceild(32*t3-_PB_N+33,2),16*t2);t4<=min(min(min(_PB_TSTEPS-1,32*t1+31),16*t2+14),16*t3-1);t4++) {
          for (t6=32*t3;t6<=32*t3+31;t6++) {
            B[1][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[1][(-2*t4+t6)] + A[1][(-2*t4+t6)-1] + A[1][1+(-2*t4+t6)] + A[1+1][(-2*t4+t6)] + A[1 -1][(-2*t4+t6)]);;
          }
          for (t5=2*t4+2;t5<=32*t2+31;t5++) {
            for (t6=32*t3;t6<=32*t3+31;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
          }
        }
        for (t4=max(ceild(32*t2-_PB_N+2,2),16*t3);t4<=min(min(min(min(floord(32*t2-_PB_N+32,2),_PB_TSTEPS-1),32*t1+31),16*t2-1),16*t3+14);t4++) {
          for (t5=32*t2;t5<=2*t4+_PB_N-2;t5++) {
            B[(-2*t4+t5)][1] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][1] + A[(-2*t4+t5)][1 -1] + A[(-2*t4+t5)][1+1] + A[1+(-2*t4+t5)][1] + A[(-2*t4+t5)-1][1]);;
            for (t6=2*t4+2;t6<=32*t3+31;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
          }
          for (t6=2*t4+2;t6<=32*t3+31;t6++) {
            A[(_PB_N-2)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)][(-2*t4+t6-1)-1] + B[(_PB_N-2)][1+(-2*t4+t6-1)] + B[1+(_PB_N-2)][(-2*t4+t6-1)] + B[(_PB_N-2)-1][(-2*t4+t6-1)]);;
          }
        }
        for (t4=max(ceild(32*t2-_PB_N+33,2),16*t3);t4<=min(min(min(_PB_TSTEPS-1,32*t1+31),16*t2-1),16*t3+14);t4++) {
          for (t5=32*t2;t5<=32*t2+31;t5++) {
            B[(-2*t4+t5)][1] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][1] + A[(-2*t4+t5)][1 -1] + A[(-2*t4+t5)][1+1] + A[1+(-2*t4+t5)][1] + A[(-2*t4+t5)-1][1]);;
            for (t6=2*t4+2;t6<=32*t3+31;t6++) {
              B[(-2*t4+t5)][(-2*t4+t6)] = SCALAR_VAL(0.2) * (A[(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)][(-2*t4+t6)-1] + A[(-2*t4+t5)][1+(-2*t4+t6)] + A[1+(-2*t4+t5)][(-2*t4+t6)] + A[(-2*t4+t5)-1][(-2*t4+t6)]);;
              A[(-2*t4+t5-1)][(-2*t4+t6-1)] = SCALAR_VAL(0.2) * (B[(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)][(-2*t4+t6-1)-1] + B[(-2*t4+t5-1)][1+(-2*t4+t6-1)] + B[1+(-2*t4+t5-1)][(-2*t4+t6-1)] + B[(-2*t4+t5-1)-1][(-2*t4+t6-1)]);;
            }
          }
        }
        if ((t1 >= ceild(t2-1,2)) && (t2 <= min(floord(_PB_TSTEPS-16,16),t3-1))) {
          for (t6=32*t3;t6<=min(32*t3+31,32*t2+_PB_N+28);t6++) {
            B[1][(-32*t2+t6-30)] = SCALAR_VAL(0.2) * (A[1][(-32*t2+t6-30)] + A[1][(-32*t2+t6-30)-1] + A[1][1+(-32*t2+t6-30)] + A[1+1][(-32*t2+t6-30)] + A[1 -1][(-32*t2+t6-30)]);;
          }
        }
        if ((t1 >= ceild(t3-1,2)) && (t2 >= t3) && (t3 <= floord(_PB_TSTEPS-16,16))) {
          for (t5=max(32*t2,32*t3+31);t5<=min(32*t2+31,32*t3+_PB_N+28);t5++) {
            B[(-32*t3+t5-30)][1] = SCALAR_VAL(0.2) * (A[(-32*t3+t5-30)][1] + A[(-32*t3+t5-30)][1 -1] + A[(-32*t3+t5-30)][1+1] + A[1+(-32*t3+t5-30)][1] + A[(-32*t3+t5-30)-1][1]);;
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
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>
// CHECK:   func private @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<1300x1300xf64>, %arg3: memref<1300x1300xf64>) {
// CHECK-NEXT:      %cst = constant 2.000000e-01 : f64
// CHECK-NEXT:      %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:      %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      affine.for %arg4 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:          affine.for %arg6 = 1 to #map()[%0] {
// CHECK-NEXT:            %2 = affine.load %arg2[%arg5, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %3 = affine.load %arg2[%arg5, %arg6 - 1] : memref<1300x1300xf64>
// CHECK-NEXT:            %4 = addf %2, %3 : f64
// CHECK-NEXT:            %5 = affine.load %arg2[%arg5, %arg6 + 1] : memref<1300x1300xf64>
// CHECK-NEXT:            %6 = addf %4, %5 : f64
// CHECK-NEXT:            %7 = affine.load %arg2[%arg5 + 1, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %8 = addf %6, %7 : f64
// CHECK-NEXT:            %9 = affine.load %arg2[%arg5 - 1, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %10 = addf %8, %9 : f64
// CHECK-NEXT:            %11 = mulf %cst, %10 : f64
// CHECK-NEXT:            affine.store %11, %arg3[%arg5, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:          affine.for %arg6 = 1 to #map()[%0] {
// CHECK-NEXT:            %2 = affine.load %arg3[%arg5, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %3 = affine.load %arg3[%arg5, %arg6 - 1] : memref<1300x1300xf64>
// CHECK-NEXT:            %4 = addf %2, %3 : f64
// CHECK-NEXT:            %5 = affine.load %arg3[%arg5, %arg6 + 1] : memref<1300x1300xf64>
// CHECK-NEXT:            %6 = addf %4, %5 : f64
// CHECK-NEXT:            %7 = affine.load %arg3[%arg5 + 1, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %8 = addf %6, %7 : f64
// CHECK-NEXT:            %9 = affine.load %arg3[%arg5 - 1, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:            %10 = addf %8, %9 : f64
// CHECK-NEXT:            %11 = mulf %cst, %10 : f64
// CHECK-NEXT:            affine.store %11, %arg2[%arg5, %arg6] : memref<1300x1300xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
