#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
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
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

  int t1, t2, t3, t4;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if ((_PB_N >= 3) && (_PB_TSTEPS >= 1)) {
  for (t1=-1;t1<=floord(_PB_TSTEPS-1,8);t1++) {
    for (t2=ceild(t1,2);t2<=min(floord(2*_PB_TSTEPS+_PB_N-3,32),floord(16*t1+_PB_N+13,32));t2++) {
      if ((t1 <= floord(32*t2-_PB_N+1,16)) && (t2 >= ceild(_PB_N-1,32))) {
        if ((_PB_N+1)%2 == 0) {
          A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
        }
      }
      if (t1 == 2*t2) {
        if (t1%2 == 0) {
          A[1] = 0.33333 * (B[1 -1] + B[1] + B[1 + 1]);;
        }
      }
      if (t1 == 2*t2) {
        for (t3=8*t1+1;t3<=min(min(floord(16*t1+_PB_N-2,2),_PB_TSTEPS-1),8*t1+7);t3++) {
          if (t1%2 == 0) {
            B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
          }
          for (t4=2*t3+2;t4<=-16*t1+4*t3;t4++) {
            if (t1%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if (t1%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
          for (t4=-16*t1+4*t3+1;t4<=min(-16*t1+4*t3+2,2*t3+_PB_N-1);t4++) {
            if (t1%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
        }
      }
      if ((_PB_N >= 4) && (t1 == 2*t2)) {
        for (t3=ceild(16*t1+_PB_N-1,2);t3<=min(_PB_TSTEPS-1,8*t1+7);t3++) {
          if (t1%2 == 0) {
            B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
          }
          for (t4=2*t3+2;t4<=2*t3+_PB_N-2;t4++) {
            if (t1%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if (t1%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
          if (t1%2 == 0) {
            A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
          }
        }
      }
      if ((_PB_N == 3) && (t1 == 2*t2)) {
        for (t3=8*t1+1;t3<=min(_PB_TSTEPS-1,8*t1+14);t3++) {
          if (t1%2 == 0) {
            B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
          }
          if (t1%2 == 0) {
            A[1] = 0.33333 * (B[1 -1] + B[1] + B[1 + 1]);;
          }
        }
      }
      for (t3=max(max(0,8*t1),16*t1-16*t2+16);t3<=min(min(floord(32*t1-32*t2+_PB_N-2,2),_PB_TSTEPS-1),8*t1+7);t3++) {
        for (t4=32*t2;t4<=-32*t1+32*t2+4*t3;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
          A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
        }
        for (t4=-32*t1+32*t2+4*t3+1;t4<=min(2*t3+_PB_N-1,-32*t1+32*t2+4*t3+2);t4++) {
          A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
        }
      }
      for (t3=max(max(max(0,ceild(32*t2-_PB_N+2,2)),ceild(32*t1-32*t2+_PB_N-1,2)),16*t1-16*t2+16);t3<=min(_PB_TSTEPS-1,8*t1+7);t3++) {
        for (t4=32*t2;t4<=2*t3+_PB_N-2;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
          A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
        }
        A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
      }
      if (t1 == 2*t2-1) {
        for (t3=max(0,8*t1);t3<=min(min(floord(16*t1+_PB_N-18,2),_PB_TSTEPS-1),8*t1+7);t3++) {
          for (t4=16*t1+16;t4<=-16*t1+4*t3+16;t4++) {
            if ((t1+1)%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if ((t1+1)%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
          for (t4=-16*t1+4*t3+17;t4<=min(-16*t1+4*t3+18,2*t3+_PB_N-1);t4++) {
            if ((t1+1)%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
        }
      }
      if (t1 == 2*t2-1) {
        for (t3=max(max(0,ceild(16*t1-_PB_N+18,2)),ceild(16*t1+_PB_N-17,2));t3<=min(_PB_TSTEPS-1,8*t1+7);t3++) {
          for (t4=16*t1+16;t4<=2*t3+_PB_N-2;t4++) {
            if ((t1+1)%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if ((t1+1)%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
          if ((t1+1)%2 == 0) {
            A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
          }
        }
      }
      if ((_PB_N >= 5) && (_PB_N <= 32) && (t1 == 2*t2-1) && (t1 <= floord(_PB_TSTEPS-9,8))) {
        if ((t1+1)%2 == 0) {
          B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
        }
        if ((t1+1)%2 == 0) {
          B[2] = 0.33333 * (A[2 -1] + A[2] + A[2 + 1]);;
        }
        for (t4=16*t1+19;t4<=16*t1+_PB_N+14;t4++) {
          if ((t1+1)%2 == 0) {
            B[(-16*t1+t4-16)] = 0.33333 * (A[(-16*t1+t4-16)-1] + A[(-16*t1+t4-16)] + A[(-16*t1+t4-16) + 1]);;
          }
          if ((t1+1)%2 == 0) {
            A[(-16*t1+t4-17)] = 0.33333 * (B[(-16*t1+t4-17)-1] + B[(-16*t1+t4-17)] + B[(-16*t1+t4-17) + 1]);;
          }
        }
        if ((t1+1)%2 == 0) {
          A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
        }
      }
      if ((_PB_N >= 33) && (t1 == 2*t2-1) && (t1 <= floord(_PB_TSTEPS-9,8))) {
        if ((t1+1)%2 == 0) {
          B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
        }
        if ((t1+1)%2 == 0) {
          B[2] = 0.33333 * (A[2 -1] + A[2] + A[2 + 1]);;
        }
        for (t4=16*t1+19;t4<=16*t1+47;t4++) {
          if ((t1+1)%2 == 0) {
            B[(-16*t1+t4-16)] = 0.33333 * (A[(-16*t1+t4-16)-1] + A[(-16*t1+t4-16)] + A[(-16*t1+t4-16) + 1]);;
          }
          if ((t1+1)%2 == 0) {
            A[(-16*t1+t4-17)] = 0.33333 * (B[(-16*t1+t4-17)-1] + B[(-16*t1+t4-17)] + B[(-16*t1+t4-17) + 1]);;
          }
        }
      }
      if ((_PB_N == 4) && (t1 == 2*t2-1) && (t1 <= floord(_PB_TSTEPS-9,8))) {
        if ((t1+1)%2 == 0) {
          B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
        }
        if ((t1+1)%2 == 0) {
          B[2] = 0.33333 * (A[2 -1] + A[2] + A[2 + 1]);;
        }
        if ((t1+1)%2 == 0) {
          A[2] = 0.33333 * (B[2 -1] + B[2] + B[2 + 1]);;
        }
      }
      if ((_PB_N >= 4) && (t1 == 2*t2)) {
        for (t3=8*t1+8;t3<=min(floord(16*t1-_PB_N+32,2),_PB_TSTEPS-1);t3++) {
          if (t1%2 == 0) {
            B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
          }
          for (t4=2*t3+2;t4<=2*t3+_PB_N-2;t4++) {
            if (t1%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if (t1%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
          if (t1%2 == 0) {
            A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
          }
        }
      }
      if (t1 == 2*t2) {
        for (t3=max(ceild(16*t1-_PB_N+33,2),8*t1+8);t3<=min(_PB_TSTEPS-1,8*t1+14);t3++) {
          if (t1%2 == 0) {
            B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
          }
          for (t4=2*t3+2;t4<=16*t1+31;t4++) {
            if (t1%2 == 0) {
              B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
            }
            if (t1%2 == 0) {
              A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
            }
          }
        }
      }
      for (t3=max(8*t1+8,16*t1-16*t2+17);t3<=min(min(floord(32*t2-_PB_N+32,2),floord(32*t1-32*t2+_PB_N+27,2)),_PB_TSTEPS-1);t3++) {
        for (t4=-32*t1+32*t2+4*t3-31;t4<=-32*t1+32*t2+4*t3-30;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
        }
        for (t4=-32*t1+32*t2+4*t3-29;t4<=2*t3+_PB_N-2;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
          A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
        }
        A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
      }
      for (t3=max(max(ceild(32*t2-_PB_N+33,2),8*t1+8),16*t1-16*t2+17);t3<=min(_PB_TSTEPS-1,8*t1+15);t3++) {
        for (t4=-32*t1+32*t2+4*t3-31;t4<=-32*t1+32*t2+4*t3-30;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
        }
        for (t4=-32*t1+32*t2+4*t3-29;t4<=32*t2+31;t4++) {
          B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
          A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
        }
      }
      if ((_PB_N >= 6) && (t1 <= min(floord(32*t2-_PB_N+2,16),floord(32*t2+2*_PB_TSTEPS-_PB_N-30,32)))) {
        if (_PB_N%2 == 0) {
          for (t4=32*t1-32*t2+2*_PB_N+25;t4<=32*t1-32*t2+2*_PB_N+26;t4++) {
            B[(-32*t1+32*t2+t4-_PB_N-28)] = 0.33333 * (A[(-32*t1+32*t2+t4-_PB_N-28)-1] + A[(-32*t1+32*t2+t4-_PB_N-28)] + A[(-32*t1+32*t2+t4-_PB_N-28) + 1]);;
          }
          A[(_PB_N-2)] = 0.33333 * (B[(_PB_N-2)-1] + B[(_PB_N-2)] + B[(_PB_N-2) + 1]);;
        }
      }
      if (t1 <= min(floord(32*t2-_PB_N+1,16),floord(32*t2+2*_PB_TSTEPS-_PB_N-31,32))) {
        if ((_PB_N+1)%2 == 0) {
          B[(_PB_N-2)] = 0.33333 * (A[(_PB_N-2)-1] + A[(_PB_N-2)] + A[(_PB_N-2) + 1]);;
        }
      }
      if ((t1 == 2*t2) && (t1 <= floord(_PB_TSTEPS-16,8))) {
        if (t1%2 == 0) {
          B[1] = 0.33333 * (A[1 -1] + A[1] + A[1 + 1]);;
        }
      }
    }
  }
}

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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
// CHECK: func private @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
// CHECK-NEXT:      %cst = constant 3.333300e-01 : f64
// CHECK-NEXT:      %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:      %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      affine.for %arg4 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:          %2 = affine.load %arg2[%arg5 - 1] : memref<2000xf64>
// CHECK-NEXT:          %3 = affine.load %arg2[%arg5] : memref<2000xf64>
// CHECK-NEXT:          %4 = addf %2, %3 : f64
// CHECK-NEXT:          %5 = affine.load %arg2[%arg5 + 1] : memref<2000xf64>
// CHECK-NEXT:          %6 = addf %4, %5 : f64
// CHECK-NEXT:          %7 = mulf %cst, %6 : f64
// CHECK-NEXT:          affine.store %7, %arg3[%arg5] : memref<2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:          %2 = affine.load %arg3[%arg5 - 1] : memref<2000xf64>
// CHECK-NEXT:          %3 = affine.load %arg3[%arg5] : memref<2000xf64>
// CHECK-NEXT:          %4 = addf %2, %3 : f64
// CHECK-NEXT:          %5 = affine.load %arg3[%arg5 + 1] : memref<2000xf64>
// CHECK-NEXT:          %6 = addf %4, %5 : f64
// CHECK-NEXT:          %7 = mulf %cst, %6 : f64
// CHECK-NEXT:          affine.store %7, %arg2[%arg5] : memref<2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
