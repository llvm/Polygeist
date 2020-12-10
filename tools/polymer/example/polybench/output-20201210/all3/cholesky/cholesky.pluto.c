#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// TODO: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 -lm && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 -lm && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
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
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

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
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;


  int t1, t2, t3, t4, t5, t6;
 register int lbv, ubv;
if (_PB_N >= 1) {
  for (t1=0;t1<=floord(_PB_N-1,32);t1++) {
    for (t2=0;t2<=t1;t2++) {
      for (t3=0;t3<=t1;t3++) {
        if ((t1 >= t2+1) && (t2 == t3)) {
          for (t4=32*t1;t4<=min(_PB_N-1,32*t1+31);t4++) {
            A[t4][32*t2] /= A[32*t2][32*t2];;
            for (t5=32*t2+1;t5<=32*t2+31;t5++) {
              for (t6=32*t2;t6<=t5-1;t6++) {
                A[t4][t5] -= A[t4][t6] * A[t5][t6];;
              }
              A[t4][t5] /= A[t5][t5];;
            }
          }
        }
        if (t2 >= t3+1) {
          for (t4=max(32*t1,32*t2+1);t4<=min(_PB_N-1,32*t1+31);t4++) {
            for (t5=32*t2;t5<=min(32*t2+31,t4-1);t5++) {
              for (t6=32*t3;t6<=32*t3+31;t6++) {
                A[t4][t5] -= A[t4][t6] * A[t5][t6];;
              }
            }
          }
        }
        if ((t1 == t3) && (t1 >= t2+1)) {
          for (t4=32*t1;t4<=min(_PB_N-1,32*t1+31);t4++) {
            for (t5=32*t2;t5<=32*t2+31;t5++) {
              A[t4][t4] -= A[t4][t5] * A[t4][t5];;
            }
          }
        }
        if ((t1 == t2) && (t1 == t3)) {
          A[32*t1][32*t1] = SQRT_FUN(A[32*t1][32*t1]);;
        }
        if ((t1 == t2) && (t1 == t3) && (t1 <= floord(_PB_N-2,32))) {
          A[(32*t1+1)][32*t1] /= A[32*t1][32*t1];;
          A[(32*t1+1)][(32*t1+1)] -= A[(32*t1+1)][32*t1] * A[(32*t1+1)][32*t1];;
          A[(32*t1+1)][(32*t1+1)] = SQRT_FUN(A[(32*t1+1)][(32*t1+1)]);;
        }
        if ((t1 == t2) && (t1 == t3)) {
          for (t4=32*t1+2;t4<=min(_PB_N-1,32*t1+31);t4++) {
            A[t4][32*t1] /= A[32*t1][32*t1];;
            A[t4][t4] -= A[t4][32*t1] * A[t4][32*t1];;
            for (t5=32*t1+1;t5<=t4-1;t5++) {
              for (t6=32*t1;t6<=t5-1;t6++) {
                A[t4][t5] -= A[t4][t6] * A[t5][t6];;
              }
              A[t4][t5] /= A[t5][t5];;
              A[t4][t4] -= A[t4][t5] * A[t4][t5];;
            }
            A[t4][t4] = SQRT_FUN(A[t4][t4]);;
          }
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

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK:  func @kernel_cholesky(%arg0: i32, %arg1: memref<2000x2000xf64>) {
// CHECK-NEXT:  %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  affine.for %arg2 = 0 to %0 {
// CHECK-NEXT:    affine.for %arg3 = 0 to #map(%arg2) {
// CHECK-NEXT:      %4 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      affine.for %arg4 = 0 to #map(%arg3) {
// CHECK-NEXT:        %8 = affine.load %arg1[%arg2, %arg4] : memref<2000x2000xf64>
// CHECK-NEXT:        %9 = affine.load %arg1[%arg3, %arg4] : memref<2000x2000xf64>
// CHECK-NEXT:        %10 = mulf %8, %9 : f64
// CHECK-NEXT:        %11 = subf %4, %10 : f64
// CHECK-NEXT:        affine.store %11, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = affine.load %arg1[%arg3, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      %6 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      %7 = divf %6, %5 : f64
// CHECK-NEXT:      affine.store %7, %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = affine.load %arg1[%arg2, %arg2] : memref<2000x2000xf64>
// CHECK-NEXT:    affine.for %arg3 = 0 to #map(%arg2) {
// CHECK-NEXT:      %4 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      %5 = affine.load %arg1[%arg2, %arg3] : memref<2000x2000xf64>
// CHECK-NEXT:      %6 = mulf %4, %5 : f64
// CHECK-NEXT:      %7 = subf %1, %6 : f64
// CHECK-NEXT:      affine.store %7, %arg1[%arg2, %arg2] : memref<2000x2000xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = affine.load %arg1[%arg2, %arg2] : memref<2000x2000xf64>
// CHECK-NEXT:    %3 = sqrt %2 : f64
// CHECK-NEXT:    affine.store %3, %arg1[%arg2, %arg2] : memref<2000x2000xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}
