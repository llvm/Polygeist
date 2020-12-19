#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
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
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syrk(int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (_PB_N >= 1) {
  for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
    for (t3=0;t3<=t2;t3++) {
      for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(t4,32*t3+31);t5++) {
          C[t4][t5] *= beta;;
        }
      }
    }
  }
  if (_PB_M >= 1) {
    for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
      for (t3=0;t3<=t2;t3++) {
        for (t4=0;t4<=floord(_PB_M-1,32);t4++) {
          for (t5=32*t2;t5<=min(_PB_N-1,32*t2+31);t5++) {
            for (t6=32*t3;t6<=min(t5,32*t3+31);t6++) {
              for (t7=32*t4;t7<=min(_PB_M-1,32*t4+31);t7++) {
                C[t5][t6] += alpha * A[t5][t7] * A[t6][t7];;
              }
            }
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
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK:  func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>) {
// CHECK-NEXT:    %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    affine.for %arg6 = 0 to %0 {
// CHECK-NEXT:      affine.for %arg7 = 0 to #map(%arg6) {
// CHECK-NEXT:        %2 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        %3 = mulf %2, %arg3 : f64
// CHECK-NEXT:        affine.store %3, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        %2 = affine.load %arg5[%arg6, %arg7] : memref<1200x1000xf64>
// CHECK-NEXT:        %3 = mulf %arg2, %2 : f64
// CHECK-NEXT:        affine.for %arg8 = 0 to #map(%arg6) {
// CHECK-NEXT:          %4 = affine.load %arg5[%arg8, %arg7] : memref<1200x1000xf64>
// CHECK-NEXT:          %5 = mulf %3, %4 : f64
// CHECK-NEXT:          %6 = affine.load %arg4[%arg6, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:          %7 = addf %6, %5 : f64
// CHECK-NEXT:          affine.store %7, %arg4[%arg6, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// EXEC: {{[0-9]\.[0-9]+}}
