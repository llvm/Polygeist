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
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }
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
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  int t1, t2, t3, t4, t5;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (_PB_N >= 1) {
  for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
    for (t3=32*t2;t3<=min(_PB_N-1,32*t2+31);t3++) {
      x[t3] = b[t3];;
    }
  }
  for (t2=0;t2<=floord(_PB_N-1,16);t2++) {
    for (t3=max(0,ceild(32*t2-_PB_N+1,32));t3<=floord(t2,2);t3++) {
      if (t2 >= 2*t3+1) {
        for (t4=32*t2-32*t3;t4<=min(_PB_N-1,32*t2-32*t3+31);t4++) {
          for (t5=32*t3;t5<=32*t3+31;t5++) {
            x[t4] -= L[t4][t5] * x[t5];;
          }
        }
      }
      if (t2 == 2*t3) {
        if (t2%2 == 0) {
          x[16*t2] = x[16*t2] / L[16*t2][16*t2];;
        }
      }
      if (t2 == 2*t3) {
        for (t4=16*t2+1;t4<=min(_PB_N-1,16*t2+31);t4++) {
          for (t5=16*t2;t5<=t4-1;t5++) {
            if (t2%2 == 0) {
              x[t4] -= L[t4][t5] * x[t5];;
            }
          }
          if (t2%2 == 0) {
            x[t4] = x[t4] / L[t4][t4];;
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
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(
    print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}

// CHECK:   #map = affine_map<(d0) -> (d0)>
// CHECK:   func @kernel_trisolv(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
// CHECK-NEXT:     %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg4 = 0 to %0 {
// CHECK-NEXT:       %1 = affine.load %arg3[%arg4] : memref<2000xf64>
// CHECK-NEXT:       affine.store %1, %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:       %2 = affine.load %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:       affine.for %arg5 = 0 to #map(%arg4) {
// CHECK-NEXT:         %6 = affine.load %arg1[%arg4, %arg5] : memref<2000x2000xf64>
// CHECK-NEXT:         %7 = affine.load %arg2[%arg5] : memref<2000xf64>
// CHECK-NEXT:         %8 = mulf %6, %7 : f64
// CHECK-NEXT:         %9 = subf %2, %8 : f64
// CHECK-NEXT:         affine.store %9, %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = affine.load %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:       %4 = affine.load %arg1[%arg4, %arg4] : memref<2000x2000xf64>
// CHECK-NEXT:       %5 = divf %3, %4 : f64
// CHECK-NEXT:       affine.store %5, %arg2[%arg4] : memref<2000xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// EXEC: {{[0-9]\.[0-9]+}}
