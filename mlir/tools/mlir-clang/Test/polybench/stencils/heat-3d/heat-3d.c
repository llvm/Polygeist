// RUN: mlir-clang %s %stdinclude | FileCheck %s

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;

#pragma scop
    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < _PB_N-1; i++) {
            for (j = 1; j < _PB_N-1; j++) {
                for (k = 1; k < _PB_N-1; k++) {
                    B[i][j][k] =   SCALAR_VAL(0.125) * (A[i+1][j][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i-1][j][k])
                                 + SCALAR_VAL(0.125) * (A[i][j+1][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j-1][k])
                                 + SCALAR_VAL(0.125) * (A[i][j][k+1] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (i = 1; i < _PB_N-1; i++) {
           for (j = 1; j < _PB_N-1; j++) {
               for (k = 1; k < _PB_N-1; k++) {
                   A[i][j][k] =   SCALAR_VAL(0.125) * (B[i+1][j][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i-1][j][k])
                                + SCALAR_VAL(0.125) * (B[i][j+1][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j-1][k])
                                + SCALAR_VAL(0.125) * (B[i][j][k+1] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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

// CHECK: #map0 = affine_map<()[s0] -> (s0 - 1)>
// CHECK: #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK: #map2 = affine_map<(d0) -> (d0 - 1)>

// CHECK: func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
// CHECK-NEXT:  %cst = constant 1.250000e-01 : f64
// CHECK-NEXT:  %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:  %c1 = constant 1 : index
// CHECK-NEXT:  %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  affine.for %arg4 = 1 to 501 {
// CHECK-NEXT:    affine.for %arg5 = 1 to #map0()[%0] {
// CHECK-NEXT:      affine.for %arg6 = 1 to #map0()[%0] {
// CHECK-NEXT:        affine.for %arg7 = 1 to #map0()[%0] {
// CHECK-NEXT:          %1 = affine.apply #map1(%arg5)
// CHECK-NEXT:          %2 = affine.load %arg2[%1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %3 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %4 = mulf %cst_0, %3 : f64
// CHECK-NEXT:          %5 = subf %2, %4 : f64
// CHECK-NEXT:          %6 = affine.apply #map2(%arg5)
// CHECK-NEXT:          %7 = affine.load %arg2[%6, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %8 = addf %5, %7 : f64
// CHECK-NEXT:          %9 = mulf %cst, %8 : f64
// CHECK-NEXT:          %10 = affine.apply #map1(%arg6)
// CHECK-NEXT:          %11 = affine.load %arg2[%arg5, %10, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %12 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %13 = mulf %cst_0, %12 : f64
// CHECK-NEXT:          %14 = subf %11, %13 : f64
// CHECK-NEXT:          %15 = affine.apply #map2(%arg6)
// CHECK-NEXT:          %16 = affine.load %arg2[%arg5, %15, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %17 = addf %14, %16 : f64
// CHECK-NEXT:          %18 = mulf %cst, %17 : f64
// CHECK-NEXT:          %19 = addf %9, %18 : f64
// CHECK-NEXT:          %20 = affine.apply #map1(%arg7)
// CHECK-NEXT:          %21 = affine.load %arg2[%arg5, %arg6, %20] : memref<120x120x120xf64>
// CHECK-NEXT:          %22 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %23 = mulf %cst_0, %22 : f64
// CHECK-NEXT:          %24 = subf %21, %23 : f64
// CHECK-NEXT:          %25 = affine.apply #map2(%arg7)
// CHECK-NEXT:          %26 = affine.load %arg2[%arg5, %arg6, %25] : memref<120x120x120xf64>
// CHECK-NEXT:          %27 = addf %24, %26 : f64
// CHECK-NEXT:          %28 = mulf %cst, %27 : f64
// CHECK-NEXT:          %29 = addf %19, %28 : f64
// CHECK-NEXT:          %30 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          %31 = addf %29, %30 : f64
// CHECK-NEXT:          affine.store %31, %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.for %arg5 = 1 to #map0()[%0] {
// CHECK-NEXT:    affine.for %arg6 = 1 to #map0()[%0] {
// CHECK-NEXT:      affine.for %arg7 = 1 to #map0()[%0] {
// CHECK-NEXT:        %1 = affine.apply #map1(%arg5)
// CHECK-NEXT:        %2 = affine.load %arg3[%1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %3 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %4 = mulf %cst_0, %3 : f64
// CHECK-NEXT:        %5 = subf %2, %4 : f64
// CHECK-NEXT:        %6 = affine.apply #map2(%arg5)
// CHECK-NEXT:        %7 = affine.load %arg3[%6, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %8 = addf %5, %7 : f64
// CHECK-NEXT:        %9 = mulf %cst, %8 : f64
// CHECK-NEXT:        %10 = affine.apply #map1(%arg6)
// CHECK-NEXT:        %11 = affine.load %arg3[%arg5, %10, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %12 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %13 = mulf %cst_0, %12 : f64
// CHECK-NEXT:        %14 = subf %11, %13 : f64
// CHECK-NEXT:        %15 = affine.apply #map2(%arg6)
// CHECK-NEXT:        %16 = affine.load %arg3[%arg5, %15, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %17 = addf %14, %16 : f64
// CHECK-NEXT:        %18 = mulf %cst, %17 : f64
// CHECK-NEXT:        %19 = addf %9, %18 : f64
// CHECK-NEXT:        %20 = affine.apply #map1(%arg7)
// CHECK-NEXT:        %21 = affine.load %arg3[%arg5, %arg6, %20] : memref<120x120x120xf64>
// CHECK-NEXT:        %22 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %23 = mulf %cst_0, %22 : f64
// CHECK-NEXT:        %24 = subf %21, %23 : f64
// CHECK-NEXT:        %25 = affine.apply #map2(%arg7)
// CHECK-NEXT:        %26 = affine.load %arg3[%arg5, %arg6, %25] : memref<120x120x120xf64>
// CHECK-NEXT:        %27 = addf %24, %26 : f64
// CHECK-NEXT:        %28 = mulf %cst, %27 : f64
// CHECK-NEXT:        %29 = addf %19, %28 : f64
// CHECK-NEXT:        %30 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:        %31 = addf %29, %30 : f64
// CHECK-NEXT:        affine.store %31, %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT:return
// CHECK-NEXT:} 
