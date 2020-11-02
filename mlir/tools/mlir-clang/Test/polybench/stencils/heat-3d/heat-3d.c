// RUN: mlir-clang %s main %stdinclude | FileCheck %s
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

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c120_i32 = constant 120 : i32
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c120_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = muli %1, %0 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %9 = load %6[%c0] : memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?x120x120xf64> to memref<120x120x120xf64>
// CHECK-NEXT:     %12 = load %8[%c0] : memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?x120x120xf64> to memref<120x120x120xf64>
// CHECK-NEXT:     call @init_array(%c120_i32, %11, %14) : (i32, memref<120x120x120xf64>, memref<120x120x120xf64>) -> ()
// CHECK-NEXT:     %15 = load %6[%c0] : memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x120x120xf64> to memref<120x120x120xf64>
// CHECK-NEXT:     %18 = load %8[%c0] : memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x120x120xf64> to memref<120x120x120xf64>
// CHECK-NEXT:     call @kernel_heat_3d(%c500_i32, %c120_i32, %17, %20) : (i32, i32, memref<120x120x120xf64>, memref<120x120x120xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %21 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %22 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = load %arg1[%23] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %25 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %26 = call @strcmp(%24, %25) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %27 = trunci %26 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %28 = xor %27, %true : i1
// CHECK-NEXT:     %29 = and %21, %28 : i1
// CHECK-NEXT:     scf.if %29 {
// CHECK-NEXT:       %31 = load %6[%c0] : memref<?xmemref<120x120x120xf64>>
// CHECK-NEXT:       %32 = memref_cast %31 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:       %33 = memref_cast %32 : memref<?x120x120xf64> to memref<120x120x120xf64>
// CHECK-NEXT:       call @print_array(%c120_i32, %33) : (i32, memref<120x120x120xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %30 = memref_cast %6 : memref<?xmemref<120x120x120xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%30) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<120x120x120xf64>, %arg2: memref<120x120x120xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg0 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = index_cast %0 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = memref_cast %arg1 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %10 = index_cast %2 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %9 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %13 = index_cast %5 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = memref_cast %arg2 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %17 = addi %0, %2 : i32
// CHECK-NEXT:     %18 = subi %arg0, %5 : i32
// CHECK-NEXT:     %19 = addi %17, %18 : i32
// CHECK-NEXT:     %20 = sitofp %19 : i32 to f64
// CHECK-NEXT:     %c10_i32 = constant 10 : i32
// CHECK-NEXT:     %21 = sitofp %c10_i32 : i32 to f64
// CHECK-NEXT:     %22 = mulf %20, %21 : f64
// CHECK-NEXT:     %23 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %24 = divf %22, %23 : f64
// CHECK-NEXT:     store %24, %16[%8, %11, %14] : memref<?x120x120xf64>
// CHECK-NEXT:     %25 = load %16[%8, %11, %14] : memref<?x120x120xf64>
// CHECK-NEXT:     store %25, %12[%8, %11, %14] : memref<?x120x120xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %5, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%26 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %27 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%27 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c1_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb15
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %1 = cmpi "sle", %0, %c500_i32 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c1_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     %4 = cmpi "slt", %2, %3 : i32
// CHECK-NEXT:     cond_br %4, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c1_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb13(%c1_i32 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb12
// CHECK-NEXT:     %6 = cmpi "slt", %5, %3 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c1_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %7 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%7 : i32)
// CHECK-NEXT:   ^bb10(%8: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %9 = cmpi "slt", %8, %3 : i32
// CHECK-NEXT:     cond_br %9, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %10 = index_cast %2 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %arg3 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %13 = index_cast %5 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = memref_cast %12 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %16 = index_cast %8 : i32 to index
// CHECK-NEXT:     %17 = addi %c0, %16 : index
// CHECK-NEXT:     %cst = constant 1.250000e-01 : f64
// CHECK-NEXT:     %18 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     %19 = index_cast %18 : i32 to index
// CHECK-NEXT:     %20 = addi %c0, %19 : index
// CHECK-NEXT:     %21 = memref_cast %arg2 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %23 = load %22[%20, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %24 = load %22[%11, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %25 = mulf %cst_0, %24 : f64
// CHECK-NEXT:     %26 = subf %23, %25 : f64
// CHECK-NEXT:     %27 = subi %2, %c1_i32 : i32
// CHECK-NEXT:     %28 = index_cast %27 : i32 to index
// CHECK-NEXT:     %29 = addi %c0, %28 : index
// CHECK-NEXT:     %30 = load %22[%29, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %31 = addf %26, %30 : f64
// CHECK-NEXT:     %32 = mulf %cst, %31 : f64
// CHECK-NEXT:     %33 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     %34 = index_cast %33 : i32 to index
// CHECK-NEXT:     %35 = addi %c0, %34 : index
// CHECK-NEXT:     %36 = load %22[%11, %35, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %37 = load %22[%11, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %38 = mulf %cst_0, %37 : f64
// CHECK-NEXT:     %39 = subf %36, %38 : f64
// CHECK-NEXT:     %40 = subi %5, %c1_i32 : i32
// CHECK-NEXT:     %41 = index_cast %40 : i32 to index
// CHECK-NEXT:     %42 = addi %c0, %41 : index
// CHECK-NEXT:     %43 = load %22[%11, %42, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %44 = addf %39, %43 : f64
// CHECK-NEXT:     %45 = mulf %cst, %44 : f64
// CHECK-NEXT:     %46 = addf %32, %45 : f64
// CHECK-NEXT:     %47 = addi %8, %c1_i32 : i32
// CHECK-NEXT:     %48 = index_cast %47 : i32 to index
// CHECK-NEXT:     %49 = addi %c0, %48 : index
// CHECK-NEXT:     %50 = load %22[%11, %14, %49] : memref<?x120x120xf64>
// CHECK-NEXT:     %51 = load %22[%11, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %52 = mulf %cst_0, %51 : f64
// CHECK-NEXT:     %53 = subf %50, %52 : f64
// CHECK-NEXT:     %54 = subi %8, %c1_i32 : i32
// CHECK-NEXT:     %55 = index_cast %54 : i32 to index
// CHECK-NEXT:     %56 = addi %c0, %55 : index
// CHECK-NEXT:     %57 = load %22[%11, %14, %56] : memref<?x120x120xf64>
// CHECK-NEXT:     %58 = addf %53, %57 : f64
// CHECK-NEXT:     %59 = mulf %cst, %58 : f64
// CHECK-NEXT:     %60 = addf %46, %59 : f64
// CHECK-NEXT:     %61 = load %22[%11, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     %62 = addf %60, %61 : f64
// CHECK-NEXT:     store %62, %15[%11, %14, %17] : memref<?x120x120xf64>
// CHECK-NEXT:     br ^bb10(%47 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %63 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%63 : i32)
// CHECK-NEXT:   ^bb13(%64: i32):  // 2 preds: ^bb6, ^bb18
// CHECK-NEXT:     %65 = cmpi "slt", %64, %3 : i32
// CHECK-NEXT:     cond_br %65, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c1_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %66 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%66 : i32)
// CHECK-NEXT:   ^bb16(%67: i32):  // 2 preds: ^bb14, ^bb21
// CHECK-NEXT:     %68 = cmpi "slt", %67, %3 : i32
// CHECK-NEXT:     cond_br %68, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     br ^bb19(%c1_i32 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %69 = addi %64, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%69 : i32)
// CHECK-NEXT:   ^bb19(%70: i32):  // 2 preds: ^bb17, ^bb20
// CHECK-NEXT:     %71 = cmpi "slt", %70, %3 : i32
// CHECK-NEXT:     cond_br %71, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %72 = index_cast %64 : i32 to index
// CHECK-NEXT:     %73 = addi %c0, %72 : index
// CHECK-NEXT:     %74 = memref_cast %arg2 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %75 = index_cast %67 : i32 to index
// CHECK-NEXT:     %76 = addi %c0, %75 : index
// CHECK-NEXT:     %77 = memref_cast %74 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %78 = index_cast %70 : i32 to index
// CHECK-NEXT:     %79 = addi %c0, %78 : index
// CHECK-NEXT:     %cst_1 = constant 1.250000e-01 : f64
// CHECK-NEXT:     %80 = addi %64, %c1_i32 : i32
// CHECK-NEXT:     %81 = index_cast %80 : i32 to index
// CHECK-NEXT:     %82 = addi %c0, %81 : index
// CHECK-NEXT:     %83 = memref_cast %arg3 : memref<120x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %84 = memref_cast %83 : memref<?x120x120xf64> to memref<?x120x120xf64>
// CHECK-NEXT:     %85 = load %84[%82, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %cst_2 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %86 = load %84[%73, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %87 = mulf %cst_2, %86 : f64
// CHECK-NEXT:     %88 = subf %85, %87 : f64
// CHECK-NEXT:     %89 = subi %64, %c1_i32 : i32
// CHECK-NEXT:     %90 = index_cast %89 : i32 to index
// CHECK-NEXT:     %91 = addi %c0, %90 : index
// CHECK-NEXT:     %92 = load %84[%91, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %93 = addf %88, %92 : f64
// CHECK-NEXT:     %94 = mulf %cst_1, %93 : f64
// CHECK-NEXT:     %95 = addi %67, %c1_i32 : i32
// CHECK-NEXT:     %96 = index_cast %95 : i32 to index
// CHECK-NEXT:     %97 = addi %c0, %96 : index
// CHECK-NEXT:     %98 = load %84[%73, %97, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %99 = load %84[%73, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %100 = mulf %cst_2, %99 : f64
// CHECK-NEXT:     %101 = subf %98, %100 : f64
// CHECK-NEXT:     %102 = subi %67, %c1_i32 : i32
// CHECK-NEXT:     %103 = index_cast %102 : i32 to index
// CHECK-NEXT:     %104 = addi %c0, %103 : index
// CHECK-NEXT:     %105 = load %84[%73, %104, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %106 = addf %101, %105 : f64
// CHECK-NEXT:     %107 = mulf %cst_1, %106 : f64
// CHECK-NEXT:     %108 = addf %94, %107 : f64
// CHECK-NEXT:     %109 = addi %70, %c1_i32 : i32
// CHECK-NEXT:     %110 = index_cast %109 : i32 to index
// CHECK-NEXT:     %111 = addi %c0, %110 : index
// CHECK-NEXT:     %112 = load %84[%73, %76, %111] : memref<?x120x120xf64>
// CHECK-NEXT:     %113 = load %84[%73, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %114 = mulf %cst_2, %113 : f64
// CHECK-NEXT:     %115 = subf %112, %114 : f64
// CHECK-NEXT:     %116 = subi %70, %c1_i32 : i32
// CHECK-NEXT:     %117 = index_cast %116 : i32 to index
// CHECK-NEXT:     %118 = addi %c0, %117 : index
// CHECK-NEXT:     %119 = load %84[%73, %76, %118] : memref<?x120x120xf64>
// CHECK-NEXT:     %120 = addf %115, %119 : f64
// CHECK-NEXT:     %121 = mulf %cst_1, %120 : f64
// CHECK-NEXT:     %122 = addf %108, %121 : f64
// CHECK-NEXT:     %123 = load %84[%73, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     %124 = addf %122, %123 : f64
// CHECK-NEXT:     store %124, %77[%73, %76, %79] : memref<?x120x120xf64>
// CHECK-NEXT:     br ^bb19(%109 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %125 = addi %67, %c1_i32 : i32
// CHECK-NEXT:     br ^bb16(%125 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<120x120x120xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg0 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = muli %0, %arg0 : i32
// CHECK-NEXT:     %8 = muli %7, %arg0 : i32
// CHECK-NEXT:     %9 = muli %2, %arg0 : i32
// CHECK-NEXT:     %10 = addi %8, %9 : i32
// CHECK-NEXT:     %11 = addi %10, %5 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %12 = remi_signed %11, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %13 = addi %5, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%13 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%14 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }