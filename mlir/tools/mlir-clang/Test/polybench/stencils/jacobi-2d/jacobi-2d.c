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

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
	for (j = 1; j < _PB_N - 1; j++)
	  B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      for (i = 1; i < _PB_N - 1; i++)
	for (j = 1; j < _PB_N - 1; j++)
	  A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
    }
#pragma endscop

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

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1300_i32 = constant 1300 : i32
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c1300_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = zexti %1 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %3 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %4 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %5 = memref_cast %4 : memref<?xi8> to memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %8 = load %5[%c0] : memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?x1300xf64> to memref<1300x1300xf64>
// CHECK-NEXT:     %11 = load %7[%c0] : memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<?x1300xf64> to memref<1300x1300xf64>
// CHECK-NEXT:     call @init_array(%c1300_i32, %10, %13) : (i32, memref<1300x1300xf64>, memref<1300x1300xf64>) -> ()
// CHECK-NEXT:     %14 = load %5[%c0] : memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?x1300xf64> to memref<1300x1300xf64>
// CHECK-NEXT:     %17 = load %7[%c0] : memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?x1300xf64> to memref<1300x1300xf64>
// CHECK-NEXT:     call @kernel_jacobi_2d(%c500_i32, %c1300_i32, %16, %19) : (i32, i32, memref<1300x1300xf64>, memref<1300x1300xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %20 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %21 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = load %arg1[%22] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %24 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %25 = call @strcmp(%23, %24) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %26 = trunci %25 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %27 = xor %26, %true : i1
// CHECK-NEXT:     %28 = and %20, %27 : i1
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       %31 = load %5[%c0] : memref<?xmemref<1300x1300xf64>>
// CHECK-NEXT:       %32 = memref_cast %31 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:       %33 = memref_cast %32 : memref<?x1300xf64> to memref<1300x1300xf64>
// CHECK-NEXT:       call @print_array(%c1300_i32, %33) : (i32, memref<1300x1300xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %29 = memref_cast %5 : memref<?xmemref<1300x1300xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%29) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %30 = memref_cast %7 : memref<?xmemref<1300x1300xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%30) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<1300x1300xf64>, %arg2: memref<1300x1300xf64>) {
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
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg1 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = sitofp %0 : i32 to f64
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %10 = addi %2, %c2_i32 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = mulf %9, %11 : f64
// CHECK-NEXT:     %13 = sitofp %c2_i32 : i32 to f64
// CHECK-NEXT:     %14 = addf %12, %13 : f64
// CHECK-NEXT:     %15 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %16 = divf %14, %15 : f64
// CHECK-NEXT:     store %16, %6[%5, %8] : memref<?x1300xf64>
// CHECK-NEXT:     %17 = memref_cast %arg2 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %18 = addi %2, %c3_i32 : i32
// CHECK-NEXT:     %19 = sitofp %18 : i32 to f64
// CHECK-NEXT:     %20 = mulf %9, %19 : f64
// CHECK-NEXT:     %21 = sitofp %c3_i32 : i32 to f64
// CHECK-NEXT:     %22 = addf %20, %21 : f64
// CHECK-NEXT:     %23 = divf %22, %15 : f64
// CHECK-NEXT:     store %23, %17[%5, %8] : memref<?x1300xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %24 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%24 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %25 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%25 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<1300x1300xf64>, %arg3: memref<1300x1300xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb12
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
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
// CHECK-NEXT:     br ^bb10(%c1_i32 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %3 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = memref_cast %arg3 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %10 = index_cast %5 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %cst = constant 2.000000e-01 : f64
// CHECK-NEXT:     %12 = memref_cast %arg2 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %13 = load %12[%8, %11] : memref<?x1300xf64>
// CHECK-NEXT:     %14 = subi %5, %c1_i32 : i32
// CHECK-NEXT:     %15 = index_cast %14 : i32 to index
// CHECK-NEXT:     %16 = addi %c0, %15 : index
// CHECK-NEXT:     %17 = load %12[%8, %16] : memref<?x1300xf64>
// CHECK-NEXT:     %18 = addf %13, %17 : f64
// CHECK-NEXT:     %19 = addi %c1_i32, %5 : i32
// CHECK-NEXT:     %20 = index_cast %19 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = load %12[%8, %21] : memref<?x1300xf64>
// CHECK-NEXT:     %23 = addf %18, %22 : f64
// CHECK-NEXT:     %24 = addi %c1_i32, %2 : i32
// CHECK-NEXT:     %25 = index_cast %24 : i32 to index
// CHECK-NEXT:     %26 = addi %c0, %25 : index
// CHECK-NEXT:     %27 = load %12[%26, %11] : memref<?x1300xf64>
// CHECK-NEXT:     %28 = addf %23, %27 : f64
// CHECK-NEXT:     %29 = subi %2, %c1_i32 : i32
// CHECK-NEXT:     %30 = index_cast %29 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = load %12[%31, %11] : memref<?x1300xf64>
// CHECK-NEXT:     %33 = addf %28, %32 : f64
// CHECK-NEXT:     %34 = mulf %cst, %33 : f64
// CHECK-NEXT:     store %34, %9[%8, %11] : memref<?x1300xf64>
// CHECK-NEXT:     %35 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%35 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %36 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%36 : i32)
// CHECK-NEXT:   ^bb10(%37: i32):  // 2 preds: ^bb6, ^bb15
// CHECK-NEXT:     %38 = cmpi "slt", %37, %3 : i32
// CHECK-NEXT:     cond_br %38, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     br ^bb13(%c1_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %39 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%39 : i32)
// CHECK-NEXT:   ^bb13(%40: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %41 = cmpi "slt", %40, %3 : i32
// CHECK-NEXT:     cond_br %41, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %42 = index_cast %37 : i32 to index
// CHECK-NEXT:     %43 = addi %c0, %42 : index
// CHECK-NEXT:     %44 = memref_cast %arg2 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %45 = index_cast %40 : i32 to index
// CHECK-NEXT:     %46 = addi %c0, %45 : index
// CHECK-NEXT:     %cst_0 = constant 2.000000e-01 : f64
// CHECK-NEXT:     %47 = memref_cast %arg3 : memref<1300x1300xf64> to memref<?x1300xf64>
// CHECK-NEXT:     %48 = load %47[%43, %46] : memref<?x1300xf64>
// CHECK-NEXT:     %49 = subi %40, %c1_i32 : i32
// CHECK-NEXT:     %50 = index_cast %49 : i32 to index
// CHECK-NEXT:     %51 = addi %c0, %50 : index
// CHECK-NEXT:     %52 = load %47[%43, %51] : memref<?x1300xf64>
// CHECK-NEXT:     %53 = addf %48, %52 : f64
// CHECK-NEXT:     %54 = addi %c1_i32, %40 : i32
// CHECK-NEXT:     %55 = index_cast %54 : i32 to index
// CHECK-NEXT:     %56 = addi %c0, %55 : index
// CHECK-NEXT:     %57 = load %47[%43, %56] : memref<?x1300xf64>
// CHECK-NEXT:     %58 = addf %53, %57 : f64
// CHECK-NEXT:     %59 = addi %c1_i32, %37 : i32
// CHECK-NEXT:     %60 = index_cast %59 : i32 to index
// CHECK-NEXT:     %61 = addi %c0, %60 : index
// CHECK-NEXT:     %62 = load %47[%61, %46] : memref<?x1300xf64>
// CHECK-NEXT:     %63 = addf %58, %62 : f64
// CHECK-NEXT:     %64 = subi %37, %c1_i32 : i32
// CHECK-NEXT:     %65 = index_cast %64 : i32 to index
// CHECK-NEXT:     %66 = addi %c0, %65 : index
// CHECK-NEXT:     %67 = load %47[%66, %46] : memref<?x1300xf64>
// CHECK-NEXT:     %68 = addf %63, %67 : f64
// CHECK-NEXT:     %69 = mulf %cst_0, %68 : f64
// CHECK-NEXT:     store %69, %44[%43, %46] : memref<?x1300xf64>
// CHECK-NEXT:     %70 = addi %40, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%70 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %71 = addi %37, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%71 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1300x1300xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = muli %0, %arg0 : i32
// CHECK-NEXT:     %5 = addi %4, %2 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %6 = remi_signed %5, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%7 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %8 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%8 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }