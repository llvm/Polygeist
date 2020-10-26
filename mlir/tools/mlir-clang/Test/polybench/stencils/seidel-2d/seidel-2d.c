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
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
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
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
    for (i = 1; i<= _PB_N - 2; i++)
      for (j = 1; j <= _PB_N - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/SCALAR_VAL(9.0);
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d (tsteps, n, POLYBENCH_ARRAY(A));

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
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = zexti %1 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %3 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %4 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %5 = memref_cast %4 : memref<?xi8> to memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %6 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %8) : (i32, memref<2000x2000xf64>) -> ()
// CHECK-NEXT:     %9 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     call @kernel_seidel_2d(%c500_i32, %c2000_i32, %11) : (i32, i32, memref<2000x2000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %12 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %13 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = load %arg1[%14] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %16 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %17 = call @strcmp(%15, %16) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %18 = trunci %17 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %19 = xor %18, %true : i1
// CHECK-NEXT:     %20 = and %12, %19 : i1
// CHECK-NEXT:     scf.if %20 {
// CHECK-NEXT:       %22 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:       %23 = memref_cast %22 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:       %24 = memref_cast %23 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %24) : (i32, memref<2000x2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %21 = memref_cast %5 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%21) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2000x2000xf64>) {
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
// CHECK-NEXT:     %6 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
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
// CHECK-NEXT:     store %16, %6[%5, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%17 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %18 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%18 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %1 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     %2 = cmpi "sle", %0, %1 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c1_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%3: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %4 = subi %arg1, %c2_i32 : i32
// CHECK-NEXT:     %5 = cmpi "sle", %3, %4 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c1_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %6 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%6 : i32)
// CHECK-NEXT:   ^bb7(%7: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %8 = cmpi "sle", %7, %4 : i32
// CHECK-NEXT:     cond_br %8, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %9 = index_cast %3 : i32 to index
// CHECK-NEXT:     %10 = addi %c0, %9 : index
// CHECK-NEXT:     %11 = memref_cast %arg2 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %12 = index_cast %7 : i32 to index
// CHECK-NEXT:     %13 = addi %c0, %12 : index
// CHECK-NEXT:     %14 = subi %3, %c1_i32 : i32
// CHECK-NEXT:     %15 = index_cast %14 : i32 to index
// CHECK-NEXT:     %16 = addi %c0, %15 : index
// CHECK-NEXT:     %17 = subi %7, %c1_i32 : i32
// CHECK-NEXT:     %18 = index_cast %17 : i32 to index
// CHECK-NEXT:     %19 = addi %c0, %18 : index
// CHECK-NEXT:     %20 = load %11[%16, %19] : memref<?x2000xf64>
// CHECK-NEXT:     %21 = load %11[%16, %13] : memref<?x2000xf64>
// CHECK-NEXT:     %22 = addf %20, %21 : f64
// CHECK-NEXT:     %23 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     %24 = index_cast %23 : i32 to index
// CHECK-NEXT:     %25 = addi %c0, %24 : index
// CHECK-NEXT:     %26 = load %11[%16, %25] : memref<?x2000xf64>
// CHECK-NEXT:     %27 = addf %22, %26 : f64
// CHECK-NEXT:     %28 = load %11[%10, %19] : memref<?x2000xf64>
// CHECK-NEXT:     %29 = addf %27, %28 : f64
// CHECK-NEXT:     %30 = load %11[%10, %13] : memref<?x2000xf64>
// CHECK-NEXT:     %31 = addf %29, %30 : f64
// CHECK-NEXT:     %32 = load %11[%10, %25] : memref<?x2000xf64>
// CHECK-NEXT:     %33 = addf %31, %32 : f64
// CHECK-NEXT:     %34 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     %35 = index_cast %34 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %11[%36, %19] : memref<?x2000xf64>
// CHECK-NEXT:     %38 = addf %33, %37 : f64
// CHECK-NEXT:     %39 = load %11[%36, %13] : memref<?x2000xf64>
// CHECK-NEXT:     %40 = addf %38, %39 : f64
// CHECK-NEXT:     %41 = load %11[%36, %25] : memref<?x2000xf64>
// CHECK-NEXT:     %42 = addf %40, %41 : f64
// CHECK-NEXT:     %cst = constant 9.000000e+00 : f64
// CHECK-NEXT:     %43 = divf %42, %cst : f64
// CHECK-NEXT:     store %43, %11[%10, %13] : memref<?x2000xf64>
// CHECK-NEXT:     br ^bb7(%23 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %44 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%44 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000x2000xf64>) {
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