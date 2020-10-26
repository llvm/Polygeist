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
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt(int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = SQRT_FUN(nrm);
      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < _PB_N; j++)
	{
	  R[k][j] = SCALAR_VAL(0.0);
	  for (i = 0; i < _PB_M; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < _PB_M; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %7 = muli %1, %1 : i32
// CHECK-NEXT:     %8 = zexti %7 : i32 to i64
// CHECK-NEXT:     %9 = call @polybench_alloc_data(%8, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?xi8> to memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %13 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %16 = load %10[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %19 = load %12[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     call @init_array(%c1000_i32, %c1200_i32, %15, %18, %21) : (i32, i32, memref<1000x1200xf64>, memref<1200x1200xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     %22 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %25 = load %10[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %28 = load %12[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     call @kernel_gramschmidt(%c1000_i32, %c1200_i32, %24, %27, %30) : (i32, i32, memref<1000x1200xf64>, memref<1200x1200xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %31 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %32 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = load %arg1[%33] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %35 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %36 = call @strcmp(%34, %35) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %37 = trunci %36 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %38 = xor %37, %true : i1
// CHECK-NEXT:     %39 = and %31, %38 : i1
// CHECK-NEXT:     scf.if %39 {
// CHECK-NEXT:       %43 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %44 = memref_cast %43 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %45 = memref_cast %44 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       %46 = load %10[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:       %47 = memref_cast %46 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %48 = memref_cast %47 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:       %49 = load %12[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %50 = memref_cast %49 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %51 = memref_cast %50 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       call @print_array(%c1000_i32, %c1200_i32, %45, %48, %51) : (i32, i32, memref<1000x1200xf64>, memref<1200x1200xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %40 = memref_cast %6 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%40) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %41 = memref_cast %10 : memref<?xmemref<1200x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%41) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %42 = memref_cast %12 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%42) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>, %arg3: memref<1200x1200xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg2 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = muli %0, %2 : i32
// CHECK-NEXT:     %10 = remi_signed %9, %arg0 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %13 = divf %11, %12 : f64
// CHECK-NEXT:     %c100_i32 = constant 100 : i32
// CHECK-NEXT:     %14 = sitofp %c100_i32 : i32 to f64
// CHECK-NEXT:     %15 = mulf %13, %14 : f64
// CHECK-NEXT:     %c10_i32 = constant 10 : i32
// CHECK-NEXT:     %16 = sitofp %c10_i32 : i32 to f64
// CHECK-NEXT:     %17 = addf %15, %16 : f64
// CHECK-NEXT:     store %17, %6[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %18 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %18[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %19 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%19 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %20 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%20 : i32)
// CHECK-NEXT:   ^bb7(%21: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %22 = cmpi "slt", %21, %arg1 : i32
// CHECK-NEXT:     cond_br %22, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%23: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg1 : i32
// CHECK-NEXT:     cond_br %24, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %25 = index_cast %21 : i32 to index
// CHECK-NEXT:     %26 = addi %c0, %25 : index
// CHECK-NEXT:     %27 = memref_cast %arg3 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %28 = index_cast %23 : i32 to index
// CHECK-NEXT:     %29 = addi %c0, %28 : index
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_1, %27[%26, %29] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %30 = addi %23, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%30 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %31 = addi %21, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%31 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_gramschmidt(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>, %arg3: memref<1200x1200xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb12
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg1 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     br ^bb4(%c0_i32, %cst : i32, f64)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32, %3: f64):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %4 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %4, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     %6 = addi %c0, %5 : index
// CHECK-NEXT:     %7 = memref_cast %arg2 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %8 = index_cast %0 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %10 = load %7[%6, %9] : memref<?x1200xf64>
// CHECK-NEXT:     %11 = load %7[%6, %9] : memref<?x1200xf64>
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = addf %3, %12 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%14, %13 : i32, f64)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %15 = index_cast %0 : i32 to index
// CHECK-NEXT:     %16 = addi %c0, %15 : index
// CHECK-NEXT:     %17 = memref_cast %arg3 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %18 = sqrt %3 : f64
// CHECK-NEXT:     store %18, %17[%16, %16] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%19: i32):  // 2 preds: ^bb6, ^bb8
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg0 : i32
// CHECK-NEXT:     cond_br %20, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %21 = index_cast %19 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %24 = memref_cast %arg2 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = load %24[%22, %16] : memref<?x1200xf64>
// CHECK-NEXT:     %26 = load %17[%16, %16] : memref<?x1200xf64>
// CHECK-NEXT:     %27 = divf %25, %26 : f64
// CHECK-NEXT:     store %27, %23[%22, %16] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %28 = addi %19, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%28 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %29 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%29 : i32)
// CHECK-NEXT:   ^bb10(%30: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %31 = cmpi "slt", %30, %arg1 : i32
// CHECK-NEXT:     cond_br %31, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %32 = index_cast %30 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     store %cst, %17[%16, %33] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     br ^bb1(%29 : i32)
// CHECK-NEXT:   ^bb13(%34: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %35 = cmpi "slt", %34, %arg0 : i32
// CHECK-NEXT:     cond_br %35, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %36 = index_cast %34 : i32 to index
// CHECK-NEXT:     %37 = addi %c0, %36 : index
// CHECK-NEXT:     %38 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %39 = load %38[%37, %16] : memref<?x1200xf64>
// CHECK-NEXT:     %40 = memref_cast %arg2 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %41 = load %40[%37, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %42 = mulf %39, %41 : f64
// CHECK-NEXT:     %43 = load %17[%16, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %44 = addf %43, %42 : f64
// CHECK-NEXT:     store %44, %17[%16, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %45 = addi %34, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb13(%45 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%46: i32):  // 2 preds: ^bb15, ^bb17
// CHECK-NEXT:     %47 = cmpi "slt", %46, %arg0 : i32
// CHECK-NEXT:     cond_br %47, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %48 = index_cast %46 : i32 to index
// CHECK-NEXT:     %49 = addi %c0, %48 : index
// CHECK-NEXT:     %50 = memref_cast %arg2 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %51 = load %50[%49, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %52 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %53 = load %52[%49, %16] : memref<?x1200xf64>
// CHECK-NEXT:     %54 = load %17[%16, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %55 = mulf %53, %54 : f64
// CHECK-NEXT:     %56 = subf %51, %55 : f64
// CHECK-NEXT:     store %56, %50[%49, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %57 = addi %46, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb16(%57 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %58 = addi %30, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%58 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>, %arg3: memref<1200x1200xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg1 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = muli %0, %arg1 : i32
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
// CHECK-NEXT:   ^bb7(%9: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg0 : i32
// CHECK-NEXT:     cond_br %10, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%11: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg1 : i32
// CHECK-NEXT:     cond_br %12, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %13 = muli %9, %arg1 : i32
// CHECK-NEXT:     %14 = addi %13, %11 : i32
// CHECK-NEXT:     %c20_i32_1 = constant 20 : i32
// CHECK-NEXT:     %15 = remi_signed %14, %c20_i32_1 : i32
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %11, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%16 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %9, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%17 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }