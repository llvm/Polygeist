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


#pragma scop
  for (i = 0; i < _PB_N; i++) {
     //j<i
     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }
     // i==j case
     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = SQRT_FUN(A[i][i]);
  }
#pragma endscop

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

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
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
// CHECK-NEXT:     call @kernel_cholesky(%c2000_i32, %11) : (i32, memref<2000x2000xf64>) -> ()
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
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb9
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %2 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = muli %2, %2 : i32
// CHECK-NEXT:     %4 = zexti %3 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %5 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%8: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %9 = cmpi "sle", %8, %0 : i32
// CHECK-NEXT:     cond_br %9, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %10 = index_cast %0 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %13 = index_cast %8 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = subi %c0_i32, %8 : i32
// CHECK-NEXT:     %16 = remi_signed %15, %arg0 : i32
// CHECK-NEXT:     %17 = sitofp %16 : i32 to f64
// CHECK-NEXT:     %18 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %19 = divf %17, %18 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %20 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:     %21 = addf %19, %20 : f64
// CHECK-NEXT:     store %21, %12[%11, %14] : memref<?x2000xf64>
// CHECK-NEXT:     %22 = addi %8, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%22 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %23 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%23 : i32)
// CHECK-NEXT:   ^bb7(%24: i32):  // 2 preds: ^bb6, ^bb8
// CHECK-NEXT:     %25 = cmpi "slt", %24, %arg0 : i32
// CHECK-NEXT:     cond_br %25, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %26 = index_cast %0 : i32 to index
// CHECK-NEXT:     %27 = addi %c0, %26 : index
// CHECK-NEXT:     %28 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %29 = index_cast %24 : i32 to index
// CHECK-NEXT:     %30 = addi %c0, %29 : index
// CHECK-NEXT:     %31 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %31, %28[%27, %30] : memref<?x2000xf64>
// CHECK-NEXT:     %32 = addi %24, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %33 = index_cast %0 : i32 to index
// CHECK-NEXT:     %34 = addi %c0, %33 : index
// CHECK-NEXT:     %35 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %36 = sitofp %c1_i32_0 : i32 to f64
// CHECK-NEXT:     store %36, %35[%34, %34] : memref<?x2000xf64>
// CHECK-NEXT:     br ^bb1(%23 : i32)
// CHECK-NEXT:   ^bb10(%37: i32):  // 2 preds: ^bb3, ^bb15
// CHECK-NEXT:     %38 = cmpi "slt", %37, %arg0 : i32
// CHECK-NEXT:     cond_br %38, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb13(%39: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %40 = cmpi "slt", %39, %arg0 : i32
// CHECK-NEXT:     cond_br %40, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %41 = load %7[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %42 = memref_cast %41 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %43 = index_cast %37 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %45 = memref_cast %42 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %46 = index_cast %39 : i32 to index
// CHECK-NEXT:     %47 = addi %c0, %46 : index
// CHECK-NEXT:     %48 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %48, %45[%44, %47] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %49 = addi %39, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb13(%49 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %50 = addi %37, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%50 : i32)
// CHECK-NEXT:   ^bb16(%51: i32):  // 2 preds: ^bb12, ^bb21
// CHECK-NEXT:     %52 = cmpi "slt", %51, %arg0 : i32
// CHECK-NEXT:     cond_br %52, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     br ^bb25(%c0_i32 : i32)
// CHECK-NEXT:   ^bb19(%53: i32):  // 2 preds: ^bb17, ^bb24
// CHECK-NEXT:     %54 = cmpi "slt", %53, %arg0 : i32
// CHECK-NEXT:     cond_br %54, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %55 = addi %51, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb16(%55 : i32)
// CHECK-NEXT:   ^bb22(%56: i32):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %57 = cmpi "slt", %56, %arg0 : i32
// CHECK-NEXT:     cond_br %57, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %58 = load %7[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %59 = memref_cast %58 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %60 = index_cast %53 : i32 to index
// CHECK-NEXT:     %61 = addi %c0, %60 : index
// CHECK-NEXT:     %62 = memref_cast %59 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %63 = index_cast %56 : i32 to index
// CHECK-NEXT:     %64 = addi %c0, %63 : index
// CHECK-NEXT:     %65 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %66 = index_cast %51 : i32 to index
// CHECK-NEXT:     %67 = addi %c0, %66 : index
// CHECK-NEXT:     %68 = load %65[%61, %67] : memref<?x2000xf64>
// CHECK-NEXT:     %69 = load %65[%64, %67] : memref<?x2000xf64>
// CHECK-NEXT:     %70 = mulf %68, %69 : f64
// CHECK-NEXT:     %71 = load %62[%61, %64] : memref<?x2000xf64>
// CHECK-NEXT:     %72 = addf %71, %70 : f64
// CHECK-NEXT:     store %72, %62[%61, %64] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %73 = addi %56, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb22(%73 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %74 = addi %53, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb19(%74 : i32)
// CHECK-NEXT:   ^bb25(%75: i32):  // 2 preds: ^bb18, ^bb30
// CHECK-NEXT:     %76 = cmpi "slt", %75, %arg0 : i32
// CHECK-NEXT:     cond_br %76, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     br ^bb28(%c0_i32 : i32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %77 = memref_cast %7 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%77) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb28(%78: i32):  // 2 preds: ^bb26, ^bb29
// CHECK-NEXT:     %79 = cmpi "slt", %78, %arg0 : i32
// CHECK-NEXT:     cond_br %79, ^bb29, ^bb30
// CHECK-NEXT:   ^bb29:  // pred: ^bb28
// CHECK-NEXT:     %80 = index_cast %75 : i32 to index
// CHECK-NEXT:     %81 = addi %c0, %80 : index
// CHECK-NEXT:     %82 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %83 = index_cast %78 : i32 to index
// CHECK-NEXT:     %84 = addi %c0, %83 : index
// CHECK-NEXT:     %85 = load %7[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %86 = memref_cast %85 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %87 = memref_cast %86 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %88 = load %87[%81, %84] : memref<?x2000xf64>
// CHECK-NEXT:     store %88, %82[%81, %84] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %89 = addi %78, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb28(%89 : i32)
// CHECK-NEXT:   ^bb30:  // pred: ^bb28
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %90 = addi %75, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb25(%90 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_cholesky(%arg0: i32, %arg1: memref<2000x2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb12
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%4: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %5 = cmpi "slt", %4, %2 : i32
// CHECK-NEXT:     cond_br %5, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %6 = index_cast %0 : i32 to index
// CHECK-NEXT:     %7 = addi %c0, %6 : index
// CHECK-NEXT:     %8 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %9 = index_cast %2 : i32 to index
// CHECK-NEXT:     %10 = addi %c0, %9 : index
// CHECK-NEXT:     %11 = index_cast %4 : i32 to index
// CHECK-NEXT:     %12 = addi %c0, %11 : index
// CHECK-NEXT:     %13 = load %8[%7, %12] : memref<?x2000xf64>
// CHECK-NEXT:     %14 = load %8[%10, %12] : memref<?x2000xf64>
// CHECK-NEXT:     %15 = mulf %13, %14 : f64
// CHECK-NEXT:     %16 = load %8[%7, %10] : memref<?x2000xf64>
// CHECK-NEXT:     %17 = subf %16, %15 : f64
// CHECK-NEXT:     store %17, %8[%7, %10] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %18 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%18 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %19 = index_cast %0 : i32 to index
// CHECK-NEXT:     %20 = addi %c0, %19 : index
// CHECK-NEXT:     %21 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %22 = index_cast %2 : i32 to index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = load %21[%23, %23] : memref<?x2000xf64>
// CHECK-NEXT:     %25 = load %21[%20, %23] : memref<?x2000xf64>
// CHECK-NEXT:     %26 = divf %25, %24 : f64
// CHECK-NEXT:     store %26, %21[%20, %23] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %27 = addi %2, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb4(%27 : i32)
// CHECK-NEXT:   ^bb10(%28: i32):  // 2 preds: ^bb6, ^bb11
// CHECK-NEXT:     %29 = cmpi "slt", %28, %0 : i32
// CHECK-NEXT:     cond_br %29, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %30 = index_cast %0 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %33 = index_cast %28 : i32 to index
// CHECK-NEXT:     %34 = addi %c0, %33 : index
// CHECK-NEXT:     %35 = load %32[%31, %34] : memref<?x2000xf64>
// CHECK-NEXT:     %36 = load %32[%31, %34] : memref<?x2000xf64>
// CHECK-NEXT:     %37 = mulf %35, %36 : f64
// CHECK-NEXT:     %38 = load %32[%31, %31] : memref<?x2000xf64>
// CHECK-NEXT:     %39 = subf %38, %37 : f64
// CHECK-NEXT:     store %39, %32[%31, %31] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %40 = addi %28, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%40 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %41 = index_cast %0 : i32 to index
// CHECK-NEXT:     %42 = addi %c0, %41 : index
// CHECK-NEXT:     %43 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %44 = load %43[%42, %42] : memref<?x2000xf64>
// CHECK-NEXT:     %45 = sqrt %44 : f64
// CHECK-NEXT:     store %45, %43[%42, %42] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %46 = addi %0, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb1(%46 : i32)
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
// CHECK-NEXT:     %3 = cmpi "sle", %2, %0 : i32
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