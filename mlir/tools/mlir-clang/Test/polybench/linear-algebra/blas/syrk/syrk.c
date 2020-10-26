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
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop

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

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %2 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = muli %2, %2 : i32
// CHECK-NEXT:     %4 = zexti %3 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %5 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %8 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %9 = muli %2, %8 : i32
// CHECK-NEXT:     %10 = zexti %9 : i32 to i64
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%10, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %13 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %14 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %15 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %18 = load %12[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1000_i32, %13, %14, %17, %20) : (i32, i32, memref<?xf64>, memref<?xf64>, memref<1200x1200xf64>, memref<1200x1000xf64>) -> ()
// CHECK-NEXT:     %21 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %22 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %23 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %26 = load %12[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     call @kernel_syrk(%c1200_i32, %c1000_i32, %21, %22, %25, %28) : (i32, i32, f64, f64, memref<1200x1200xf64>, memref<1200x1000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %29 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %30 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = load %arg1[%31] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %33 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %34 = call @strcmp(%32, %33) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %35 = trunci %34 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %36 = xor %35, %true : i1
// CHECK-NEXT:     %37 = and %29, %36 : i1
// CHECK-NEXT:     scf.if %37 {
// CHECK-NEXT:       %40 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:       %41 = memref_cast %40 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %42 = memref_cast %41 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:       call @print_array(%c1200_i32, %42) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %38 = memref_cast %7 : memref<?xmemref<1200x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%38) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %39 = memref_cast %12 : memref<?xmemref<1200x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%39) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     store %cst, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     %cst_0 = constant 1.200000e+00 : f64
// CHECK-NEXT:     store %cst_0, %arg3[%c0] : memref<?xf64>
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
// CHECK-NEXT:     %6 = memref_cast %arg5 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = muli %0, %2 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %10 = addi %9, %c1_i32 : i32
// CHECK-NEXT:     %11 = remi_signed %10, %arg0 : i32
// CHECK-NEXT:     %12 = sitofp %11 : i32 to f64
// CHECK-NEXT:     %13 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %14 = divf %12, %13 : f64
// CHECK-NEXT:     store %14, %6[%5, %8] : memref<?x1000xf64>
// CHECK-NEXT:     %15 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%15 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%16 : i32)
// CHECK-NEXT:   ^bb7(%17: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg0 : i32
// CHECK-NEXT:     cond_br %18, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%19: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg0 : i32
// CHECK-NEXT:     cond_br %20, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %21 = index_cast %17 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %24 = index_cast %19 : i32 to index
// CHECK-NEXT:     %25 = addi %c0, %24 : index
// CHECK-NEXT:     %26 = muli %17, %19 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %27 = addi %26, %c2_i32 : i32
// CHECK-NEXT:     %28 = remi_signed %27, %arg1 : i32
// CHECK-NEXT:     %29 = sitofp %28 : i32 to f64
// CHECK-NEXT:     %30 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %31 = divf %29, %30 : f64
// CHECK-NEXT:     store %31, %23[%22, %25] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %19, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %33 = addi %17, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%33 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb9
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
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = load %6[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %10 = mulf %9, %arg3 : f64
// CHECK-NEXT:     store %10, %6[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %11 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%11 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%12: i32):  // 2 preds: ^bb6, ^bb12
// CHECK-NEXT:     %13 = cmpi "slt", %12, %arg1 : i32
// CHECK-NEXT:     cond_br %13, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   ^bb10(%15: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %16 = cmpi "sle", %15, %0 : i32
// CHECK-NEXT:     cond_br %16, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %17 = index_cast %0 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %20 = index_cast %15 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = memref_cast %arg5 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %23 = index_cast %12 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %22[%18, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %26 = mulf %arg2, %25 : f64
// CHECK-NEXT:     %27 = load %22[%21, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %28 = mulf %26, %27 : f64
// CHECK-NEXT:     %29 = load %19[%18, %21] : memref<?x1200xf64>
// CHECK-NEXT:     %30 = addf %29, %28 : f64
// CHECK-NEXT:     store %30, %19[%18, %21] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %31 = addi %15, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%31 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %12, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%32 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
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