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
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;

//BLAS PARAMS
//SIDE = 'L'
//UPLO = 'L'
// =>  Form  C := alpha*A*B + beta*C
// A is MxM
// B is MxN
// C is MxN
//note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
#pragma scop
   for (i = 0; i < _PB_M; i++)
      for (j = 0; j < _PB_N; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %2 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %4 = muli %2, %3 : i32
// CHECK-NEXT:     %5 = zexti %4 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %6 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%5, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %9 = muli %2, %2 : i32
// CHECK-NEXT:     %10 = zexti %9 : i32 to i64
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%10, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%5, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %15 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %16 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %17 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %20 = load %12[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     %23 = load %14[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     call @init_array(%c1000_i32, %c1200_i32, %15, %16, %19, %22, %25) : (i32, i32, memref<?xf64>, memref<?xf64>, memref<1000x1200xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     %26 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %27 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %28 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %31 = load %12[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     %34 = load %14[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     call @kernel_symm(%c1000_i32, %c1200_i32, %26, %27, %30, %33, %36) : (i32, i32, f64, f64, memref<1000x1200xf64>, memref<1000x1000xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %37 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %38 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %39 = addi %c0, %38 : index
// CHECK-NEXT:     %40 = load %arg1[%39] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %41 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %42 = call @strcmp(%40, %41) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %43 = trunci %42 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %44 = xor %43, %true : i1
// CHECK-NEXT:     %45 = and %37, %44 : i1
// CHECK-NEXT:     scf.if %45 {
// CHECK-NEXT:       %49 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %50 = memref_cast %49 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %51 = memref_cast %50 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       call @print_array(%c1000_i32, %c1200_i32, %51) : (i32, i32, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %46 = memref_cast %8 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%46) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %47 = memref_cast %12 : memref<?xmemref<1000x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%47) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %48 = memref_cast %14 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%48) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
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
// CHECK-NEXT:     %6 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = addi %0, %2 : i32
// CHECK-NEXT:     %c100_i32 = constant 100 : i32
// CHECK-NEXT:     %10 = remi_signed %9, %c100_i32 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %13 = divf %11, %12 : f64
// CHECK-NEXT:     store %13, %6[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %14 = memref_cast %arg6 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %15 = addi %arg1, %0 : i32
// CHECK-NEXT:     %16 = subi %15, %2 : i32
// CHECK-NEXT:     %17 = remi_signed %16, %c100_i32 : i32
// CHECK-NEXT:     %18 = sitofp %17 : i32 to f64
// CHECK-NEXT:     %19 = divf %18, %12 : f64
// CHECK-NEXT:     store %19, %14[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %20 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%20 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %21 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%21 : i32)
// CHECK-NEXT:   ^bb7(%22: i32):  // 2 preds: ^bb3, ^bb15
// CHECK-NEXT:     %23 = cmpi "slt", %22, %arg0 : i32
// CHECK-NEXT:     cond_br %23, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%24: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %25 = cmpi "sle", %24, %22 : i32
// CHECK-NEXT:     cond_br %25, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %26 = index_cast %22 : i32 to index
// CHECK-NEXT:     %27 = addi %c0, %26 : index
// CHECK-NEXT:     %28 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %29 = index_cast %24 : i32 to index
// CHECK-NEXT:     %30 = addi %c0, %29 : index
// CHECK-NEXT:     %31 = addi %22, %24 : i32
// CHECK-NEXT:     %c100_i32_2 = constant 100 : i32
// CHECK-NEXT:     %32 = remi_signed %31, %c100_i32_2 : i32
// CHECK-NEXT:     %33 = sitofp %32 : i32 to f64
// CHECK-NEXT:     %34 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %35 = divf %33, %34 : f64
// CHECK-NEXT:     store %35, %28[%27, %30] : memref<?x1000xf64>
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %36 = addi %24, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb10(%36 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %37 = addi %22, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb13(%37 : i32)
// CHECK-NEXT:   ^bb13(%38: i32):  // 2 preds: ^bb12, ^bb14
// CHECK-NEXT:     %39 = cmpi "slt", %38, %arg0 : i32
// CHECK-NEXT:     cond_br %39, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %40 = index_cast %22 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %42 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %43 = index_cast %38 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %c999_i32 = constant 999 : i32
// CHECK-NEXT:     %45 = subi %c0_i32, %c999_i32 : i32
// CHECK-NEXT:     %46 = sitofp %45 : i32 to f64
// CHECK-NEXT:     store %46, %42[%41, %44] : memref<?x1000xf64>
// CHECK-NEXT:     %47 = addi %38, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb13(%47 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb7(%37 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64>, %arg6: memref<1000x1200xf64>) {
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     br ^bb7(%c0_i32, %4 : i32, f64)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %5 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb7(%6: i32, %7: f64):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %8 = cmpi "slt", %6, %0 : i32
// CHECK-NEXT:     cond_br %8, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %9 = index_cast %6 : i32 to index
// CHECK-NEXT:     %10 = addi %c0, %9 : index
// CHECK-NEXT:     %11 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %12 = index_cast %2 : i32 to index
// CHECK-NEXT:     %13 = addi %c0, %12 : index
// CHECK-NEXT:     %14 = index_cast %0 : i32 to index
// CHECK-NEXT:     %15 = addi %c0, %14 : index
// CHECK-NEXT:     %16 = memref_cast %arg6 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %17 = load %16[%15, %13] : memref<?x1200xf64>
// CHECK-NEXT:     %18 = mulf %arg2, %17 : f64
// CHECK-NEXT:     %19 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %20 = load %19[%15, %10] : memref<?x1000xf64>
// CHECK-NEXT:     %21 = mulf %18, %20 : f64
// CHECK-NEXT:     %22 = load %11[%10, %13] : memref<?x1200xf64>
// CHECK-NEXT:     %23 = addf %22, %21 : f64
// CHECK-NEXT:     store %23, %11[%10, %13] : memref<?x1200xf64>
// CHECK-NEXT:     %24 = load %16[%10, %13] : memref<?x1200xf64>
// CHECK-NEXT:     %25 = load %19[%15, %10] : memref<?x1000xf64>
// CHECK-NEXT:     %26 = mulf %24, %25 : f64
// CHECK-NEXT:     %27 = addf %7, %26 : f64
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %28 = addi %6, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%28, %27 : i32, f64)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %29 = index_cast %0 : i32 to index
// CHECK-NEXT:     %30 = addi %c0, %29 : index
// CHECK-NEXT:     %31 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %32 = index_cast %2 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = load %31[%30, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %35 = mulf %arg3, %34 : f64
// CHECK-NEXT:     %36 = memref_cast %arg6 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %37 = load %36[%30, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %38 = mulf %arg2, %37 : f64
// CHECK-NEXT:     %39 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %40 = load %39[%30, %30] : memref<?x1000xf64>
// CHECK-NEXT:     %41 = mulf %38, %40 : f64
// CHECK-NEXT:     %42 = addf %35, %41 : f64
// CHECK-NEXT:     %43 = mulf %arg2, %7 : f64
// CHECK-NEXT:     %44 = addf %42, %43 : f64
// CHECK-NEXT:     store %44, %31[%30, %33] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %45 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%45 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>) {
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
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