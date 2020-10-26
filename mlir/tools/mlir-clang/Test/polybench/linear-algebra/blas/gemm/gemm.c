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
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

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
// CHECK-NEXT:     %c1100_i32 = constant 1100 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %2 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = addi %c1100_i32, %c0_i32 : i32
// CHECK-NEXT:     %4 = muli %2, %3 : i32
// CHECK-NEXT:     %5 = zexti %4 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %6 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%5, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<1000x1100xf64>>
// CHECK-NEXT:     %9 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %10 = muli %2, %9 : i32
// CHECK-NEXT:     %11 = zexti %10 : i32 to i64
// CHECK-NEXT:     %12 = call @polybench_alloc_data(%11, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %14 = muli %9, %3 : i32
// CHECK-NEXT:     %15 = zexti %14 : i32 to i64
// CHECK-NEXT:     %16 = call @polybench_alloc_data(%15, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?xi8> to memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %18 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %19 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %20 = load %8[%c0] : memref<?xmemref<1000x1100xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x1100xf64> to memref<1000x1100xf64>
// CHECK-NEXT:     %23 = load %13[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %26 = load %17[%c0] : memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<?x1100xf64> to memref<1200x1100xf64>
// CHECK-NEXT:     call @init_array(%c1000_i32, %c1100_i32, %c1200_i32, %18, %19, %22, %25, %28) : (i32, i32, i32, memref<?xf64>, memref<?xf64>, memref<1000x1100xf64>, memref<1000x1200xf64>, memref<1200x1100xf64>) -> ()
// CHECK-NEXT:     %29 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %30 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %31 = load %8[%c0] : memref<?xmemref<1000x1100xf64>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?x1100xf64> to memref<1000x1100xf64>
// CHECK-NEXT:     %34 = load %13[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %37 = load %17[%c0] : memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %38 = memref_cast %37 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %39 = memref_cast %38 : memref<?x1100xf64> to memref<1200x1100xf64>
// CHECK-NEXT:     call @kernel_gemm(%c1000_i32, %c1100_i32, %c1200_i32, %29, %30, %33, %36, %39) : (i32, i32, i32, f64, f64, memref<1000x1100xf64>, memref<1000x1200xf64>, memref<1200x1100xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %40 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %41 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %42 = addi %c0, %41 : index
// CHECK-NEXT:     %43 = load %arg1[%42] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %44 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %45 = call @strcmp(%43, %44) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %46 = trunci %45 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %47 = xor %46, %true : i1
// CHECK-NEXT:     %48 = and %40, %47 : i1
// CHECK-NEXT:     scf.if %48 {
// CHECK-NEXT:       %52 = load %8[%c0] : memref<?xmemref<1000x1100xf64>>
// CHECK-NEXT:       %53 = memref_cast %52 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:       %54 = memref_cast %53 : memref<?x1100xf64> to memref<1000x1100xf64>
// CHECK-NEXT:       call @print_array(%c1000_i32, %c1100_i32, %54) : (i32, i32, memref<1000x1100xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %49 = memref_cast %8 : memref<?xmemref<1000x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%49) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %50 = memref_cast %13 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%50) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %51 = memref_cast %17 : memref<?xmemref<1200x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%51) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     store %cst, %arg3[%c0] : memref<?xf64>
// CHECK-NEXT:     %cst_0 = constant 1.200000e+00 : f64
// CHECK-NEXT:     store %cst_0, %arg4[%c0] : memref<?xf64>
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
// CHECK-NEXT:     %6 = memref_cast %arg5 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = muli %0, %2 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %10 = addi %9, %c1_i32 : i32
// CHECK-NEXT:     %11 = remi_signed %10, %arg0 : i32
// CHECK-NEXT:     %12 = sitofp %11 : i32 to f64
// CHECK-NEXT:     %13 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %14 = divf %12, %13 : f64
// CHECK-NEXT:     store %14, %6[%5, %8] : memref<?x1100xf64>
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
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%19: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg2 : i32
// CHECK-NEXT:     cond_br %20, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %21 = index_cast %17 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = memref_cast %arg6 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %24 = index_cast %19 : i32 to index
// CHECK-NEXT:     %25 = addi %c0, %24 : index
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %19, %c1_i32_2 : i32
// CHECK-NEXT:     %27 = muli %17, %26 : i32
// CHECK-NEXT:     %28 = remi_signed %27, %arg2 : i32
// CHECK-NEXT:     %29 = sitofp %28 : i32 to f64
// CHECK-NEXT:     %30 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %31 = divf %29, %30 : f64
// CHECK-NEXT:     store %31, %23[%22, %25] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb10(%26 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %17, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%32 : i32)
// CHECK-NEXT:   ^bb13(%33: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %34 = cmpi "slt", %33, %arg2 : i32
// CHECK-NEXT:     cond_br %34, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb16(%35: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %36 = cmpi "slt", %35, %arg1 : i32
// CHECK-NEXT:     cond_br %36, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %37 = index_cast %33 : i32 to index
// CHECK-NEXT:     %38 = addi %c0, %37 : index
// CHECK-NEXT:     %39 = memref_cast %arg7 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %40 = index_cast %35 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %42 = addi %35, %c2_i32 : i32
// CHECK-NEXT:     %43 = muli %33, %42 : i32
// CHECK-NEXT:     %44 = remi_signed %43, %arg1 : i32
// CHECK-NEXT:     %45 = sitofp %44 : i32 to f64
// CHECK-NEXT:     %46 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %47 = divf %45, %46 : f64
// CHECK-NEXT:     store %47, %39[%38, %41] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %48 = addi %35, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb16(%48 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %49 = addi %33, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb13(%49 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg5 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = load %6[%5, %8] : memref<?x1100xf64>
// CHECK-NEXT:     %10 = mulf %9, %arg4 : f64
// CHECK-NEXT:     store %10, %6[%5, %8] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %11 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%11 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%12: i32):  // 2 preds: ^bb6, ^bb12
// CHECK-NEXT:     %13 = cmpi "slt", %12, %arg2 : i32
// CHECK-NEXT:     cond_br %13, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   ^bb10(%15: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %16 = cmpi "slt", %15, %arg1 : i32
// CHECK-NEXT:     cond_br %16, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %17 = index_cast %0 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = memref_cast %arg5 : memref<1000x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %20 = index_cast %15 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = memref_cast %arg6 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %23 = index_cast %12 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %22[%18, %24] : memref<?x1200xf64>
// CHECK-NEXT:     %26 = mulf %arg3, %25 : f64
// CHECK-NEXT:     %27 = memref_cast %arg7 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %28 = load %27[%24, %21] : memref<?x1100xf64>
// CHECK-NEXT:     %29 = mulf %26, %28 : f64
// CHECK-NEXT:     %30 = load %19[%18, %21] : memref<?x1100xf64>
// CHECK-NEXT:     %31 = addf %30, %29 : f64
// CHECK-NEXT:     store %31, %19[%18, %21] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %15, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %33 = addi %12, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%33 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1100xf64>) {
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