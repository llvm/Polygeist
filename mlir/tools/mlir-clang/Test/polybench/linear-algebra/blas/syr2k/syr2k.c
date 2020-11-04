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
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
    }
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
void kernel_syr2k(int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++)
      for (j = 0; j <= i; j++)
	{
	  C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
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
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syr2k (n, m,
		alpha, beta,
		POLYBENCH_ARRAY(C),
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("C")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
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
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%10, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %15 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %16 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %17 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %20 = load %12[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     %23 = load %14[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1000_i32, %15, %16, %19, %22, %25) : (i32, i32, memref<?xf64>, memref<?xf64>, memref<1200x1200xf64>, memref<1200x1000xf64>, memref<1200x1000xf64>) -> ()
// CHECK-NEXT:     %26 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %27 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %28 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %31 = load %12[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     %34 = load %14[%c0] : memref<?xmemref<1200x1000xf64>>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<?x1000xf64> to memref<1200x1000xf64>
// CHECK-NEXT:     call @kernel_syr2k(%c1200_i32, %c1000_i32, %26, %27, %30, %33, %36) : (i32, i32, f64, f64, memref<1200x1200xf64>, memref<1200x1000xf64>, memref<1200x1000xf64>) -> ()
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
// CHECK-NEXT:       %49 = load %7[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:       %50 = memref_cast %49 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %51 = memref_cast %50 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:       call @print_array(%c1200_i32, %51) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %46 = memref_cast %7 : memref<?xmemref<1200x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%46) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %47 = memref_cast %12 : memref<?xmemref<1200x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%47) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %48 = memref_cast %14 : memref<?xmemref<1200x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%48) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>, %arg6: memref<1200x1000xf64>) {
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
// CHECK-NEXT:     %15 = memref_cast %arg6 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %16 = addi %9, %c2_i32 : i32
// CHECK-NEXT:     %17 = remi_signed %16, %arg1 : i32
// CHECK-NEXT:     %18 = sitofp %17 : i32 to f64
// CHECK-NEXT:     %19 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %20 = divf %18, %19 : f64
// CHECK-NEXT:     store %20, %15[%5, %8] : memref<?x1000xf64>
// CHECK-NEXT:     %21 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%21 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%22 : i32)
// CHECK-NEXT:   ^bb7(%23: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg0 : i32
// CHECK-NEXT:     cond_br %24, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%25: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %26 = cmpi "slt", %25, %arg0 : i32
// CHECK-NEXT:     cond_br %26, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %27 = index_cast %23 : i32 to index
// CHECK-NEXT:     %28 = addi %c0, %27 : index
// CHECK-NEXT:     %29 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %30 = index_cast %25 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = muli %23, %25 : i32
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %33 = addi %32, %c3_i32 : i32
// CHECK-NEXT:     %34 = remi_signed %33, %arg0 : i32
// CHECK-NEXT:     %35 = sitofp %34 : i32 to f64
// CHECK-NEXT:     %36 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %37 = divf %35, %36 : f64
// CHECK-NEXT:     store %37, %29[%28, %31] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %38 = addi %25, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%38 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %39 = addi %23, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%39 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>, %arg6: memref<1200x1000xf64>) {
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
// CHECK-NEXT:     %25 = load %22[%21, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %26 = mulf %25, %arg2 : f64
// CHECK-NEXT:     %27 = memref_cast %arg6 : memref<1200x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %28 = load %27[%18, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %29 = mulf %26, %28 : f64
// CHECK-NEXT:     %30 = load %27[%21, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %31 = mulf %30, %arg2 : f64
// CHECK-NEXT:     %32 = load %22[%18, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %33 = mulf %31, %32 : f64
// CHECK-NEXT:     %34 = addf %29, %33 : f64
// CHECK-NEXT:     %35 = load %19[%18, %21] : memref<?x1200xf64>
// CHECK-NEXT:     %36 = addf %35, %34 : f64
// CHECK-NEXT:     store %36, %19[%18, %21] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %37 = addi %15, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%37 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %38 = addi %12, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%38 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %1 = llvm.mlir.addressof @str0 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %3 = llvm.getelementptr %1[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %4 = llvm.call @fprintf(%0, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %5 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %6 = llvm.mlir.addressof @str1 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %7 = llvm.getelementptr %6[%2, %2] : (!llvm.ptr<array<14 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.call @fprintf(%5, %7, %9) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg0 : i32
// CHECK-NEXT:     cond_br %12, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %13 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %14 = llvm.mlir.addressof @str5 : !llvm.ptr<array<16 x i8>>
// CHECK-NEXT:     %15 = llvm.getelementptr %14[%2, %2] : (!llvm.ptr<array<16 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str2 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %18 = llvm.call @fprintf(%13, %15, %17) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %19 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %20 = llvm.mlir.addressof @str6 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %21 = llvm.getelementptr %20[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %22 = llvm.call @fprintf(%19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%23: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg0 : i32
// CHECK-NEXT:     cond_br %24, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %25 = muli %11, %arg0 : i32
// CHECK-NEXT:     %26 = addi %25, %23 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %27 = remi_signed %26, %c20_i32 : i32
// CHECK-NEXT:     %28 = cmpi "eq", %27, %c0_i32 : i32
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       %42 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:       %43 = llvm.mlir.addressof @str3 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %44 = llvm.getelementptr %43[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %45 = llvm.call @fprintf(%42, %44) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %29 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %30 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %31 = llvm.getelementptr %30[%2, %2] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %32 = index_cast %11 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = memref_cast %arg1 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %35 = index_cast %23 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %34[%33, %36] : memref<?x1200xf64>
// CHECK-NEXT:     %38 = llvm.mlir.cast %37 : f64 to !llvm.double
// CHECK-NEXT:     %39 = llvm.call @fprintf(%29, %31, %38) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %40 = addi %23, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%40 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %41 = addi %11, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%41 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }