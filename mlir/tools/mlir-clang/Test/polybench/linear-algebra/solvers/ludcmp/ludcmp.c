// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - | FileCheck %s --check-prefix EXEC
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

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
  /* not necessary for LU, but using same code as cholesky */
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
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE w;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("x\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %cst = constant 2.000000e+00 : f64
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %4, %arg3[%3] : memref<2000xf64>
// CHECK-NEXT:     store %4, %arg4[%3] : memref<2000xf64>
// CHECK-NEXT:     %5 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     %6 = sitofp %5 : i32 to f64
// CHECK-NEXT:     %7 = divf %6, %0 : f64
// CHECK-NEXT:     %8 = divf %7, %cst : f64
// CHECK-NEXT:     %9 = sitofp %c4_i32 : i32 to f64
// CHECK-NEXT:     %10 = addf %8, %9 : f64
// CHECK-NEXT:     store %10, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb3(%11: i32):  // 2 preds: ^bb1, ^bb10
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg0 : i32
// CHECK-NEXT:     cond_br %12, ^bb5(%c0_i32 : i32), ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %13 = alloc() : memref<2000x2000xf64>
// CHECK-NEXT:     br ^bb11(%c0_i32 : i32)
// CHECK-NEXT:   ^bb5(%14: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %15 = cmpi "sle", %14, %11 : i32
// CHECK-NEXT:     cond_br %15, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %16 = index_cast %11 : i32 to index
// CHECK-NEXT:     %17 = index_cast %14 : i32 to index
// CHECK-NEXT:     %18 = subi %c0_i32, %14 : i32
// CHECK-NEXT:     %19 = remi_signed %18, %arg0 : i32
// CHECK-NEXT:     %20 = sitofp %19 : i32 to f64
// CHECK-NEXT:     %21 = divf %20, %0 : f64
// CHECK-NEXT:     %22 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:     %23 = addf %21, %22 : f64
// CHECK-NEXT:     store %23, %arg1[%16, %17] : memref<2000x2000xf64>
// CHECK-NEXT:     %24 = addi %14, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%24 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %25 = addi %11, %c1_i32 : i32
// CHECK-NEXT:     br ^bb8(%25 : i32)
// CHECK-NEXT:   ^bb8(%26: i32):  // 2 preds: ^bb7, ^bb9
// CHECK-NEXT:     %27 = cmpi "slt", %26, %arg0 : i32
// CHECK-NEXT:     cond_br %27, ^bb9, ^bb10
// CHECK-NEXT:   ^bb9:  // pred: ^bb8
// CHECK-NEXT:     %28 = index_cast %11 : i32 to index
// CHECK-NEXT:     %29 = index_cast %26 : i32 to index
// CHECK-NEXT:     %30 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %30, %arg1[%28, %29] : memref<2000x2000xf64>
// CHECK-NEXT:     %31 = addi %26, %c1_i32 : i32
// CHECK-NEXT:     br ^bb8(%31 : i32)
// CHECK-NEXT:   ^bb10:  // pred: ^bb8
// CHECK-NEXT:     %32 = index_cast %11 : i32 to index
// CHECK-NEXT:     %33 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:     store %33, %arg1[%32, %32] : memref<2000x2000xf64>
// CHECK-NEXT:     br ^bb3(%25 : i32)
// CHECK-NEXT:   ^bb11(%34: i32):  // 2 preds: ^bb4, ^bb14
// CHECK-NEXT:     %35 = cmpi "slt", %34, %arg0 : i32
// CHECK-NEXT:     cond_br %35, ^bb12(%c0_i32 : i32), ^bb15(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12(%36: i32):  // 2 preds: ^bb11, ^bb13
// CHECK-NEXT:     %37 = cmpi "slt", %36, %arg0 : i32
// CHECK-NEXT:     cond_br %37, ^bb13, ^bb14
// CHECK-NEXT:   ^bb13:  // pred: ^bb12
// CHECK-NEXT:     %38 = index_cast %34 : i32 to index
// CHECK-NEXT:     %39 = index_cast %36 : i32 to index
// CHECK-NEXT:     %40 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %40, %13[%38, %39] : memref<2000x2000xf64>
// CHECK-NEXT:     %41 = addi %36, %c1_i32 : i32
// CHECK-NEXT:     br ^bb12(%41 : i32)
// CHECK-NEXT:   ^bb14:  // pred: ^bb12
// CHECK-NEXT:     %42 = addi %34, %c1_i32 : i32
// CHECK-NEXT:     br ^bb11(%42 : i32)
// CHECK-NEXT:   ^bb15(%43: i32):  // 2 preds: ^bb11, ^bb17
// CHECK-NEXT:     %44 = cmpi "slt", %43, %arg0 : i32
// CHECK-NEXT:     cond_br %44, ^bb16(%c0_i32 : i32), ^bb21(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%45: i32):  // 2 preds: ^bb15, ^bb20
// CHECK-NEXT:     %46 = cmpi "slt", %45, %arg0 : i32
// CHECK-NEXT:     cond_br %46, ^bb18(%c0_i32 : i32), ^bb17
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %47 = addi %43, %c1_i32 : i32
// CHECK-NEXT:     br ^bb15(%47 : i32)
// CHECK-NEXT:   ^bb18(%48: i32):  // 2 preds: ^bb16, ^bb19
// CHECK-NEXT:     %49 = cmpi "slt", %48, %arg0 : i32
// CHECK-NEXT:     cond_br %49, ^bb19, ^bb20
// CHECK-NEXT:   ^bb19:  // pred: ^bb18
// CHECK-NEXT:     %50 = index_cast %45 : i32 to index
// CHECK-NEXT:     %51 = index_cast %48 : i32 to index
// CHECK-NEXT:     %52 = index_cast %43 : i32 to index
// CHECK-NEXT:     %53 = load %arg1[%50, %52] : memref<2000x2000xf64>
// CHECK-NEXT:     %54 = load %arg1[%51, %52] : memref<2000x2000xf64>
// CHECK-NEXT:     %55 = mulf %53, %54 : f64
// CHECK-NEXT:     %56 = load %13[%50, %51] : memref<2000x2000xf64>
// CHECK-NEXT:     %57 = addf %56, %55 : f64
// CHECK-NEXT:     store %57, %13[%50, %51] : memref<2000x2000xf64>
// CHECK-NEXT:     %58 = addi %48, %c1_i32 : i32
// CHECK-NEXT:     br ^bb18(%58 : i32)
// CHECK-NEXT:   ^bb20:  // pred: ^bb18
// CHECK-NEXT:     %59 = addi %45, %c1_i32 : i32
// CHECK-NEXT:     br ^bb16(%59 : i32)
// CHECK-NEXT:   ^bb21(%60: i32):  // 2 preds: ^bb15, ^bb25
// CHECK-NEXT:     %61 = cmpi "slt", %60, %arg0 : i32
// CHECK-NEXT:     cond_br %61, ^bb23(%c0_i32 : i32), ^bb22
// CHECK-NEXT:   ^bb22:  // pred: ^bb21
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb23(%62: i32):  // 2 preds: ^bb21, ^bb24
// CHECK-NEXT:     %63 = cmpi "slt", %62, %arg0 : i32
// CHECK-NEXT:     cond_br %63, ^bb24, ^bb25
// CHECK-NEXT:   ^bb24:  // pred: ^bb23
// CHECK-NEXT:     %64 = index_cast %60 : i32 to index
// CHECK-NEXT:     %65 = index_cast %62 : i32 to index
// CHECK-NEXT:     %66 = load %13[%64, %65] : memref<2000x2000xf64>
// CHECK-NEXT:     store %66, %arg1[%64, %65] : memref<2000x2000xf64>
// CHECK-NEXT:     %67 = addi %62, %c1_i32 : i32
// CHECK-NEXT:     br ^bb23(%67 : i32)
// CHECK-NEXT:   ^bb25:  // pred: ^bb23
// CHECK-NEXT:     %68 = addi %60, %c1_i32 : i32
// CHECK-NEXT:     br ^bb21(%68 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb9
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2(%c0_i32 : i32), ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2(%2: i32):  // 2 preds: ^bb1, ^bb6
// CHECK-NEXT:     %3 = cmpi "slt", %2, %0 : i32
// CHECK-NEXT:     cond_br %3, ^bb3, ^bb7(%0 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     %6 = load %arg1[%4, %5] : memref<2000x2000xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32, %6 : i32, f64)
// CHECK-NEXT:   ^bb4(%7: i32, %8: f64):  // 2 preds: ^bb3, ^bb5
// CHECK-NEXT:     %9 = cmpi "slt", %7, %2 : i32
// CHECK-NEXT:     cond_br %9, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %10 = index_cast %7 : i32 to index
// CHECK-NEXT:     %11 = load %arg1[%4, %10] : memref<2000x2000xf64>
// CHECK-NEXT:     %12 = load %arg1[%10, %5] : memref<2000x2000xf64>
// CHECK-NEXT:     %13 = mulf %11, %12 : f64
// CHECK-NEXT:     %14 = subf %8, %13 : f64
// CHECK-NEXT:     %15 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%15, %14 : i32, f64)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %16 = load %arg1[%5, %5] : memref<2000x2000xf64>
// CHECK-NEXT:     %17 = divf %8, %16 : f64
// CHECK-NEXT:     store %17, %arg1[%4, %5] : memref<2000x2000xf64>
// CHECK-NEXT:     %18 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%18 : i32)
// CHECK-NEXT:   ^bb7(%19: i32):  // 2 preds: ^bb2, ^bb12
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg0 : i32
// CHECK-NEXT:     cond_br %20, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %21 = index_cast %0 : i32 to index
// CHECK-NEXT:     %22 = index_cast %19 : i32 to index
// CHECK-NEXT:     %23 = load %arg1[%21, %22] : memref<2000x2000xf64>
// CHECK-NEXT:     br ^bb10(%c0_i32, %23 : i32, f64)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %24 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%24 : i32)
// CHECK-NEXT:   ^bb10(%25: i32, %26: f64):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %27 = cmpi "slt", %25, %0 : i32
// CHECK-NEXT:     cond_br %27, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %28 = index_cast %25 : i32 to index
// CHECK-NEXT:     %29 = load %arg1[%21, %28] : memref<2000x2000xf64>
// CHECK-NEXT:     %30 = load %arg1[%28, %22] : memref<2000x2000xf64>
// CHECK-NEXT:     %31 = mulf %29, %30 : f64
// CHECK-NEXT:     %32 = subf %26, %31 : f64
// CHECK-NEXT:     %33 = addi %25, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%33, %32 : i32, f64)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     store %26, %arg1[%21, %22] : memref<2000x2000xf64>
// CHECK-NEXT:     %34 = addi %19, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%34 : i32)
// CHECK-NEXT:   ^bb13(%35: i32):  // 2 preds: ^bb1, ^bb18
// CHECK-NEXT:     %36 = cmpi "slt", %35, %arg0 : i32
// CHECK-NEXT:     cond_br %36, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %37 = index_cast %35 : i32 to index
// CHECK-NEXT:     %38 = load %arg2[%37] : memref<2000xf64>
// CHECK-NEXT:     br ^bb16(%c0_i32, %38 : i32, f64)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %39 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb19(%39 : i32)
// CHECK-NEXT:   ^bb16(%40: i32, %41: f64):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %42 = cmpi "slt", %40, %35 : i32
// CHECK-NEXT:     cond_br %42, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %43 = index_cast %40 : i32 to index
// CHECK-NEXT:     %44 = load %arg1[%37, %43] : memref<2000x2000xf64>
// CHECK-NEXT:     %45 = load %arg4[%43] : memref<2000xf64>
// CHECK-NEXT:     %46 = mulf %44, %45 : f64
// CHECK-NEXT:     %47 = subf %41, %46 : f64
// CHECK-NEXT:     %48 = addi %40, %c1_i32 : i32
// CHECK-NEXT:     br ^bb16(%48, %47 : i32, f64)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     store %41, %arg4[%37] : memref<2000xf64>
// CHECK-NEXT:     %49 = addi %35, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%49 : i32)
// CHECK-NEXT:   ^bb19(%50: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %51 = cmpi "sge", %50, %c0_i32 : i32
// CHECK-NEXT:     cond_br %51, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %52 = index_cast %50 : i32 to index
// CHECK-NEXT:     %53 = load %arg4[%52] : memref<2000xf64>
// CHECK-NEXT:     %54 = addi %50, %c1_i32 : i32
// CHECK-NEXT:     br ^bb22(%54, %53 : i32, f64)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb22(%55: i32, %56: f64):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %57 = cmpi "slt", %55, %arg0 : i32
// CHECK-NEXT:     cond_br %57, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %58 = index_cast %55 : i32 to index
// CHECK-NEXT:     %59 = load %arg1[%52, %58] : memref<2000x2000xf64>
// CHECK-NEXT:     %60 = load %arg3[%58] : memref<2000xf64>
// CHECK-NEXT:     %61 = mulf %59, %60 : f64
// CHECK-NEXT:     %62 = subf %56, %61 : f64
// CHECK-NEXT:     %63 = addi %55, %c1_i32 : i32
// CHECK-NEXT:     br ^bb22(%63, %62 : i32, f64)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %64 = load %arg1[%52, %52] : memref<2000x2000xf64>
// CHECK-NEXT:     %65 = divf %56, %64 : f64
// CHECK-NEXT:     store %65, %arg3[%52] : memref<2000xf64>
// CHECK-NEXT:     %66 = subi %50, %c1_i32 : i32
// CHECK-NEXT:     br ^bb19(%66 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @[[str0]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @[[str1]] : !llvm.ptr<array<15 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %14, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %15 = remi_signed %13, %c20_i32 : i32
// CHECK-NEXT:     %16 = cmpi "eq", %15, %c0_i32 : i32
// CHECK-NEXT:     scf.if %16 {
// CHECK-NEXT:       %38 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %39 = llvm.load %38 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %40 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %41 = llvm.getelementptr %40[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %42 = llvm.call @fprintf(%39, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %17 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %18 = llvm.load %17 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %19 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = index_cast %13 : i32 to index
// CHECK-NEXT:     %22 = load %arg1[%21] : memref<2000xf64>
// CHECK-NEXT:     %23 = llvm.mlir.cast %22 : f64 to !llvm.double
// CHECK-NEXT:     %24 = llvm.call @fprintf(%18, %20, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %25 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%25 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %26 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %27 = llvm.load %26 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %28 = llvm.mlir.addressof @[[str5]] : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %29 = llvm.getelementptr %28[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %30 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %31 = llvm.getelementptr %30[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %32 = llvm.call @fprintf(%27, %29, %31) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @[[str6]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}