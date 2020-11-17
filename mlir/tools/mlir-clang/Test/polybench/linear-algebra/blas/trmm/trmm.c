// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | opt -O3 -S | lli - &> %s.out2
// RUN: rm -f %s.exec1
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trmm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
#pragma scop
  for (i = 0; i < _PB_M; i++)
     for (j = 0; j < _PB_N; j++) {
        for (k = i+1; k < _PB_M; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
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
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm (m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("B\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %cst_0 = constant 1.000000e+00 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     store %cst, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb8
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %3 = cmpi "slt", %2, %0 : i32
// CHECK-NEXT:     cond_br %3, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     %6 = addi %0, %2 : i32
// CHECK-NEXT:     %7 = remi_signed %6, %arg0 : i32
// CHECK-NEXT:     %8 = sitofp %7 : i32 to f64
// CHECK-NEXT:     %9 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %10 = divf %8, %9 : f64
// CHECK-NEXT:     store %10, %arg3[%4, %5] : memref<1000x1000xf64>
// CHECK-NEXT:     %11 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%11 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %12 = index_cast %0 : i32 to index
// CHECK-NEXT:     store %cst_0, %arg3[%12, %12] : memref<1000x1000xf64>
// CHECK-NEXT:     br ^bb6(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6(%13: i32):  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg1 : i32
// CHECK-NEXT:     cond_br %14, ^bb7, ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %15 = index_cast %13 : i32 to index
// CHECK-NEXT:     %16 = subi %0, %13 : i32
// CHECK-NEXT:     %17 = addi %arg1, %16 : i32
// CHECK-NEXT:     %18 = remi_signed %17, %arg1 : i32
// CHECK-NEXT:     %19 = sitofp %18 : i32 to f64
// CHECK-NEXT:     %20 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %21 = divf %19, %20 : f64
// CHECK-NEXT:     store %21, %arg4[%12, %15] : memref<1000x1200xf64>
// CHECK-NEXT:     %22 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb6(%22 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-NEXT:     %23 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%23 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb8
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb6(%4 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %5 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb6(%6: i32):  // 2 preds: ^bb4, ^bb7
// CHECK-NEXT:     %7 = cmpi "slt", %6, %arg0 : i32
// CHECK-NEXT:     cond_br %7, ^bb7, ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %8 = index_cast %0 : i32 to index
// CHECK-NEXT:     %9 = index_cast %2 : i32 to index
// CHECK-NEXT:     %10 = index_cast %6 : i32 to index
// CHECK-NEXT:     %11 = load %arg3[%10, %8] : memref<1000x1000xf64>
// CHECK-NEXT:     %12 = load %arg4[%10, %9] : memref<1000x1200xf64>
// CHECK-NEXT:     %13 = mulf %11, %12 : f64
// CHECK-NEXT:     %14 = load %arg4[%8, %9] : memref<1000x1200xf64>
// CHECK-NEXT:     %15 = addf %14, %13 : f64
// CHECK-NEXT:     store %15, %arg4[%8, %9] : memref<1000x1200xf64>
// CHECK-NEXT:     %16 = addi %6, %c1_i32 : i32
// CHECK-NEXT:     br ^bb6(%16 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-NEXT:     %17 = index_cast %0 : i32 to index
// CHECK-NEXT:     %18 = index_cast %2 : i32 to index
// CHECK-NEXT:     %19 = load %arg4[%17, %18] : memref<1000x1200xf64>
// CHECK-NEXT:     %20 = mulf %arg2, %19 : f64
// CHECK-NEXT:     store %20, %arg4[%17, %18] : memref<1000x1200xf64>
// CHECK-NEXT:     %21 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%21 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>) {
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
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %16 = llvm.load %15 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @[[str5]] : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %18 = llvm.getelementptr %17[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %19 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = llvm.call @fprintf(%16, %18, %20) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %22 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %23 = llvm.load %22 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.mlir.addressof @[[str6]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %25 = llvm.getelementptr %24[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %26 = llvm.call @fprintf(%23, %25) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%27: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg1 : i32
// CHECK-NEXT:     cond_br %28, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %29 = muli %13, %arg0 : i32
// CHECK-NEXT:     %30 = addi %29, %27 : i32
// CHECK-NEXT:     %31 = remi_signed %30, %c20_i32 : i32
// CHECK-NEXT:     %32 = cmpi "eq", %31, %c0_i32 : i32
// CHECK-NEXT:     scf.if %32 {
// CHECK-NEXT:       %44 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %45 = llvm.load %44 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %46 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = index_cast %13 : i32 to index
// CHECK-NEXT:     %38 = index_cast %27 : i32 to index
// CHECK-NEXT:     %39 = load %arg2[%37, %38] : memref<1000x1200xf64>
// CHECK-NEXT:     %40 = llvm.mlir.cast %39 : f64 to !llvm.double
// CHECK-NEXT:     %41 = llvm.call @fprintf(%34, %36, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %42 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%42 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %43 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%43 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}