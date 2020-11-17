// RUN: mlir-clang %s %stdinclude | FileCheck %s
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;

#pragma scop
    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < _PB_N-1; i++) {
            for (j = 1; j < _PB_N-1; j++) {
                for (k = 1; k < _PB_N-1; k++) {
                    B[i][j][k] =   SCALAR_VAL(0.125) * (A[i+1][j][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i-1][j][k])
                                 + SCALAR_VAL(0.125) * (A[i][j+1][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j-1][k])
                                 + SCALAR_VAL(0.125) * (A[i][j][k+1] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (i = 1; i < _PB_N-1; i++) {
           for (j = 1; j < _PB_N-1; j++) {
               for (k = 1; k < _PB_N-1; k++) {
                   A[i][j][k] =   SCALAR_VAL(0.125) * (B[i+1][j][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i-1][j][k])
                                + SCALAR_VAL(0.125) * (B[i][j+1][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j-1][k])
                                + SCALAR_VAL(0.125) * (B[i][j][k+1] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: memref<120x120x120xf64>, %arg2: memref<120x120x120xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c10_i32 = constant 10 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb7
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5(%c0_i32 : i32), ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb5(%5: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg0 : i32
// CHECK-NEXT:     cond_br %6, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %7 = index_cast %0 : i32 to index
// CHECK-NEXT:     %8 = index_cast %2 : i32 to index
// CHECK-NEXT:     %9 = index_cast %5 : i32 to index
// CHECK-NEXT:     %10 = addi %0, %2 : i32
// CHECK-NEXT:     %11 = subi %arg0, %5 : i32
// CHECK-NEXT:     %12 = addi %10, %11 : i32
// CHECK-NEXT:     %13 = sitofp %12 : i32 to f64
// CHECK-NEXT:     %14 = sitofp %c10_i32 : i32 to f64
// CHECK-NEXT:     %15 = mulf %13, %14 : f64
// CHECK-NEXT:     %16 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %17 = divf %15, %16 : f64
// CHECK-NEXT:     store %17, %arg2[%7, %8, %9] : memref<120x120x120xf64>
// CHECK-NEXT:     %18 = load %arg2[%7, %8, %9] : memref<120x120x120xf64>
// CHECK-NEXT:     store %18, %arg1[%7, %8, %9] : memref<120x120x120xf64>
// CHECK-NEXT:     %19 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%19 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %20 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%20 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %cst = constant 1.250000e-01 : f64
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:     br ^bb1(%c1_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb10
// CHECK-NEXT:     %1 = cmpi "sle", %0, %c500_i32 : i32
// CHECK-NEXT:     cond_br %1, ^bb3(%c1_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb5
// CHECK-NEXT:     %3 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     %4 = cmpi "slt", %2, %3 : i32
// CHECK-NEXT:     cond_br %4, ^bb4(%c1_i32 : i32), ^bb9(%c1_i32 : i32)
// CHECK-NEXT:   ^bb4(%5: i32):  // 2 preds: ^bb3, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %3 : i32
// CHECK-NEXT:     cond_br %6, ^bb6(%c1_i32 : i32), ^bb5
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %7 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%7 : i32)
// CHECK-NEXT:   ^bb6(%8: i32):  // 2 preds: ^bb4, ^bb7
// CHECK-NEXT:     %9 = cmpi "slt", %8, %3 : i32
// CHECK-NEXT:     cond_br %9, ^bb7, ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %10 = index_cast %2 : i32 to index
// CHECK-NEXT:     %11 = index_cast %5 : i32 to index
// CHECK-NEXT:     %12 = index_cast %8 : i32 to index
// CHECK-NEXT:     %13 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     %14 = index_cast %13 : i32 to index
// CHECK-NEXT:     %15 = load %arg2[%14, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %16 = load %arg2[%10, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %17 = mulf %cst_0, %16 : f64
// CHECK-NEXT:     %18 = subf %15, %17 : f64
// CHECK-NEXT:     %19 = subi %2, %c1_i32 : i32
// CHECK-NEXT:     %20 = index_cast %19 : i32 to index
// CHECK-NEXT:     %21 = load %arg2[%20, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %22 = addf %18, %21 : f64
// CHECK-NEXT:     %23 = mulf %cst, %22 : f64
// CHECK-NEXT:     %24 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     %25 = index_cast %24 : i32 to index
// CHECK-NEXT:     %26 = load %arg2[%10, %25, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %27 = load %arg2[%10, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %28 = mulf %cst_0, %27 : f64
// CHECK-NEXT:     %29 = subf %26, %28 : f64
// CHECK-NEXT:     %30 = subi %5, %c1_i32 : i32
// CHECK-NEXT:     %31 = index_cast %30 : i32 to index
// CHECK-NEXT:     %32 = load %arg2[%10, %31, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %33 = addf %29, %32 : f64
// CHECK-NEXT:     %34 = mulf %cst, %33 : f64
// CHECK-NEXT:     %35 = addf %23, %34 : f64
// CHECK-NEXT:     %36 = addi %8, %c1_i32 : i32
// CHECK-NEXT:     %37 = index_cast %36 : i32 to index
// CHECK-NEXT:     %38 = load %arg2[%10, %11, %37] : memref<120x120x120xf64>
// CHECK-NEXT:     %39 = load %arg2[%10, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %40 = mulf %cst_0, %39 : f64
// CHECK-NEXT:     %41 = subf %38, %40 : f64
// CHECK-NEXT:     %42 = subi %8, %c1_i32 : i32
// CHECK-NEXT:     %43 = index_cast %42 : i32 to index
// CHECK-NEXT:     %44 = load %arg2[%10, %11, %43] : memref<120x120x120xf64>
// CHECK-NEXT:     %45 = addf %41, %44 : f64
// CHECK-NEXT:     %46 = mulf %cst, %45 : f64
// CHECK-NEXT:     %47 = addf %35, %46 : f64
// CHECK-NEXT:     %48 = load %arg2[%10, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     %49 = addf %47, %48 : f64
// CHECK-NEXT:     store %49, %arg3[%10, %11, %12] : memref<120x120x120xf64>
// CHECK-NEXT:     br ^bb6(%36 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-NEXT:     %50 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%50 : i32)
// CHECK-NEXT:   ^bb9(%51: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %52 = cmpi "slt", %51, %3 : i32
// CHECK-NEXT:     cond_br %52, ^bb11(%c1_i32 : i32), ^bb10
// CHECK-NEXT:   ^bb10:  // pred: ^bb9
// CHECK-NEXT:     %53 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%53 : i32)
// CHECK-NEXT:   ^bb11(%54: i32):  // 2 preds: ^bb9, ^bb15
// CHECK-NEXT:     %55 = cmpi "slt", %54, %3 : i32
// CHECK-NEXT:     cond_br %55, ^bb13(%c1_i32 : i32), ^bb12
// CHECK-NEXT:   ^bb12:  // pred: ^bb11
// CHECK-NEXT:     %56 = addi %51, %c1_i32 : i32
// CHECK-NEXT:     br ^bb9(%56 : i32)
// CHECK-NEXT:   ^bb13(%57: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %58 = cmpi "slt", %57, %3 : i32
// CHECK-NEXT:     cond_br %58, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %59 = index_cast %51 : i32 to index
// CHECK-NEXT:     %60 = index_cast %54 : i32 to index
// CHECK-NEXT:     %61 = index_cast %57 : i32 to index
// CHECK-NEXT:     %62 = addi %51, %c1_i32 : i32
// CHECK-NEXT:     %63 = index_cast %62 : i32 to index
// CHECK-NEXT:     %64 = load %arg3[%63, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %65 = load %arg3[%59, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %66 = mulf %cst_0, %65 : f64
// CHECK-NEXT:     %67 = subf %64, %66 : f64
// CHECK-NEXT:     %68 = subi %51, %c1_i32 : i32
// CHECK-NEXT:     %69 = index_cast %68 : i32 to index
// CHECK-NEXT:     %70 = load %arg3[%69, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %71 = addf %67, %70 : f64
// CHECK-NEXT:     %72 = mulf %cst, %71 : f64
// CHECK-NEXT:     %73 = addi %54, %c1_i32 : i32
// CHECK-NEXT:     %74 = index_cast %73 : i32 to index
// CHECK-NEXT:     %75 = load %arg3[%59, %74, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %76 = load %arg3[%59, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %77 = mulf %cst_0, %76 : f64
// CHECK-NEXT:     %78 = subf %75, %77 : f64
// CHECK-NEXT:     %79 = subi %54, %c1_i32 : i32
// CHECK-NEXT:     %80 = index_cast %79 : i32 to index
// CHECK-NEXT:     %81 = load %arg3[%59, %80, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %82 = addf %78, %81 : f64
// CHECK-NEXT:     %83 = mulf %cst, %82 : f64
// CHECK-NEXT:     %84 = addf %72, %83 : f64
// CHECK-NEXT:     %85 = addi %57, %c1_i32 : i32
// CHECK-NEXT:     %86 = index_cast %85 : i32 to index
// CHECK-NEXT:     %87 = load %arg3[%59, %60, %86] : memref<120x120x120xf64>
// CHECK-NEXT:     %88 = load %arg3[%59, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %89 = mulf %cst_0, %88 : f64
// CHECK-NEXT:     %90 = subf %87, %89 : f64
// CHECK-NEXT:     %91 = subi %57, %c1_i32 : i32
// CHECK-NEXT:     %92 = index_cast %91 : i32 to index
// CHECK-NEXT:     %93 = load %arg3[%59, %60, %92] : memref<120x120x120xf64>
// CHECK-NEXT:     %94 = addf %90, %93 : f64
// CHECK-NEXT:     %95 = mulf %cst, %94 : f64
// CHECK-NEXT:     %96 = addf %84, %95 : f64
// CHECK-NEXT:     %97 = load %arg3[%59, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     %98 = addf %96, %97 : f64
// CHECK-NEXT:     store %98, %arg2[%59, %60, %61] : memref<120x120x120xf64>
// CHECK-NEXT:     br ^bb13(%85 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %99 = addi %54, %c1_i32 : i32
// CHECK-NEXT:     br ^bb11(%99 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<120x120x120xf64>) {
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
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb4
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
// CHECK-NEXT:   ^bb3(%27: i32):  // 2 preds: ^bb1, ^bb7
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg0 : i32
// CHECK-NEXT:     cond_br %28, ^bb5(%c0_i32 : i32), ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %29 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%29 : i32)
// CHECK-NEXT:   ^bb5(%30: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %31 = cmpi "slt", %30, %arg0 : i32
// CHECK-NEXT:     cond_br %31, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %32 = muli %13, %arg0 : i32
// CHECK-NEXT:     %33 = muli %32, %arg0 : i32
// CHECK-NEXT:     %34 = muli %27, %arg0 : i32
// CHECK-NEXT:     %35 = addi %33, %34 : i32
// CHECK-NEXT:     %36 = addi %35, %30 : i32
// CHECK-NEXT:     %37 = remi_signed %36, %c20_i32 : i32
// CHECK-NEXT:     %38 = cmpi "eq", %37, %c0_i32 : i32
// CHECK-NEXT:     scf.if %38 {
// CHECK-NEXT:       %51 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %52 = llvm.load %51 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %53 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %54 = llvm.getelementptr %53[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %55 = llvm.call @fprintf(%52, %54) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %39 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %40 = llvm.load %39 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %41 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %42 = llvm.getelementptr %41[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %43 = index_cast %13 : i32 to index
// CHECK-NEXT:     %44 = index_cast %27 : i32 to index
// CHECK-NEXT:     %45 = index_cast %30 : i32 to index
// CHECK-NEXT:     %46 = load %arg1[%43, %44, %45] : memref<120x120x120xf64>
// CHECK-NEXT:     %47 = llvm.mlir.cast %46 : f64 to !llvm.double
// CHECK-NEXT:     %48 = llvm.call @fprintf(%40, %42, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %49 = addi %30, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%49 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %50 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%50 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }