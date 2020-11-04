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
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      for(i = 0; i < _PB_N; i++)
	for (j = 0; j < _PB_N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%d ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("path")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2800_i32 = constant 2800 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2800_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = zexti %1 : i32 to i64
// CHECK-NEXT:     %c4_i64 = constant 4 : i64
// CHECK-NEXT:     %3 = trunci %c4_i64 : i64 to i32
// CHECK-NEXT:     %4 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %5 = memref_cast %4 : memref<?xi8> to memref<?xmemref<2800x2800xi32>>
// CHECK-NEXT:     %6 = load %5[%c0] : memref<?xmemref<2800x2800xi32>>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?x2800xi32> to memref<2800x2800xi32>
// CHECK-NEXT:     call @init_array(%c2800_i32, %8) : (i32, memref<2800x2800xi32>) -> ()
// CHECK-NEXT:     %9 = load %5[%c0] : memref<?xmemref<2800x2800xi32>>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?x2800xi32> to memref<2800x2800xi32>
// CHECK-NEXT:     call @kernel_floyd_warshall(%c2800_i32, %11) : (i32, memref<2800x2800xi32>) -> ()
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
// CHECK-NEXT:       %22 = load %5[%c0] : memref<?xmemref<2800x2800xi32>>
// CHECK-NEXT:       %23 = memref_cast %22 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:       %24 = memref_cast %23 : memref<?x2800xi32> to memref<2800x2800xi32>
// CHECK-NEXT:       call @print_array(%c2800_i32, %24) : (i32, memref<2800x2800xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %21 = memref_cast %5 : memref<?xmemref<2800x2800xi32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%21) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2800x2800xi32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = alloca() : memref<1xi32>
// CHECK-NEXT:     %1 = alloca() : memref<1xi32>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%2: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     store %c0_i32, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%4: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %5 = cmpi "slt", %4, %arg0 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %6 = index_cast %2 : i32 to index
// CHECK-NEXT:     %7 = addi %c0, %6 : index
// CHECK-NEXT:     %8 = memref_cast %arg1 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:     %9 = index_cast %4 : i32 to index
// CHECK-NEXT:     %10 = addi %c0, %9 : index
// CHECK-NEXT:     %11 = muli %2, %4 : i32
// CHECK-NEXT:     %c7_i32 = constant 7 : i32
// CHECK-NEXT:     %12 = remi_signed %11, %c7_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %13 = addi %12, %c1_i32 : i32
// CHECK-NEXT:     store %13, %8[%7, %10] : memref<?x2800xi32>
// CHECK-NEXT:     %14 = addi %2, %4 : i32
// CHECK-NEXT:     %c13_i32 = constant 13 : i32
// CHECK-NEXT:     %15 = remi_signed %14, %c13_i32 : i32
// CHECK-NEXT:     %16 = cmpi "eq", %15, %c0_i32 : i32
// CHECK-NEXT:     %17 = remi_signed %14, %c7_i32 : i32
// CHECK-NEXT:     %18 = cmpi "eq", %17, %c0_i32 : i32
// CHECK-NEXT:     %19 = or %16, %18 : i1
// CHECK-NEXT:     %c11_i32 = constant 11 : i32
// CHECK-NEXT:     %20 = remi_signed %14, %c11_i32 : i32
// CHECK-NEXT:     %21 = cmpi "eq", %20, %c0_i32 : i32
// CHECK-NEXT:     %22 = or %19, %21 : i1
// CHECK-NEXT:     scf.if %22 {
// CHECK-NEXT:       %25 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %26 = index_cast %25 : i32 to index
// CHECK-NEXT:       %27 = addi %c0, %26 : index
// CHECK-NEXT:       %28 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %29 = index_cast %28 : i32 to index
// CHECK-NEXT:       %30 = addi %c0, %29 : index
// CHECK-NEXT:       %c999_i32 = constant 999 : i32
// CHECK-NEXT:       store %c999_i32, %8[%27, %30] : memref<?x2800xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %23 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     store %23, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%23 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %24 = addi %2, %c1_i32_0 : i32
// CHECK-NEXT:     store %24, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%24 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<2800x2800xi32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = alloca() : memref<1xi32>
// CHECK-NEXT:     %1 = alloca() : memref<1xi32>
// CHECK-NEXT:     %2 = alloca() : memref<1xi32>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     store %c0_i32, %2[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%3: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %4 = cmpi "slt", %3, %arg0 : i32
// CHECK-NEXT:     cond_br %4, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%5: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg0 : i32
// CHECK-NEXT:     cond_br %6, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     store %c0_i32, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     store %7, %2[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%7 : i32)
// CHECK-NEXT:   ^bb7(%8: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %9 = cmpi "slt", %8, %arg0 : i32
// CHECK-NEXT:     cond_br %9, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %10 = index_cast %5 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %arg1 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:     %13 = index_cast %8 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = load %12[%11, %14] : memref<?x2800xi32>
// CHECK-NEXT:     %16 = index_cast %3 : i32 to index
// CHECK-NEXT:     %17 = addi %c0, %16 : index
// CHECK-NEXT:     %18 = load %12[%11, %17] : memref<?x2800xi32>
// CHECK-NEXT:     %19 = load %12[%17, %14] : memref<?x2800xi32>
// CHECK-NEXT:     %20 = addi %18, %19 : i32
// CHECK-NEXT:     %21 = cmpi "slt", %15, %20 : i32
// CHECK-NEXT:     %22 = scf.if %21 -> (i32) {
// CHECK-NEXT:       %25 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %26 = index_cast %25 : i32 to index
// CHECK-NEXT:       %27 = addi %c0, %26 : index
// CHECK-NEXT:       %28 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %29 = index_cast %28 : i32 to index
// CHECK-NEXT:       %30 = addi %c0, %29 : index
// CHECK-NEXT:       %31 = load %12[%27, %30] : memref<?x2800xi32>
// CHECK-NEXT:       scf.yield %31 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %25 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %26 = index_cast %25 : i32 to index
// CHECK-NEXT:       %27 = addi %c0, %26 : index
// CHECK-NEXT:       %28 = load %2[%c0] : memref<1xi32>
// CHECK-NEXT:       %29 = index_cast %28 : i32 to index
// CHECK-NEXT:       %30 = addi %c0, %29 : index
// CHECK-NEXT:       %31 = load %12[%27, %30] : memref<?x2800xi32>
// CHECK-NEXT:       %32 = load %2[%c0] : memref<1xi32>
// CHECK-NEXT:       %33 = index_cast %32 : i32 to index
// CHECK-NEXT:       %34 = addi %c0, %33 : index
// CHECK-NEXT:       %35 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %36 = index_cast %35 : i32 to index
// CHECK-NEXT:       %37 = addi %c0, %36 : index
// CHECK-NEXT:       %38 = load %12[%34, %37] : memref<?x2800xi32>
// CHECK-NEXT:       %39 = addi %31, %38 : i32
// CHECK-NEXT:       scf.yield %39 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     store %22, %12[%11, %14] : memref<?x2800xi32>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %23 = addi %8, %c1_i32_0 : i32
// CHECK-NEXT:     store %23, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%23 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %24 = addi %5, %c1_i32_1 : i32
// CHECK-NEXT:     store %24, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%24 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2800x2800xi32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %1 = llvm.mlir.addressof @str0 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %3 = llvm.getelementptr %1[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %4 = llvm.call @fprintf(%0, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %5 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %6 = llvm.mlir.addressof @str1 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %7 = llvm.getelementptr %6[%2, %2] : (!llvm.ptr<array<14 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%2, %2] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str2 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
// CHECK-NEXT:     %30 = llvm.mlir.addressof @str4 : !llvm.ptr<array<3 x i8>>
// CHECK-NEXT:     %31 = llvm.getelementptr %30[%2, %2] : (!llvm.ptr<array<3 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %32 = index_cast %11 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = memref_cast %arg1 : memref<2800x2800xi32> to memref<?x2800xi32>
// CHECK-NEXT:     %35 = index_cast %23 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %34[%33, %36] : memref<?x2800xi32>
// CHECK-NEXT:     %38 = llvm.mlir.cast %37 : i32 to !llvm.i32
// CHECK-NEXT:     %39 = llvm.call @fprintf(%29, %31, %38) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
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