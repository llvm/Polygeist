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
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    y[i] = 0;
  for (i = 0; i < _PB_M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("y")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1900_i32 = constant 1900 : i32
// CHECK-NEXT:     %c2100_i32 = constant 2100 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c1900_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c2100_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<1900x2100xf64>>
// CHECK-NEXT:     %7 = zexti %1 : i32 to i64
// CHECK-NEXT:     %8 = call @polybench_alloc_data(%7, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xi8> to memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%7, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %12 = zexti %0 : i32 to i64
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%12, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %15 = load %6[%c0] : memref<?xmemref<1900x2100xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<1900x2100xf64> to memref<?x2100xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x2100xf64> to memref<1900x2100xf64>
// CHECK-NEXT:     %18 = load %9[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     call @init_array(%c1900_i32, %c2100_i32, %17, %20) : (i32, i32, memref<1900x2100xf64>, memref<2100xf64>) -> ()
// CHECK-NEXT:     %21 = load %6[%c0] : memref<?xmemref<1900x2100xf64>>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<1900x2100xf64> to memref<?x2100xf64>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<?x2100xf64> to memref<1900x2100xf64>
// CHECK-NEXT:     %24 = load %9[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     %27 = load %11[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     %30 = load %14[%c0] : memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<1900xf64> to memref<?xf64>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<?xf64> to memref<1900xf64>
// CHECK-NEXT:     call @kernel_atax(%c1900_i32, %c2100_i32, %23, %26, %29, %32) : (i32, i32, memref<1900x2100xf64>, memref<2100xf64>, memref<2100xf64>, memref<1900xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %33 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %34 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %35 = addi %c0, %34 : index
// CHECK-NEXT:     %36 = load %arg1[%35] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %37 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %38 = call @strcmp(%36, %37) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %39 = trunci %38 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %40 = xor %39, %true : i1
// CHECK-NEXT:     %41 = and %33, %40 : i1
// CHECK-NEXT:     scf.if %41 {
// CHECK-NEXT:       %46 = load %11[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:       %47 = memref_cast %46 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:       %48 = memref_cast %47 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:       call @print_array(%c2100_i32, %48) : (i32, memref<2100xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %42 = memref_cast %6 : memref<?xmemref<1900x2100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%42) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %43 = memref_cast %9 : memref<?xmemref<2100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%43) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %44 = memref_cast %11 : memref<?xmemref<2100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%44) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %45 = memref_cast %14 : memref<?xmemref<1900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%45) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg1 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = addi %c0, %3 : index
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %5 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:     %6 = sitofp %1 : i32 to f64
// CHECK-NEXT:     %7 = divf %6, %0 : f64
// CHECK-NEXT:     %8 = addf %5, %7 : f64
// CHECK-NEXT:     store %8, %arg3[%4] : memref<2100xf64>
// CHECK-NEXT:     %9 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%9 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%10: i32):  // 2 preds: ^bb3, ^bb9
// CHECK-NEXT:     %11 = cmpi "slt", %10, %arg0 : i32
// CHECK-NEXT:     cond_br %11, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%12: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %13 = cmpi "slt", %12, %arg1 : i32
// CHECK-NEXT:     cond_br %13, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %14 = index_cast %10 : i32 to index
// CHECK-NEXT:     %15 = addi %c0, %14 : index
// CHECK-NEXT:     %16 = memref_cast %arg2 : memref<1900x2100xf64> to memref<?x2100xf64>
// CHECK-NEXT:     %17 = index_cast %12 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = addi %10, %12 : i32
// CHECK-NEXT:     %20 = remi_signed %19, %arg1 : i32
// CHECK-NEXT:     %21 = sitofp %20 : i32 to f64
// CHECK-NEXT:     %c5_i32 = constant 5 : i32
// CHECK-NEXT:     %22 = muli %c5_i32, %arg0 : i32
// CHECK-NEXT:     %23 = sitofp %22 : i32 to f64
// CHECK-NEXT:     %24 = divf %21, %23 : f64
// CHECK-NEXT:     store %24, %16[%15, %18] : memref<?x2100xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %25 = addi %12, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%25 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %10, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%26 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg1 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %4, %arg4[%3] : memref<2100xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %5 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%6: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %7 = cmpi "slt", %6, %arg0 : i32
// CHECK-NEXT:     cond_br %7, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %8 = index_cast %6 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %arg5[%9] : memref<1900xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%10: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %11 = cmpi "slt", %10, %arg1 : i32
// CHECK-NEXT:     cond_br %11, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %12 = load %arg5[%9] : memref<1900xf64>
// CHECK-NEXT:     %13 = memref_cast %arg2 : memref<1900x2100xf64> to memref<?x2100xf64>
// CHECK-NEXT:     %14 = index_cast %10 : i32 to index
// CHECK-NEXT:     %15 = addi %c0, %14 : index
// CHECK-NEXT:     %16 = load %13[%9, %15] : memref<?x2100xf64>
// CHECK-NEXT:     %17 = load %arg3[%15] : memref<2100xf64>
// CHECK-NEXT:     %18 = mulf %16, %17 : f64
// CHECK-NEXT:     %19 = addf %12, %18 : f64
// CHECK-NEXT:     store %19, %arg5[%9] : memref<1900xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %20 = addi %10, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%20 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%21: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %22 = cmpi "slt", %21, %arg1 : i32
// CHECK-NEXT:     cond_br %22, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %23 = index_cast %21 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %arg4[%24] : memref<2100xf64>
// CHECK-NEXT:     %26 = memref_cast %arg2 : memref<1900x2100xf64> to memref<?x2100xf64>
// CHECK-NEXT:     %27 = load %26[%9, %24] : memref<?x2100xf64>
// CHECK-NEXT:     %28 = load %arg5[%9] : memref<1900xf64>
// CHECK-NEXT:     %29 = mulf %27, %28 : f64
// CHECK-NEXT:     %30 = addf %25, %29 : f64
// CHECK-NEXT:     store %30, %arg4[%24] : memref<2100xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %31 = addi %21, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%31 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %6, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb4(%32 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2100xf64>) {
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
// CHECK-NEXT:   ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg0 : i32
// CHECK-NEXT:     cond_br %12, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %13 = remi_signed %11, %c20_i32 : i32
// CHECK-NEXT:     %14 = cmpi "eq", %13, %c0_i32 : i32
// CHECK-NEXT:     scf.if %14 {
// CHECK-NEXT:       %34 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:       %35 = llvm.mlir.addressof @str3 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %36 = llvm.getelementptr %35[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %37 = llvm.call @fprintf(%34, %36) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %18 = index_cast %11 : i32 to index
// CHECK-NEXT:     %19 = addi %c0, %18 : index
// CHECK-NEXT:     %20 = load %arg1[%19] : memref<2100xf64>
// CHECK-NEXT:     %21 = llvm.mlir.cast %20 : f64 to !llvm.double
// CHECK-NEXT:     %22 = llvm.call @fprintf(%15, %17, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %23 = addi %11, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%23 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %24 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %25 = llvm.mlir.addressof @str5 : !llvm.ptr<array<16 x i8>>
// CHECK-NEXT:     %26 = llvm.getelementptr %25[%2, %2] : (!llvm.ptr<array<16 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %27 = llvm.mlir.addressof @str2 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %28 = llvm.getelementptr %27[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %29 = llvm.call @fprintf(%24, %26, %28) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %30 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %31 = llvm.mlir.addressof @str6 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %32 = llvm.getelementptr %31[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %33 = llvm.call @fprintf(%30, %32) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }