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
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("x")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
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
// CHECK-NEXT:     %6 = zexti %0 : i32 to i64
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %9 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %13 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %16 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %19 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %22 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %15, %18, %21, %24) : (i32, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %25 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %28 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %31 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %34 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @kernel_ludcmp(%c2000_i32, %27, %30, %33, %36) : (i32, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
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
// CHECK-NEXT:       %50 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %51 = memref_cast %50 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %52 = memref_cast %51 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %52) : (i32, memref<2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %46 = memref_cast %5 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%46) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %47 = memref_cast %8 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%47) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %48 = memref_cast %10 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%48) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %49 = memref_cast %12 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%49) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = addi %c0, %3 : index
// CHECK-NEXT:     %5 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %5, %arg3[%4] : memref<2000xf64>
// CHECK-NEXT:     store %5, %arg4[%4] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %6 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     %7 = sitofp %6 : i32 to f64
// CHECK-NEXT:     %8 = divf %7, %0 : f64
// CHECK-NEXT:     %cst = constant 2.000000e+00 : f64
// CHECK-NEXT:     %9 = divf %8, %cst : f64
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %10 = sitofp %c4_i32 : i32 to f64
// CHECK-NEXT:     %11 = addf %9, %10 : f64
// CHECK-NEXT:     store %11, %arg2[%4] : memref<2000xf64>
// CHECK-NEXT:     br ^bb1(%6 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%12: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %13 = cmpi "slt", %12, %arg0 : i32
// CHECK-NEXT:     cond_br %13, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %14 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %15 = muli %14, %14 : i32
// CHECK-NEXT:     %16 = zexti %15 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %17 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %18 = call @polybench_alloc_data(%16, %17) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?xi8> to memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%20: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %21 = cmpi "sle", %20, %12 : i32
// CHECK-NEXT:     cond_br %21, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %22 = index_cast %12 : i32 to index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %25 = index_cast %20 : i32 to index
// CHECK-NEXT:     %26 = addi %c0, %25 : index
// CHECK-NEXT:     %27 = subi %c0_i32, %20 : i32
// CHECK-NEXT:     %28 = remi_signed %27, %arg0 : i32
// CHECK-NEXT:     %29 = sitofp %28 : i32 to f64
// CHECK-NEXT:     %30 = divf %29, %0 : f64
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %31 = sitofp %c1_i32_0 : i32 to f64
// CHECK-NEXT:     %32 = addf %30, %31 : f64
// CHECK-NEXT:     store %32, %24[%23, %26] : memref<?x2000xf64>
// CHECK-NEXT:     %33 = addi %20, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%33 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %34 = addi %12, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%34 : i32)
// CHECK-NEXT:   ^bb10(%35: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %36 = cmpi "slt", %35, %arg0 : i32
// CHECK-NEXT:     cond_br %36, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %37 = index_cast %12 : i32 to index
// CHECK-NEXT:     %38 = addi %c0, %37 : index
// CHECK-NEXT:     %39 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %40 = index_cast %35 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %42 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %42, %39[%38, %41] : memref<?x2000xf64>
// CHECK-NEXT:     %43 = addi %35, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%43 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %44 = index_cast %12 : i32 to index
// CHECK-NEXT:     %45 = addi %c0, %44 : index
// CHECK-NEXT:     %46 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %47 = sitofp %c1_i32_1 : i32 to f64
// CHECK-NEXT:     store %47, %46[%45, %45] : memref<?x2000xf64>
// CHECK-NEXT:     br ^bb4(%34 : i32)
// CHECK-NEXT:   ^bb13(%48: i32):  // 2 preds: ^bb6, ^bb18
// CHECK-NEXT:     %49 = cmpi "slt", %48, %arg0 : i32
// CHECK-NEXT:     cond_br %49, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%50: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %51 = cmpi "slt", %50, %arg0 : i32
// CHECK-NEXT:     cond_br %51, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %52 = load %19[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %53 = memref_cast %52 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %54 = index_cast %48 : i32 to index
// CHECK-NEXT:     %55 = addi %c0, %54 : index
// CHECK-NEXT:     %56 = memref_cast %53 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %57 = index_cast %50 : i32 to index
// CHECK-NEXT:     %58 = addi %c0, %57 : index
// CHECK-NEXT:     %59 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %59, %56[%55, %58] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %60 = addi %50, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb16(%60 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %61 = addi %48, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb13(%61 : i32)
// CHECK-NEXT:   ^bb19(%62: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %63 = cmpi "slt", %62, %arg0 : i32
// CHECK-NEXT:     cond_br %63, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     br ^bb28(%c0_i32 : i32)
// CHECK-NEXT:   ^bb22(%64: i32):  // 2 preds: ^bb20, ^bb27
// CHECK-NEXT:     %65 = cmpi "slt", %64, %arg0 : i32
// CHECK-NEXT:     cond_br %65, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     br ^bb25(%c0_i32 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %66 = addi %62, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb19(%66 : i32)
// CHECK-NEXT:   ^bb25(%67: i32):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:     %68 = cmpi "slt", %67, %arg0 : i32
// CHECK-NEXT:     cond_br %68, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %69 = load %19[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %70 = memref_cast %69 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %71 = index_cast %64 : i32 to index
// CHECK-NEXT:     %72 = addi %c0, %71 : index
// CHECK-NEXT:     %73 = memref_cast %70 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %74 = index_cast %67 : i32 to index
// CHECK-NEXT:     %75 = addi %c0, %74 : index
// CHECK-NEXT:     %76 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %77 = index_cast %62 : i32 to index
// CHECK-NEXT:     %78 = addi %c0, %77 : index
// CHECK-NEXT:     %79 = load %76[%72, %78] : memref<?x2000xf64>
// CHECK-NEXT:     %80 = load %76[%75, %78] : memref<?x2000xf64>
// CHECK-NEXT:     %81 = mulf %79, %80 : f64
// CHECK-NEXT:     %82 = load %73[%72, %75] : memref<?x2000xf64>
// CHECK-NEXT:     %83 = addf %82, %81 : f64
// CHECK-NEXT:     store %83, %73[%72, %75] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %84 = addi %67, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb25(%84 : i32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %85 = addi %64, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb22(%85 : i32)
// CHECK-NEXT:   ^bb28(%86: i32):  // 2 preds: ^bb21, ^bb33
// CHECK-NEXT:     %87 = cmpi "slt", %86, %arg0 : i32
// CHECK-NEXT:     cond_br %87, ^bb29, ^bb30
// CHECK-NEXT:   ^bb29:  // pred: ^bb28
// CHECK-NEXT:     br ^bb31(%c0_i32 : i32)
// CHECK-NEXT:   ^bb30:  // pred: ^bb28
// CHECK-NEXT:     %88 = memref_cast %19 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%88) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb31(%89: i32):  // 2 preds: ^bb29, ^bb32
// CHECK-NEXT:     %90 = cmpi "slt", %89, %arg0 : i32
// CHECK-NEXT:     cond_br %90, ^bb32, ^bb33
// CHECK-NEXT:   ^bb32:  // pred: ^bb31
// CHECK-NEXT:     %91 = index_cast %86 : i32 to index
// CHECK-NEXT:     %92 = addi %c0, %91 : index
// CHECK-NEXT:     %93 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %94 = index_cast %89 : i32 to index
// CHECK-NEXT:     %95 = addi %c0, %94 : index
// CHECK-NEXT:     %96 = load %19[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %97 = memref_cast %96 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %98 = memref_cast %97 : memref<?x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %99 = load %98[%92, %95] : memref<?x2000xf64>
// CHECK-NEXT:     store %99, %93[%92, %95] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %100 = addi %89, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb31(%100 : i32)
// CHECK-NEXT:   ^bb33:  // pred: ^bb31
// CHECK-NEXT:     %c1_i32_8 = constant 1 : i32
// CHECK-NEXT:     %101 = addi %86, %c1_i32_8 : i32
// CHECK-NEXT:     br ^bb28(%101 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb12
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = load %6[%5, %8] : memref<?x2000xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32, %9 : i32, f64)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb10(%0 : i32)
// CHECK-NEXT:   ^bb7(%10: i32, %11: f64):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %12 = cmpi "slt", %10, %2 : i32
// CHECK-NEXT:     cond_br %12, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %13 = index_cast %10 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = load %6[%5, %14] : memref<?x2000xf64>
// CHECK-NEXT:     %16 = load %6[%14, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %17 = mulf %15, %16 : f64
// CHECK-NEXT:     %18 = subf %11, %17 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %19 = addi %10, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%19, %18 : i32, f64)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %20 = load %6[%8, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %21 = divf %11, %20 : f64
// CHECK-NEXT:     store %21, %6[%5, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %2, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb4(%22 : i32)
// CHECK-NEXT:   ^bb10(%23: i32):  // 2 preds: ^bb6, ^bb15
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg0 : i32
// CHECK-NEXT:     cond_br %24, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %25 = index_cast %0 : i32 to index
// CHECK-NEXT:     %26 = addi %c0, %25 : index
// CHECK-NEXT:     %27 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %28 = index_cast %23 : i32 to index
// CHECK-NEXT:     %29 = addi %c0, %28 : index
// CHECK-NEXT:     %30 = load %27[%26, %29] : memref<?x2000xf64>
// CHECK-NEXT:     br ^bb13(%c0_i32, %30 : i32, f64)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %31 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%31 : i32)
// CHECK-NEXT:   ^bb13(%32: i32, %33: f64):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %34 = cmpi "slt", %32, %0 : i32
// CHECK-NEXT:     cond_br %34, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %35 = index_cast %32 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %27[%26, %36] : memref<?x2000xf64>
// CHECK-NEXT:     %38 = load %27[%36, %29] : memref<?x2000xf64>
// CHECK-NEXT:     %39 = mulf %37, %38 : f64
// CHECK-NEXT:     %40 = subf %33, %39 : f64
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %41 = addi %32, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb13(%41, %40 : i32, f64)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     store %33, %27[%26, %29] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %42 = addi %23, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb10(%42 : i32)
// CHECK-NEXT:   ^bb16(%43: i32):  // 2 preds: ^bb3, ^bb21
// CHECK-NEXT:     %44 = cmpi "slt", %43, %arg0 : i32
// CHECK-NEXT:     cond_br %44, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %45 = index_cast %43 : i32 to index
// CHECK-NEXT:     %46 = addi %c0, %45 : index
// CHECK-NEXT:     %47 = load %arg2[%46] : memref<2000xf64>
// CHECK-NEXT:     br ^bb19(%c0_i32, %47 : i32, f64)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %48 = subi %arg0, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb22(%48 : i32)
// CHECK-NEXT:   ^bb19(%49: i32, %50: f64):  // 2 preds: ^bb17, ^bb20
// CHECK-NEXT:     %51 = cmpi "slt", %49, %43 : i32
// CHECK-NEXT:     cond_br %51, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %52 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %53 = index_cast %49 : i32 to index
// CHECK-NEXT:     %54 = addi %c0, %53 : index
// CHECK-NEXT:     %55 = load %52[%46, %54] : memref<?x2000xf64>
// CHECK-NEXT:     %56 = load %arg4[%54] : memref<2000xf64>
// CHECK-NEXT:     %57 = mulf %55, %56 : f64
// CHECK-NEXT:     %58 = subf %50, %57 : f64
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %59 = addi %49, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb19(%59, %58 : i32, f64)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     store %50, %arg4[%46] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %60 = addi %43, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb16(%60 : i32)
// CHECK-NEXT:   ^bb22(%61: i32):  // 2 preds: ^bb18, ^bb27
// CHECK-NEXT:     %62 = cmpi "sge", %61, %c0_i32 : i32
// CHECK-NEXT:     cond_br %62, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %63 = index_cast %61 : i32 to index
// CHECK-NEXT:     %64 = addi %c0, %63 : index
// CHECK-NEXT:     %65 = load %arg4[%64] : memref<2000xf64>
// CHECK-NEXT:     %66 = addi %61, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb25(%66, %65 : i32, f64)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb25(%67: i32, %68: f64):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:     %69 = cmpi "slt", %67, %arg0 : i32
// CHECK-NEXT:     cond_br %69, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %70 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %71 = index_cast %67 : i32 to index
// CHECK-NEXT:     %72 = addi %c0, %71 : index
// CHECK-NEXT:     %73 = load %70[%64, %72] : memref<?x2000xf64>
// CHECK-NEXT:     %74 = load %arg3[%72] : memref<2000xf64>
// CHECK-NEXT:     %75 = mulf %73, %74 : f64
// CHECK-NEXT:     %76 = subf %68, %75 : f64
// CHECK-NEXT:     %77 = addi %67, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb25(%77, %76 : i32, f64)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %78 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %79 = load %78[%64, %64] : memref<?x2000xf64>
// CHECK-NEXT:     %80 = divf %68, %79 : f64
// CHECK-NEXT:     store %80, %arg3[%64] : memref<2000xf64>
// CHECK-NEXT:     %81 = subi %61, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb22(%81 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
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
// CHECK-NEXT:     %20 = load %arg1[%19] : memref<2000xf64>
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