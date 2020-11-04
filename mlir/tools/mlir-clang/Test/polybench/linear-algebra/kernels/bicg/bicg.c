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
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_M; i++)
    s[i] = 0;
  for (i = 0; i < _PB_N; i++)
    {
      q[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_M; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("q")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2100_i32 = constant 2100 : i32
// CHECK-NEXT:     %c1900_i32 = constant 1900 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2100_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c1900_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<2100x1900xf64>>
// CHECK-NEXT:     %7 = zexti %1 : i32 to i64
// CHECK-NEXT:     %8 = call @polybench_alloc_data(%7, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xi8> to memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %10 = zexti %0 : i32 to i64
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%10, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%7, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %15 = call @polybench_alloc_data(%10, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?xi8> to memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %17 = load %6[%c0] : memref<?xmemref<2100x1900xf64>>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<2100x1900xf64> to memref<?x1900xf64>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?x1900xf64> to memref<2100x1900xf64>
// CHECK-NEXT:     %20 = load %16[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     %23 = load %14[%c0] : memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1900xf64> to memref<?xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?xf64> to memref<1900xf64>
// CHECK-NEXT:     call @init_array(%c1900_i32, %c2100_i32, %19, %22, %25) : (i32, i32, memref<2100x1900xf64>, memref<2100xf64>, memref<1900xf64>) -> ()
// CHECK-NEXT:     %26 = load %6[%c0] : memref<?xmemref<2100x1900xf64>>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<2100x1900xf64> to memref<?x1900xf64>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<?x1900xf64> to memref<2100x1900xf64>
// CHECK-NEXT:     %29 = load %9[%c0] : memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<1900xf64> to memref<?xf64>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<?xf64> to memref<1900xf64>
// CHECK-NEXT:     %32 = load %12[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %34 = memref_cast %33 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     %35 = load %14[%c0] : memref<?xmemref<1900xf64>>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<1900xf64> to memref<?xf64>
// CHECK-NEXT:     %37 = memref_cast %36 : memref<?xf64> to memref<1900xf64>
// CHECK-NEXT:     %38 = load %16[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:     %39 = memref_cast %38 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:     %40 = memref_cast %39 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:     call @kernel_bicg(%c1900_i32, %c2100_i32, %28, %31, %34, %37, %40) : (i32, i32, memref<2100x1900xf64>, memref<1900xf64>, memref<2100xf64>, memref<1900xf64>, memref<2100xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %41 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %42 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %43 = addi %c0, %42 : index
// CHECK-NEXT:     %44 = load %arg1[%43] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %45 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %46 = call @strcmp(%44, %45) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %47 = trunci %46 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %48 = xor %47, %true : i1
// CHECK-NEXT:     %49 = and %41, %48 : i1
// CHECK-NEXT:     scf.if %49 {
// CHECK-NEXT:       %55 = load %9[%c0] : memref<?xmemref<1900xf64>>
// CHECK-NEXT:       %56 = memref_cast %55 : memref<1900xf64> to memref<?xf64>
// CHECK-NEXT:       %57 = memref_cast %56 : memref<?xf64> to memref<1900xf64>
// CHECK-NEXT:       %58 = load %12[%c0] : memref<?xmemref<2100xf64>>
// CHECK-NEXT:       %59 = memref_cast %58 : memref<2100xf64> to memref<?xf64>
// CHECK-NEXT:       %60 = memref_cast %59 : memref<?xf64> to memref<2100xf64>
// CHECK-NEXT:       call @print_array(%c1900_i32, %c2100_i32, %57, %60) : (i32, i32, memref<1900xf64>, memref<2100xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %50 = memref_cast %6 : memref<?xmemref<2100x1900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%50) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %51 = memref_cast %9 : memref<?xmemref<1900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%51) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %52 = memref_cast %12 : memref<?xmemref<2100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%52) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %53 = memref_cast %14 : memref<?xmemref<1900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%53) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %54 = memref_cast %16 : memref<?xmemref<2100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%54) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<2100x1900xf64>, %arg3: memref<2100xf64>, %arg4: memref<1900xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = remi_signed %0, %arg0 : i32
// CHECK-NEXT:     %5 = sitofp %4 : i32 to f64
// CHECK-NEXT:     %6 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %7 = divf %5, %6 : f64
// CHECK-NEXT:     store %7, %arg4[%3] : memref<1900xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %8 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%8 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%9: i32):  // 2 preds: ^bb3, ^bb9
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg1 : i32
// CHECK-NEXT:     cond_br %10, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %11 = index_cast %9 : i32 to index
// CHECK-NEXT:     %12 = addi %c0, %11 : index
// CHECK-NEXT:     %13 = remi_signed %9, %arg1 : i32
// CHECK-NEXT:     %14 = sitofp %13 : i32 to f64
// CHECK-NEXT:     %15 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %16 = divf %14, %15 : f64
// CHECK-NEXT:     store %16, %arg3[%12] : memref<2100xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%17: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg0 : i32
// CHECK-NEXT:     cond_br %18, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %19 = memref_cast %arg2 : memref<2100x1900xf64> to memref<?x1900xf64>
// CHECK-NEXT:     %20 = index_cast %17 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %17, %c1_i32_0 : i32
// CHECK-NEXT:     %23 = muli %9, %22 : i32
// CHECK-NEXT:     %24 = remi_signed %23, %arg1 : i32
// CHECK-NEXT:     %25 = sitofp %24 : i32 to f64
// CHECK-NEXT:     %26 = divf %25, %15 : f64
// CHECK-NEXT:     store %26, %19[%12, %21] : memref<?x1900xf64>
// CHECK-NEXT:     br ^bb7(%22 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %27 = addi %9, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%27 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<2100x1900xf64>, %arg3: memref<1900xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>, %arg6: memref<2100xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     store %4, %arg3[%3] : memref<1900xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %5 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%6: i32):  // 2 preds: ^bb3, ^bb9
// CHECK-NEXT:     %7 = cmpi "slt", %6, %arg1 : i32
// CHECK-NEXT:     cond_br %7, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %8 = index_cast %6 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %arg4[%9] : memref<2100xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%10: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %11 = cmpi "slt", %10, %arg0 : i32
// CHECK-NEXT:     cond_br %11, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %12 = index_cast %10 : i32 to index
// CHECK-NEXT:     %13 = addi %c0, %12 : index
// CHECK-NEXT:     %14 = load %arg3[%13] : memref<1900xf64>
// CHECK-NEXT:     %15 = load %arg6[%9] : memref<2100xf64>
// CHECK-NEXT:     %16 = memref_cast %arg2 : memref<2100x1900xf64> to memref<?x1900xf64>
// CHECK-NEXT:     %17 = load %16[%9, %13] : memref<?x1900xf64>
// CHECK-NEXT:     %18 = mulf %15, %17 : f64
// CHECK-NEXT:     %19 = addf %14, %18 : f64
// CHECK-NEXT:     store %19, %arg3[%13] : memref<1900xf64>
// CHECK-NEXT:     %20 = load %arg4[%9] : memref<2100xf64>
// CHECK-NEXT:     %21 = load %16[%9, %13] : memref<?x1900xf64>
// CHECK-NEXT:     %22 = load %arg5[%13] : memref<1900xf64>
// CHECK-NEXT:     %23 = mulf %21, %22 : f64
// CHECK-NEXT:     %24 = addf %20, %23 : f64
// CHECK-NEXT:     store %24, %arg4[%9] : memref<2100xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %25 = addi %10, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%25 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %6, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%26 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1900xf64>, %arg3: memref<2100xf64>) {
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
// CHECK-NEXT:       %59 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:       %60 = llvm.mlir.addressof @str3 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %61 = llvm.getelementptr %60[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %62 = llvm.call @fprintf(%59, %61) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %18 = index_cast %11 : i32 to index
// CHECK-NEXT:     %19 = addi %c0, %18 : index
// CHECK-NEXT:     %20 = load %arg2[%19] : memref<1900xf64>
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
// CHECK-NEXT:     %31 = llvm.mlir.addressof @str1 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %32 = llvm.getelementptr %31[%2, %2] : (!llvm.ptr<array<14 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %33 = llvm.mlir.addressof @str6 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %34 = llvm.getelementptr %33[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %35 = llvm.call @fprintf(%30, %32, %34) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%36: i32):  // 2 preds: ^bb3, ^bb5
// CHECK-NEXT:     %37 = cmpi "slt", %36, %arg1 : i32
// CHECK-NEXT:     cond_br %37, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %c20_i32_0 = constant 20 : i32
// CHECK-NEXT:     %38 = remi_signed %36, %c20_i32_0 : i32
// CHECK-NEXT:     %39 = cmpi "eq", %38, %c0_i32 : i32
// CHECK-NEXT:     scf.if %39 {
// CHECK-NEXT:       %59 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:       %60 = llvm.mlir.addressof @str3 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %61 = llvm.getelementptr %60[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %62 = llvm.call @fprintf(%59, %61) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %40 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %41 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %42 = llvm.getelementptr %41[%2, %2] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %43 = index_cast %36 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %45 = load %arg3[%44] : memref<2100xf64>
// CHECK-NEXT:     %46 = llvm.mlir.cast %45 : f64 to !llvm.double
// CHECK-NEXT:     %47 = llvm.call @fprintf(%40, %42, %46) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %48 = addi %36, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%48 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %49 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %50 = llvm.mlir.addressof @str5 : !llvm.ptr<array<16 x i8>>
// CHECK-NEXT:     %51 = llvm.getelementptr %50[%2, %2] : (!llvm.ptr<array<16 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %52 = llvm.mlir.addressof @str6 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %53 = llvm.getelementptr %52[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %54 = llvm.call @fprintf(%49, %51, %53) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %55 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %56 = llvm.mlir.addressof @str7 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %57 = llvm.getelementptr %56[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %58 = llvm.call @fprintf(%55, %57) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }