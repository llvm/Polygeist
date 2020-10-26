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
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}

// CHECK: module {
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
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %15 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %18 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %21 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %24 = load %14[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %27 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %17, %20, %23, %26, %29) : (i32, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000x2000xf64>) -> ()
// CHECK-NEXT:     %30 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %33 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %34 = memref_cast %33 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %36 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %37 = memref_cast %36 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %38 = memref_cast %37 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %39 = load %14[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %40 = memref_cast %39 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %41 = memref_cast %40 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %42 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %43 = memref_cast %42 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %44 = memref_cast %43 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     call @kernel_mvt(%c2000_i32, %32, %35, %38, %41, %44) : (i32, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000x2000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %45 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %46 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %47 = addi %c0, %46 : index
// CHECK-NEXT:     %48 = load %arg1[%47] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %49 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %50 = call @strcmp(%48, %49) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %51 = trunci %50 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %52 = xor %51, %true : i1
// CHECK-NEXT:     %53 = and %45, %52 : i1
// CHECK-NEXT:     scf.if %53 {
// CHECK-NEXT:       %59 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %60 = memref_cast %59 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %61 = memref_cast %60 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       %62 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %63 = memref_cast %62 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %64 = memref_cast %63 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %61, %64) : (i32, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %54 = memref_cast %5 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%54) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %55 = memref_cast %8 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%55) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %56 = memref_cast %10 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%56) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %57 = memref_cast %12 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%57) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %58 = memref_cast %14 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%58) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000x2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = remi_signed %0, %arg0 : i32
// CHECK-NEXT:     %5 = sitofp %4 : i32 to f64
// CHECK-NEXT:     %6 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %7 = divf %5, %6 : f64
// CHECK-NEXT:     store %7, %arg1[%3] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %8 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     %9 = remi_signed %8, %arg0 : i32
// CHECK-NEXT:     %10 = sitofp %9 : i32 to f64
// CHECK-NEXT:     %11 = divf %10, %6 : f64
// CHECK-NEXT:     store %11, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %12 = addi %0, %c3_i32 : i32
// CHECK-NEXT:     %13 = remi_signed %12, %arg0 : i32
// CHECK-NEXT:     %14 = sitofp %13 : i32 to f64
// CHECK-NEXT:     %15 = divf %14, %6 : f64
// CHECK-NEXT:     store %15, %arg3[%3] : memref<2000xf64>
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %16 = addi %0, %c4_i32 : i32
// CHECK-NEXT:     %17 = remi_signed %16, %arg0 : i32
// CHECK-NEXT:     %18 = sitofp %17 : i32 to f64
// CHECK-NEXT:     %19 = divf %18, %6 : f64
// CHECK-NEXT:     store %19, %arg4[%3] : memref<2000xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%20: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %21 = cmpi "slt", %20, %arg0 : i32
// CHECK-NEXT:     cond_br %21, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %22 = memref_cast %arg5 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %23 = index_cast %20 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = muli %0, %20 : i32
// CHECK-NEXT:     %26 = remi_signed %25, %arg0 : i32
// CHECK-NEXT:     %27 = sitofp %26 : i32 to f64
// CHECK-NEXT:     %28 = divf %27, %6 : f64
// CHECK-NEXT:     store %28, %22[%3, %24] : memref<?x2000xf64>
// CHECK-NEXT:     %29 = addi %20, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%29 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb1(%8 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_mvt(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000x2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = load %arg1[%5] : memref<2000xf64>
// CHECK-NEXT:     %7 = memref_cast %arg5 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %8 = index_cast %2 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %10 = load %7[%5, %9] : memref<?x2000xf64>
// CHECK-NEXT:     %11 = load %arg3[%9] : memref<2000xf64>
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = addf %6, %12 : f64
// CHECK-NEXT:     store %13, %arg1[%5] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%14 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %15 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%15 : i32)
// CHECK-NEXT:   ^bb7(%16: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %17 = cmpi "slt", %16, %arg0 : i32
// CHECK-NEXT:     cond_br %17, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%18: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %19 = cmpi "slt", %18, %arg0 : i32
// CHECK-NEXT:     cond_br %19, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %20 = index_cast %16 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = load %arg2[%21] : memref<2000xf64>
// CHECK-NEXT:     %23 = index_cast %18 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = memref_cast %arg5 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %26 = load %25[%24, %21] : memref<?x2000xf64>
// CHECK-NEXT:     %27 = load %arg4[%24] : memref<2000xf64>
// CHECK-NEXT:     %28 = mulf %26, %27 : f64
// CHECK-NEXT:     %29 = addf %22, %28 : f64
// CHECK-NEXT:     store %29, %arg2[%21] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %30 = addi %18, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%30 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %31 = addi %16, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%31 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %2 = remi_signed %0, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %3 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%3 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%4: i32):  // 2 preds: ^bb3, ^bb5
// CHECK-NEXT:     %5 = cmpi "slt", %4, %arg0 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %c20_i32_0 = constant 20 : i32
// CHECK-NEXT:     %6 = remi_signed %4, %c20_i32_0 : i32
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %4, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%7 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }