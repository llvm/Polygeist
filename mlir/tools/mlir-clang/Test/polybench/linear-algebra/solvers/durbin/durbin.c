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
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
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
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
 DATA_TYPE z[N];
 DATA_TYPE alpha;
 DATA_TYPE beta;
 DATA_TYPE sum;

 int i,k;

#pragma scop
 y[0] = -r[0];
 beta = SCALAR_VAL(1.0);
 alpha = -r[0];

 for (k = 1; k < _PB_N; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = SCALAR_VAL(0.0);
   for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
 }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = zexti %0 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %2 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %3 = call @polybench_alloc_data(%1, %2) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %4 = memref_cast %3 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%1, %2) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %7 = load %4[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %9) : (i32, memref<2000xf64>) -> ()
// CHECK-NEXT:     %10 = load %4[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %13 = load %6[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @kernel_durbin(%c2000_i32, %12, %15) : (i32, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %16 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %17 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = load %arg1[%18] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %20 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %21 = call @strcmp(%19, %20) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %22 = trunci %21 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %23 = xor %22, %true : i1
// CHECK-NEXT:     %24 = and %16, %23 : i1
// CHECK-NEXT:     scf.if %24 {
// CHECK-NEXT:       %27 = load %6[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %28 = memref_cast %27 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %29 = memref_cast %28 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %29) : (i32, memref<2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %25 = memref_cast %4 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%25) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %26 = memref_cast %6 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%26) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %arg0, %c1_i32 : i32
// CHECK-NEXT:     %5 = subi %4, %0 : i32
// CHECK-NEXT:     %6 = sitofp %5 : i32 to f64
// CHECK-NEXT:     store %6, %arg1[%3] : memref<2000xf64>
// CHECK-NEXT:     %7 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%7 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_durbin(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = alloca() : memref<2000xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %1 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %2 = addi %c0, %1 : index
// CHECK-NEXT:     %3 = load %arg1[%2] : memref<2000xf64>
// CHECK-NEXT:     %4 = negf %3 : f64
// CHECK-NEXT:     store %4, %arg2[%2] : memref<2000xf64>
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:     %5 = load %arg1[%2] : memref<2000xf64>
// CHECK-NEXT:     %6 = negf %5 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%6, %cst, %c1_i32 : f64, f64, i32)
// CHECK-NEXT:   ^bb1(%7: f64, %8: f64, %9: i32):  // 2 preds: ^bb0, ^bb12
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg0 : i32
// CHECK-NEXT:     cond_br %10, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %11 = sitofp %c1_i32 : i32 to f64
// CHECK-NEXT:     %12 = mulf %7, %7 : f64
// CHECK-NEXT:     %13 = subf %11, %12 : f64
// CHECK-NEXT:     %14 = mulf %13, %8 : f64
// CHECK-NEXT:     %cst_0 = constant 0.000000e+00 : f64
// CHECK-NEXT:     br ^bb4(%cst_0, %c0_i32 : f64, i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%15: f64, %16: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %17 = cmpi "slt", %16, %9 : i32
// CHECK-NEXT:     cond_br %17, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %18 = subi %9, %16 : i32
// CHECK-NEXT:     %19 = subi %18, %c1_i32 : i32
// CHECK-NEXT:     %20 = index_cast %19 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = load %arg1[%21] : memref<2000xf64>
// CHECK-NEXT:     %23 = index_cast %16 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %arg2[%24] : memref<2000xf64>
// CHECK-NEXT:     %26 = mulf %22, %25 : f64
// CHECK-NEXT:     %27 = addf %15, %26 : f64
// CHECK-NEXT:     %28 = addi %16, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%27, %28 : f64, i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %29 = index_cast %9 : i32 to index
// CHECK-NEXT:     %30 = addi %c0, %29 : index
// CHECK-NEXT:     %31 = load %arg1[%30] : memref<2000xf64>
// CHECK-NEXT:     %32 = addf %31, %15 : f64
// CHECK-NEXT:     %33 = negf %32 : f64
// CHECK-NEXT:     %34 = divf %33, %14 : f64
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb7(%35: i32):  // 2 preds: ^bb6, ^bb8
// CHECK-NEXT:     %36 = cmpi "slt", %35, %9 : i32
// CHECK-NEXT:     cond_br %36, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %37 = memref_cast %0 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %38 = index_cast %35 : i32 to index
// CHECK-NEXT:     %39 = addi %c0, %38 : index
// CHECK-NEXT:     %40 = load %arg2[%39] : memref<2000xf64>
// CHECK-NEXT:     %41 = subi %9, %35 : i32
// CHECK-NEXT:     %42 = subi %41, %c1_i32 : i32
// CHECK-NEXT:     %43 = index_cast %42 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %45 = load %arg2[%44] : memref<2000xf64>
// CHECK-NEXT:     %46 = mulf %34, %45 : f64
// CHECK-NEXT:     %47 = addf %40, %46 : f64
// CHECK-NEXT:     store %47, %37[%39] : memref<?xf64>
// CHECK-NEXT:     %48 = addi %35, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%48 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%49: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %50 = cmpi "slt", %49, %9 : i32
// CHECK-NEXT:     cond_br %50, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %51 = index_cast %49 : i32 to index
// CHECK-NEXT:     %52 = addi %c0, %51 : index
// CHECK-NEXT:     %53 = memref_cast %0 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %54 = load %53[%52] : memref<?xf64>
// CHECK-NEXT:     store %54, %arg2[%52] : memref<2000xf64>
// CHECK-NEXT:     %55 = addi %49, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%55 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     store %34, %arg2[%30] : memref<2000xf64>
// CHECK-NEXT:     %56 = addi %9, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%34, %14, %56 : f64, f64, i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
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
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }