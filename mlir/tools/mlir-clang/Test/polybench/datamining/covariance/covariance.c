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
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      {
        cov[i][j] = SCALAR_VAL(0.0);
        for (k = 0; k < _PB_N; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
        cov[j][i] = cov[i][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %1 = addi %c1400_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = muli %1, %2 : i32
// CHECK-NEXT:     %4 = zexti %3 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %5 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<1400x1200xf64>>
// CHECK-NEXT:     %8 = muli %2, %2 : i32
// CHECK-NEXT:     %9 = zexti %8 : i32 to i64
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%9, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %12 = zexti %2 : i32 to i64
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%12, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<1200xf64>>
// CHECK-NEXT:     %15 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %16 = load %7[%c0] : memref<?xmemref<1400x1200xf64>>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?x1200xf64> to memref<1400x1200xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1400_i32, %15, %18) : (i32, i32, memref<?xf64>, memref<1400x1200xf64>) -> ()
// CHECK-NEXT:     %19 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %20 = load %7[%c0] : memref<?xmemref<1400x1200xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x1200xf64> to memref<1400x1200xf64>
// CHECK-NEXT:     %23 = load %11[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %26 = load %14[%c0] : memref<?xmemref<1200xf64>>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<1200xf64> to memref<?xf64>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<?xf64> to memref<1200xf64>
// CHECK-NEXT:     call @kernel_covariance(%c1200_i32, %c1400_i32, %19, %22, %25, %28) : (i32, i32, f64, memref<1400x1200xf64>, memref<1200x1200xf64>, memref<1200xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %29 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %30 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = load %arg1[%31] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %33 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %34 = call @strcmp(%32, %33) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %35 = trunci %34 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %36 = xor %35, %true : i1
// CHECK-NEXT:     %37 = and %29, %36 : i1
// CHECK-NEXT:     scf.if %37 {
// CHECK-NEXT:       %41 = load %11[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:       %42 = memref_cast %41 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %43 = memref_cast %42 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:       call @print_array(%c1200_i32, %43) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %38 = memref_cast %7 : memref<?xmemref<1400x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%38) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %39 = memref_cast %11 : memref<?xmemref<1200x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%39) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %40 = memref_cast %14 : memref<?xmemref<1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%40) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<1400x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %2 = cmpi "slt", %1, %c1400_i32 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%3: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %4 = cmpi "slt", %3, %c1200_i32 : i32
// CHECK-NEXT:     cond_br %4, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %5 = index_cast %1 : i32 to index
// CHECK-NEXT:     %6 = addi %c0, %5 : index
// CHECK-NEXT:     %7 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %8 = index_cast %3 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %10 = sitofp %1 : i32 to f64
// CHECK-NEXT:     %11 = sitofp %3 : i32 to f64
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = sitofp %c1200_i32 : i32 to f64
// CHECK-NEXT:     %14 = divf %12, %13 : f64
// CHECK-NEXT:     store %14, %7[%6, %9] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %15 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%15 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %1, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%16 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%4: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %5 = cmpi "slt", %4, %arg1 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %6 = index_cast %4 : i32 to index
// CHECK-NEXT:     %7 = addi %c0, %6 : index
// CHECK-NEXT:     %8 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %9 = load %8[%7, %3] : memref<?x1200xf64>
// CHECK-NEXT:     %10 = load %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %11 = addf %10, %9 : f64
// CHECK-NEXT:     store %11, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %12 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%12 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %13 = load %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %14 = divf %13, %arg2 : f64
// CHECK-NEXT:     store %14, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %15 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%15 : i32)
// CHECK-NEXT:   ^bb7(%16: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %17 = cmpi "slt", %16, %arg1 : i32
// CHECK-NEXT:     cond_br %17, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%18: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %19 = cmpi "slt", %18, %arg0 : i32
// CHECK-NEXT:     cond_br %19, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %20 = index_cast %16 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %23 = index_cast %18 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %arg5[%24] : memref<1200xf64>
// CHECK-NEXT:     %26 = load %22[%21, %24] : memref<?x1200xf64>
// CHECK-NEXT:     %27 = subf %26, %25 : f64
// CHECK-NEXT:     store %27, %22[%21, %24] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %28 = addi %18, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%28 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %29 = addi %16, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%29 : i32)
// CHECK-NEXT:   ^bb13(%30: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %31 = cmpi "slt", %30, %arg0 : i32
// CHECK-NEXT:     cond_br %31, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%30 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb16(%32: i32):  // 2 preds: ^bb14, ^bb21
// CHECK-NEXT:     %33 = cmpi "slt", %32, %arg0 : i32
// CHECK-NEXT:     cond_br %33, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %34 = index_cast %30 : i32 to index
// CHECK-NEXT:     %35 = addi %c0, %34 : index
// CHECK-NEXT:     %36 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %37 = index_cast %32 : i32 to index
// CHECK-NEXT:     %38 = addi %c0, %37 : index
// CHECK-NEXT:     %cst_3 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_3, %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %39 = addi %30, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb13(%39 : i32)
// CHECK-NEXT:   ^bb19(%40: i32):  // 2 preds: ^bb17, ^bb20
// CHECK-NEXT:     %41 = cmpi "slt", %40, %arg1 : i32
// CHECK-NEXT:     cond_br %41, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %42 = index_cast %40 : i32 to index
// CHECK-NEXT:     %43 = addi %c0, %42 : index
// CHECK-NEXT:     %44 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %45 = load %44[%43, %35] : memref<?x1200xf64>
// CHECK-NEXT:     %46 = load %44[%43, %38] : memref<?x1200xf64>
// CHECK-NEXT:     %47 = mulf %45, %46 : f64
// CHECK-NEXT:     %48 = load %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     %49 = addf %48, %47 : f64
// CHECK-NEXT:     store %49, %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %50 = addi %40, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb19(%50 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %cst_6 = constant 1.000000e+00 : f64
// CHECK-NEXT:     %51 = subf %arg2, %cst_6 : f64
// CHECK-NEXT:     %52 = load %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     %53 = divf %52, %51 : f64
// CHECK-NEXT:     store %53, %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     %54 = load %36[%35, %38] : memref<?x1200xf64>
// CHECK-NEXT:     store %54, %36[%38, %35] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %55 = addi %32, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb16(%55 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = muli %0, %arg0 : i32
// CHECK-NEXT:     %5 = addi %4, %2 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %6 = remi_signed %5, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%7 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %8 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%8 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }