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
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %2 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = muli %2, %2 : i32
// CHECK-NEXT:     %4 = zexti %3 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %5 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %8 = zexti %2 : i32 to i64
// CHECK-NEXT:     %9 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %11 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %15 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %17 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %19 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %21 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %23 = call @polybench_alloc_data(%8, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %25 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %26 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %27 = load %7[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %30 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %33 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %34 = memref_cast %33 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %36 = load %14[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %37 = memref_cast %36 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %38 = memref_cast %37 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %39 = load %16[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %40 = memref_cast %39 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %41 = memref_cast %40 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %42 = load %18[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %43 = memref_cast %42 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %44 = memref_cast %43 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %45 = load %20[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %46 = memref_cast %45 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %47 = memref_cast %46 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %48 = load %22[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %49 = memref_cast %48 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %50 = memref_cast %49 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %51 = load %24[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %52 = memref_cast %51 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %53 = memref_cast %52 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %25, %26, %29, %32, %35, %38, %41, %44, %47, %50, %53) : (i32, memref<?xf64>, memref<?xf64>, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %54 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %55 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %56 = load %7[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %57 = memref_cast %56 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %58 = memref_cast %57 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %59 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %60 = memref_cast %59 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %61 = memref_cast %60 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %62 = load %12[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %63 = memref_cast %62 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %64 = memref_cast %63 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %65 = load %14[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %66 = memref_cast %65 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %67 = memref_cast %66 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %68 = load %16[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %69 = memref_cast %68 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %70 = memref_cast %69 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %71 = load %18[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %72 = memref_cast %71 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %73 = memref_cast %72 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %74 = load %20[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %75 = memref_cast %74 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %76 = memref_cast %75 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %77 = load %22[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %78 = memref_cast %77 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %79 = memref_cast %78 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %80 = load %24[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %81 = memref_cast %80 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %82 = memref_cast %81 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @kernel_gemver(%c2000_i32, %54, %55, %58, %61, %64, %67, %70, %73, %76, %79, %82) : (i32, f64, f64, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %83 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %84 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %85 = addi %c0, %84 : index
// CHECK-NEXT:     %86 = load %arg1[%85] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %87 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %88 = call @strcmp(%86, %87) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %89 = trunci %88 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %90 = xor %89, %true : i1
// CHECK-NEXT:     %91 = and %83, %90 : i1
// CHECK-NEXT:     scf.if %91 {
// CHECK-NEXT:       %101 = load %18[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %102 = memref_cast %101 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %103 = memref_cast %102 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %103) : (i32, memref<2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %92 = memref_cast %7 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%92) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %93 = memref_cast %10 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%93) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %94 = memref_cast %12 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%94) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %95 = memref_cast %14 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%95) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %96 = memref_cast %16 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%96) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %97 = memref_cast %18 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%97) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %98 = memref_cast %20 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%98) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %99 = memref_cast %22 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%99) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %100 = memref_cast %24 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%100) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     store %cst, %arg1[%c0] : memref<?xf64>
// CHECK-NEXT:     %cst_0 = constant 1.200000e+00 : f64
// CHECK-NEXT:     store %cst_0, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     %0 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = addi %c0, %3 : index
// CHECK-NEXT:     %5 = sitofp %1 : i32 to f64
// CHECK-NEXT:     store %5, %arg4[%4] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %6 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     %7 = sitofp %6 : i32 to f64
// CHECK-NEXT:     %8 = divf %7, %0 : f64
// CHECK-NEXT:     %cst_1 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %9 = divf %8, %cst_1 : f64
// CHECK-NEXT:     store %9, %arg6[%4] : memref<2000xf64>
// CHECK-NEXT:     %cst_2 = constant 4.000000e+00 : f64
// CHECK-NEXT:     %10 = divf %8, %cst_2 : f64
// CHECK-NEXT:     store %10, %arg5[%4] : memref<2000xf64>
// CHECK-NEXT:     %cst_3 = constant 6.000000e+00 : f64
// CHECK-NEXT:     %11 = divf %8, %cst_3 : f64
// CHECK-NEXT:     store %11, %arg7[%4] : memref<2000xf64>
// CHECK-NEXT:     %cst_4 = constant 8.000000e+00 : f64
// CHECK-NEXT:     %12 = divf %8, %cst_4 : f64
// CHECK-NEXT:     store %12, %arg10[%4] : memref<2000xf64>
// CHECK-NEXT:     %cst_5 = constant 9.000000e+00 : f64
// CHECK-NEXT:     %13 = divf %8, %cst_5 : f64
// CHECK-NEXT:     store %13, %arg11[%4] : memref<2000xf64>
// CHECK-NEXT:     %cst_6 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_6, %arg9[%4] : memref<2000xf64>
// CHECK-NEXT:     store %cst_6, %arg8[%4] : memref<2000xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%14: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %15 = cmpi "slt", %14, %arg0 : i32
// CHECK-NEXT:     cond_br %15, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %16 = memref_cast %arg3 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %17 = index_cast %14 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = muli %1, %14 : i32
// CHECK-NEXT:     %20 = remi_signed %19, %arg0 : i32
// CHECK-NEXT:     %21 = sitofp %20 : i32 to f64
// CHECK-NEXT:     %22 = divf %21, %0 : f64
// CHECK-NEXT:     store %22, %16[%4, %18] : memref<?x2000xf64>
// CHECK-NEXT:     %23 = addi %14, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%23 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb1(%6 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) {
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
// CHECK-NEXT:     %6 = memref_cast %arg3 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = load %6[%5, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %10 = load %arg4[%5] : memref<2000xf64>
// CHECK-NEXT:     %11 = load %arg5[%8] : memref<2000xf64>
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = addf %9, %12 : f64
// CHECK-NEXT:     %14 = load %arg6[%5] : memref<2000xf64>
// CHECK-NEXT:     %15 = load %arg7[%8] : memref<2000xf64>
// CHECK-NEXT:     %16 = mulf %14, %15 : f64
// CHECK-NEXT:     %17 = addf %13, %16 : f64
// CHECK-NEXT:     store %17, %6[%5, %8] : memref<?x2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %18 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%18 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %19 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%19 : i32)
// CHECK-NEXT:   ^bb7(%20: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %21 = cmpi "slt", %20, %arg0 : i32
// CHECK-NEXT:     cond_br %21, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%22: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %23 = cmpi "slt", %22, %arg0 : i32
// CHECK-NEXT:     cond_br %23, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %24 = index_cast %20 : i32 to index
// CHECK-NEXT:     %25 = addi %c0, %24 : index
// CHECK-NEXT:     %26 = load %arg9[%25] : memref<2000xf64>
// CHECK-NEXT:     %27 = index_cast %22 : i32 to index
// CHECK-NEXT:     %28 = addi %c0, %27 : index
// CHECK-NEXT:     %29 = memref_cast %arg3 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %30 = load %29[%28, %25] : memref<?x2000xf64>
// CHECK-NEXT:     %31 = mulf %arg2, %30 : f64
// CHECK-NEXT:     %32 = load %arg10[%28] : memref<2000xf64>
// CHECK-NEXT:     %33 = mulf %31, %32 : f64
// CHECK-NEXT:     %34 = addf %26, %33 : f64
// CHECK-NEXT:     store %34, %arg9[%25] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %35 = addi %22, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb10(%35 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %36 = addi %20, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb7(%36 : i32)
// CHECK-NEXT:   ^bb13(%37: i32):  // 2 preds: ^bb9, ^bb14
// CHECK-NEXT:     %38 = cmpi "slt", %37, %arg0 : i32
// CHECK-NEXT:     cond_br %38, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %39 = index_cast %37 : i32 to index
// CHECK-NEXT:     %40 = addi %c0, %39 : index
// CHECK-NEXT:     %41 = load %arg9[%40] : memref<2000xf64>
// CHECK-NEXT:     %42 = load %arg11[%40] : memref<2000xf64>
// CHECK-NEXT:     %43 = addf %41, %42 : f64
// CHECK-NEXT:     store %43, %arg9[%40] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %44 = addi %37, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb13(%44 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%45: i32):  // 2 preds: ^bb15, ^bb21
// CHECK-NEXT:     %46 = cmpi "slt", %45, %arg0 : i32
// CHECK-NEXT:     cond_br %46, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb19(%47: i32):  // 2 preds: ^bb17, ^bb20
// CHECK-NEXT:     %48 = cmpi "slt", %47, %arg0 : i32
// CHECK-NEXT:     cond_br %48, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %49 = index_cast %45 : i32 to index
// CHECK-NEXT:     %50 = addi %c0, %49 : index
// CHECK-NEXT:     %51 = load %arg8[%50] : memref<2000xf64>
// CHECK-NEXT:     %52 = memref_cast %arg3 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %53 = index_cast %47 : i32 to index
// CHECK-NEXT:     %54 = addi %c0, %53 : index
// CHECK-NEXT:     %55 = load %52[%50, %54] : memref<?x2000xf64>
// CHECK-NEXT:     %56 = mulf %arg1, %55 : f64
// CHECK-NEXT:     %57 = load %arg9[%54] : memref<2000xf64>
// CHECK-NEXT:     %58 = mulf %56, %57 : f64
// CHECK-NEXT:     %59 = addf %51, %58 : f64
// CHECK-NEXT:     store %59, %arg8[%50] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %60 = addi %47, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb19(%60 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %61 = addi %45, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb16(%61 : i32)
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