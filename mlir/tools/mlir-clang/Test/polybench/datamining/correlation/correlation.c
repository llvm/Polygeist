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
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
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
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("corr")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
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
// CHECK-NEXT:     %15 = call @polybench_alloc_data(%12, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?xi8> to memref<?xmemref<1200xf64>>
// CHECK-NEXT:     %17 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %18 = load %7[%c0] : memref<?xmemref<1400x1200xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x1200xf64> to memref<1400x1200xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1400_i32, %17, %20) : (i32, i32, memref<?xf64>, memref<1400x1200xf64>) -> ()
// CHECK-NEXT:     %21 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %22 = load %7[%c0] : memref<?xmemref<1400x1200xf64>>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?x1200xf64> to memref<1400x1200xf64>
// CHECK-NEXT:     %25 = load %11[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %28 = load %14[%c0] : memref<?xmemref<1200xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<1200xf64> to memref<?xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?xf64> to memref<1200xf64>
// CHECK-NEXT:     %31 = load %16[%c0] : memref<?xmemref<1200xf64>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<1200xf64> to memref<?xf64>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?xf64> to memref<1200xf64>
// CHECK-NEXT:     call @kernel_correlation(%c1200_i32, %c1400_i32, %21, %24, %27, %30, %33) : (i32, i32, f64, memref<1400x1200xf64>, memref<1200x1200xf64>, memref<1200xf64>, memref<1200xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %34 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %35 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %arg1[%36] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %38 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %39 = call @strcmp(%37, %38) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %40 = trunci %39 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %41 = xor %40, %true : i1
// CHECK-NEXT:     %42 = and %34, %41 : i1
// CHECK-NEXT:     scf.if %42 {
// CHECK-NEXT:       %47 = load %11[%c0] : memref<?xmemref<1200x1200xf64>>
// CHECK-NEXT:       %48 = memref_cast %47 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %49 = memref_cast %48 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:       call @print_array(%c1200_i32, %49) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %43 = memref_cast %7 : memref<?xmemref<1400x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%43) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %44 = memref_cast %11 : memref<?xmemref<1200x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%44) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %45 = memref_cast %14 : memref<?xmemref<1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%45) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %46 = memref_cast %16 : memref<?xmemref<1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%46) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<1400x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %0 = sitofp %c1400_i32 : i32 to f64
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
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
// CHECK-NEXT:     %10 = muli %1, %3 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = sitofp %c1200_i32 : i32 to f64
// CHECK-NEXT:     %13 = divf %11, %12 : f64
// CHECK-NEXT:     %14 = sitofp %1 : i32 to f64
// CHECK-NEXT:     %15 = addf %13, %14 : f64
// CHECK-NEXT:     store %15, %7[%6, %9] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%16 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %1, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%17 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>, %arg6: memref<1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = alloca() : memref<1xi32>
// CHECK-NEXT:     %cst = constant 1.000000e-01 : f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = addi %c0, %3 : index
// CHECK-NEXT:     %cst_0 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_0, %arg5[%4] : memref<1200xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%5: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg1 : i32
// CHECK-NEXT:     cond_br %6, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %7 = index_cast %5 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %10 = load %9[%8, %4] : memref<?x1200xf64>
// CHECK-NEXT:     %11 = load %arg5[%4] : memref<1200xf64>
// CHECK-NEXT:     %12 = addf %11, %10 : f64
// CHECK-NEXT:     store %12, %arg5[%4] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %13 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%13 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %14 = load %arg5[%4] : memref<1200xf64>
// CHECK-NEXT:     %15 = divf %14, %arg2 : f64
// CHECK-NEXT:     store %15, %arg5[%4] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %1, %c1_i32_1 : i32
// CHECK-NEXT:     store %16, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%16 : i32)
// CHECK-NEXT:   ^bb7(%17: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg0 : i32
// CHECK-NEXT:     cond_br %18, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %19 = index_cast %17 : i32 to index
// CHECK-NEXT:     %20 = addi %c0, %19 : index
// CHECK-NEXT:     %cst_2 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_2, %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%21: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %22 = cmpi "slt", %21, %arg1 : i32
// CHECK-NEXT:     cond_br %22, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %23 = index_cast %21 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %26 = load %25[%24, %20] : memref<?x1200xf64>
// CHECK-NEXT:     %27 = load %arg5[%20] : memref<1200xf64>
// CHECK-NEXT:     %28 = subf %26, %27 : f64
// CHECK-NEXT:     %29 = load %25[%24, %20] : memref<?x1200xf64>
// CHECK-NEXT:     %30 = load %arg5[%20] : memref<1200xf64>
// CHECK-NEXT:     %31 = subf %29, %30 : f64
// CHECK-NEXT:     %32 = mulf %28, %31 : f64
// CHECK-NEXT:     %33 = load %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %34 = addf %33, %32 : f64
// CHECK-NEXT:     store %34, %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %35 = addi %21, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb10(%35 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %36 = load %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %37 = divf %36, %arg2 : f64
// CHECK-NEXT:     store %37, %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %38 = load %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %39 = sqrt %38 : f64
// CHECK-NEXT:     store %39, %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %40 = load %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %41 = cmpf "ule", %40, %cst : f64
// CHECK-NEXT:     %42 = scf.if %41 -> (f64) {
// CHECK-NEXT:       %cst_11 = constant 1.000000e+00 : f64
// CHECK-NEXT:       scf.yield %cst_11 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %90 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %91 = index_cast %90 : i32 to index
// CHECK-NEXT:       %92 = addi %c0, %91 : index
// CHECK-NEXT:       %93 = load %arg6[%92] : memref<1200xf64>
// CHECK-NEXT:       scf.yield %93 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     store %42, %arg6[%20] : memref<1200xf64>
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %43 = addi %17, %c1_i32_4 : i32
// CHECK-NEXT:     store %43, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%43 : i32)
// CHECK-NEXT:   ^bb13(%44: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %45 = cmpi "slt", %44, %arg1 : i32
// CHECK-NEXT:     cond_br %45, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%46: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %47 = cmpi "slt", %46, %arg0 : i32
// CHECK-NEXT:     cond_br %47, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %48 = index_cast %44 : i32 to index
// CHECK-NEXT:     %49 = addi %c0, %48 : index
// CHECK-NEXT:     %50 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %51 = index_cast %46 : i32 to index
// CHECK-NEXT:     %52 = addi %c0, %51 : index
// CHECK-NEXT:     %53 = load %arg5[%52] : memref<1200xf64>
// CHECK-NEXT:     %54 = load %50[%49, %52] : memref<?x1200xf64>
// CHECK-NEXT:     %55 = subf %54, %53 : f64
// CHECK-NEXT:     store %55, %50[%49, %52] : memref<?x1200xf64>
// CHECK-NEXT:     %56 = sqrt %arg2 : f64
// CHECK-NEXT:     %57 = load %arg6[%52] : memref<1200xf64>
// CHECK-NEXT:     %58 = mulf %56, %57 : f64
// CHECK-NEXT:     %59 = load %50[%49, %52] : memref<?x1200xf64>
// CHECK-NEXT:     %60 = divf %59, %58 : f64
// CHECK-NEXT:     store %60, %50[%49, %52] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %61 = addi %46, %c1_i32_5 : i32
// CHECK-NEXT:     store %61, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb16(%61 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %62 = addi %44, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb13(%62 : i32)
// CHECK-NEXT:   ^bb19(%63: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %64 = subi %arg0, %c1_i32_7 : i32
// CHECK-NEXT:     %65 = cmpi "slt", %63, %64 : i32
// CHECK-NEXT:     cond_br %65, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %66 = index_cast %63 : i32 to index
// CHECK-NEXT:     %67 = addi %c0, %66 : index
// CHECK-NEXT:     %68 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %cst_8 = constant 1.000000e+00 : f64
// CHECK-NEXT:     store %cst_8, %68[%67, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %69 = addi %63, %c1_i32_7 : i32
// CHECK-NEXT:     store %69, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb22(%69 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %70 = index_cast %64 : i32 to index
// CHECK-NEXT:     %71 = addi %c0, %70 : index
// CHECK-NEXT:     %72 = memref_cast %arg4 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %cst_9 = constant 1.000000e+00 : f64
// CHECK-NEXT:     store %cst_9, %72[%71, %71] : memref<?x1200xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb22(%73: i32):  // 2 preds: ^bb20, ^bb27
// CHECK-NEXT:     %74 = cmpi "slt", %73, %arg0 : i32
// CHECK-NEXT:     cond_br %74, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %75 = index_cast %73 : i32 to index
// CHECK-NEXT:     %76 = addi %c0, %75 : index
// CHECK-NEXT:     %cst_10 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_10, %68[%67, %76] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb25(%c0_i32 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     br ^bb19(%69 : i32)
// CHECK-NEXT:   ^bb25(%77: i32):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:     %78 = cmpi "slt", %77, %arg1 : i32
// CHECK-NEXT:     cond_br %78, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %79 = index_cast %77 : i32 to index
// CHECK-NEXT:     %80 = addi %c0, %79 : index
// CHECK-NEXT:     %81 = memref_cast %arg3 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %82 = load %81[%80, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %83 = load %81[%80, %76] : memref<?x1200xf64>
// CHECK-NEXT:     %84 = mulf %82, %83 : f64
// CHECK-NEXT:     %85 = load %68[%67, %76] : memref<?x1200xf64>
// CHECK-NEXT:     %86 = addf %85, %84 : f64
// CHECK-NEXT:     store %86, %68[%67, %76] : memref<?x1200xf64>
// CHECK-NEXT:     %87 = addi %77, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb25(%87 : i32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %88 = load %68[%67, %76] : memref<?x1200xf64>
// CHECK-NEXT:     store %88, %68[%76, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %89 = addi %73, %c1_i32_7 : i32
// CHECK-NEXT:     store %89, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb22(%89 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
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
// CHECK-NEXT:     %30 = llvm.mlir.addressof @str4 : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %31 = llvm.getelementptr %30[%2, %2] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %32 = index_cast %11 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = memref_cast %arg1 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %35 = index_cast %23 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %34[%33, %36] : memref<?x1200xf64>
// CHECK-NEXT:     %38 = llvm.mlir.cast %37 : f64 to !llvm.double
// CHECK-NEXT:     %39 = llvm.call @fprintf(%29, %31, %38) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
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