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
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("corr\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloc() : memref<1400x1200xf64>
// CHECK-NEXT:     %2 = alloc() : memref<1200x1200xf64>
// CHECK-NEXT:     %3 = alloc() : memref<1200xf64>
// CHECK-NEXT:     %4 = alloc() : memref<1200xf64>
// CHECK-NEXT:     %5 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %6 = memref_cast %1 : memref<1400x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?x1200xf64> to memref<1400x1200xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1400_i32, %5, %7) : (i32, i32, memref<?xf64>, memref<1400x1200xf64>) -> ()
// CHECK-NEXT:     %8 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %9 = memref_cast %2 : memref<1200x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?x1200xf64> to memref<1200x1200xf64>
// CHECK-NEXT:     %11 = memref_cast %3 : memref<1200xf64> to memref<?xf64>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xf64> to memref<1200xf64>
// CHECK-NEXT:     %13 = memref_cast %4 : memref<1200xf64> to memref<?xf64>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xf64> to memref<1200xf64>
// CHECK-NEXT:     call @kernel_correlation(%c1200_i32, %c1400_i32, %8, %7, %10, %12, %14) : (i32, i32, f64, memref<1400x1200xf64>, memref<1200x1200xf64>, memref<1200xf64>, memref<1200xf64>) -> ()
// CHECK-NEXT:     %15 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %16 = trunci %c0_i32 : i32 to i1
// CHECK-NEXT:     %17 = xor %16, %true : i1
// CHECK-NEXT:     %18 = and %15, %17 : i1
// CHECK-NEXT:     scf.if %18 {
// CHECK-NEXT:       call @print_array(%c1200_i32, %10) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<1400x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = sitofp %c1400_i32 : i32 to f64
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %2 = cmpi "slt", %1, %c1400_i32 : i32
// CHECK-NEXT:     cond_br %2, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %4 = cmpi "slt", %3, %c1200_i32 : i32
// CHECK-NEXT:     cond_br %4, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %5 = index_cast %1 : i32 to index
// CHECK-NEXT:     %6 = index_cast %3 : i32 to index
// CHECK-NEXT:     %7 = muli %1, %3 : i32
// CHECK-NEXT:     %8 = sitofp %7 : i32 to f64
// CHECK-NEXT:     %9 = sitofp %c1200_i32 : i32 to f64
// CHECK-NEXT:     %10 = divf %8, %9 : f64
// CHECK-NEXT:     %11 = sitofp %1 : i32 to f64
// CHECK-NEXT:     %12 = addf %10, %11 : f64
// CHECK-NEXT:     store %12, %arg3[%5, %6] : memref<1400x1200xf64>
// CHECK-NEXT:     %13 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%13 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %14 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>, %arg6: memref<1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.000000e-01 : f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %cst_0 = constant 1.000000e+00 : f64
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = alloca() : memref<1xi32>
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     store %cst_1, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%4: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %5 = cmpi "slt", %4, %arg1 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %6 = index_cast %4 : i32 to index
// CHECK-NEXT:     %7 = load %arg3[%6, %3] : memref<1400x1200xf64>
// CHECK-NEXT:     %8 = load %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %9 = addf %8, %7 : f64
// CHECK-NEXT:     store %9, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %10 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%10 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %11 = load %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %12 = divf %11, %arg2 : f64
// CHECK-NEXT:     store %12, %arg5[%3] : memref<1200xf64>
// CHECK-NEXT:     %13 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     store %13, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%13 : i32)
// CHECK-NEXT:   ^bb7(%14: i32):  // 2 preds: ^bb3, ^bb11
// CHECK-NEXT:     %15 = cmpi "slt", %14, %arg0 : i32
// CHECK-NEXT:     cond_br %15, ^bb8, ^bb12(%c0_i32 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %16 = index_cast %14 : i32 to index
// CHECK-NEXT:     store %cst_1, %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     br ^bb9(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9(%17: i32):  // 2 preds: ^bb8, ^bb10
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg1 : i32
// CHECK-NEXT:     cond_br %18, ^bb10, ^bb11
// CHECK-NEXT:   ^bb10:  // pred: ^bb9
// CHECK-NEXT:     %19 = index_cast %17 : i32 to index
// CHECK-NEXT:     %20 = load %arg3[%19, %16] : memref<1400x1200xf64>
// CHECK-NEXT:     %21 = load %arg5[%16] : memref<1200xf64>
// CHECK-NEXT:     %22 = subf %20, %21 : f64
// CHECK-NEXT:     %23 = load %arg3[%19, %16] : memref<1400x1200xf64>
// CHECK-NEXT:     %24 = load %arg5[%16] : memref<1200xf64>
// CHECK-NEXT:     %25 = subf %23, %24 : f64
// CHECK-NEXT:     %26 = mulf %22, %25 : f64
// CHECK-NEXT:     %27 = load %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %28 = addf %27, %26 : f64
// CHECK-NEXT:     store %28, %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %29 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     br ^bb9(%29 : i32)
// CHECK-NEXT:   ^bb11:  // pred: ^bb9
// CHECK-NEXT:     %30 = load %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %31 = divf %30, %arg2 : f64
// CHECK-NEXT:     store %31, %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %32 = load %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %33 = sqrt %32 : f64
// CHECK-NEXT:     store %33, %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %34 = load %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %35 = cmpf "ule", %34, %cst : f64
// CHECK-NEXT:     %36 = scf.if %35 -> (f64) {
// CHECK-NEXT:       scf.yield %cst_0 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %74 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %75 = index_cast %74 : i32 to index
// CHECK-NEXT:       %76 = load %arg6[%75] : memref<1200xf64>
// CHECK-NEXT:       scf.yield %76 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     store %36, %arg6[%16] : memref<1200xf64>
// CHECK-NEXT:     %37 = addi %14, %c1_i32 : i32
// CHECK-NEXT:     store %37, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%37 : i32)
// CHECK-NEXT:   ^bb12(%38: i32):  // 2 preds: ^bb7, ^bb16
// CHECK-NEXT:     %39 = cmpi "slt", %38, %arg1 : i32
// CHECK-NEXT:     cond_br %39, ^bb13, ^bb17(%c0_i32 : i32)
// CHECK-NEXT:   ^bb13:  // pred: ^bb12
// CHECK-NEXT:     store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb14(%c0_i32 : i32)
// CHECK-NEXT:   ^bb14(%40: i32):  // 2 preds: ^bb13, ^bb15
// CHECK-NEXT:     %41 = cmpi "slt", %40, %arg0 : i32
// CHECK-NEXT:     cond_br %41, ^bb15, ^bb16
// CHECK-NEXT:   ^bb15:  // pred: ^bb14
// CHECK-NEXT:     %42 = index_cast %38 : i32 to index
// CHECK-NEXT:     %43 = index_cast %40 : i32 to index
// CHECK-NEXT:     %44 = load %arg5[%43] : memref<1200xf64>
// CHECK-NEXT:     %45 = load %arg3[%42, %43] : memref<1400x1200xf64>
// CHECK-NEXT:     %46 = subf %45, %44 : f64
// CHECK-NEXT:     store %46, %arg3[%42, %43] : memref<1400x1200xf64>
// CHECK-NEXT:     %47 = sqrt %arg2 : f64
// CHECK-NEXT:     %48 = load %arg6[%43] : memref<1200xf64>
// CHECK-NEXT:     %49 = mulf %47, %48 : f64
// CHECK-NEXT:     %50 = load %arg3[%42, %43] : memref<1400x1200xf64>
// CHECK-NEXT:     %51 = divf %50, %49 : f64
// CHECK-NEXT:     store %51, %arg3[%42, %43] : memref<1400x1200xf64>
// CHECK-NEXT:     %52 = addi %40, %c1_i32 : i32
// CHECK-NEXT:     store %52, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb14(%52 : i32)
// CHECK-NEXT:   ^bb16:  // pred: ^bb14
// CHECK-NEXT:     %53 = addi %38, %c1_i32 : i32
// CHECK-NEXT:     br ^bb12(%53 : i32)
// CHECK-NEXT:   ^bb17(%54: i32):  // 2 preds: ^bb12, ^bb20
// CHECK-NEXT:     %55 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     %56 = cmpi "slt", %54, %55 : i32
// CHECK-NEXT:     cond_br %56, ^bb18, ^bb19
// CHECK-NEXT:   ^bb18:  // pred: ^bb17
// CHECK-NEXT:     %57 = index_cast %54 : i32 to index
// CHECK-NEXT:     store %cst_0, %arg4[%57, %57] : memref<1200x1200xf64>
// CHECK-NEXT:     %58 = addi %54, %c1_i32 : i32
// CHECK-NEXT:     store %58, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb20(%58 : i32)
// CHECK-NEXT:   ^bb19:  // pred: ^bb17
// CHECK-NEXT:     %59 = index_cast %55 : i32 to index
// CHECK-NEXT:     store %cst_0, %arg4[%59, %59] : memref<1200x1200xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb20(%60: i32):  // 2 preds: ^bb18, ^bb24
// CHECK-NEXT:     %61 = cmpi "slt", %60, %arg0 : i32
// CHECK-NEXT:     cond_br %61, ^bb21, ^bb17(%58 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb20
// CHECK-NEXT:     %62 = index_cast %60 : i32 to index
// CHECK-NEXT:     store %cst_1, %arg4[%57, %62] : memref<1200x1200xf64>
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb22(%63: i32):  // 2 preds: ^bb21, ^bb23
// CHECK-NEXT:     %64 = cmpi "slt", %63, %arg1 : i32
// CHECK-NEXT:     cond_br %64, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %65 = index_cast %63 : i32 to index
// CHECK-NEXT:     %66 = load %arg3[%65, %57] : memref<1400x1200xf64>
// CHECK-NEXT:     %67 = load %arg3[%65, %62] : memref<1400x1200xf64>
// CHECK-NEXT:     %68 = mulf %66, %67 : f64
// CHECK-NEXT:     %69 = load %arg4[%57, %62] : memref<1200x1200xf64>
// CHECK-NEXT:     %70 = addf %69, %68 : f64
// CHECK-NEXT:     store %70, %arg4[%57, %62] : memref<1200x1200xf64>
// CHECK-NEXT:     %71 = addi %63, %c1_i32 : i32
// CHECK-NEXT:     br ^bb22(%71 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %72 = load %arg4[%57, %62] : memref<1200x1200xf64>
// CHECK-NEXT:     store %72, %arg4[%62, %57] : memref<1200x1200xf64>
// CHECK-NEXT:     %73 = addi %60, %c1_i32 : i32
// CHECK-NEXT:     store %73, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb20(%73 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @str0 : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @str1 : !llvm.ptr<array<15 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<5 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<5 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %16 = llvm.load %15 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @str5 : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %18 = llvm.getelementptr %17[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %19 = llvm.mlir.addressof @str2 : !llvm.ptr<array<5 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<5 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = llvm.call @fprintf(%16, %18, %20) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %22 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %23 = llvm.load %22 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %25 = llvm.getelementptr %24[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %26 = llvm.call @fprintf(%23, %25) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%27: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg0 : i32
// CHECK-NEXT:     cond_br %28, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %29 = muli %13, %arg0 : i32
// CHECK-NEXT:     %30 = addi %29, %27 : i32
// CHECK-NEXT:     %31 = remi_signed %30, %c20_i32 : i32
// CHECK-NEXT:     %32 = cmpi "eq", %31, %c0_i32 : i32
// CHECK-NEXT:     scf.if %32 {
// CHECK-NEXT:       %44 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %45 = llvm.load %44 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %46 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @str4 : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = index_cast %13 : i32 to index
// CHECK-NEXT:     %38 = index_cast %27 : i32 to index
// CHECK-NEXT:     %39 = load %arg1[%37, %38] : memref<1200x1200xf64>
// CHECK-NEXT:     %40 = llvm.mlir.cast %39 : f64 to !llvm.double
// CHECK-NEXT:     %41 = llvm.call @fprintf(%34, %36, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %42 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%42 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %43 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%43 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }