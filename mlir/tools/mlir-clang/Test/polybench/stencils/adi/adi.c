// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | opt -O3 -S | lli - &> %s.out2
// RUN: rm -f %s.exec1
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

 for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }

      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("u\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     %6 = addi %0, %arg0 : i32
// CHECK-NEXT:     %7 = subi %6, %2 : i32
// CHECK-NEXT:     %8 = sitofp %7 : i32 to f64
// CHECK-NEXT:     %9 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %10 = divf %8, %9 : f64
// CHECK-NEXT:     store %10, %arg1[%4, %5] : memref<1000x1000xf64>
// CHECK-NEXT:     %11 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%11 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %12 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%12 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %0 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %1 = divf %cst, %0 : f64
// CHECK-NEXT:     %2 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %3 = divf %cst, %2 : f64
// CHECK-NEXT:     %4 = mulf %cst_0, %3 : f64
// CHECK-NEXT:     %5 = mulf %1, %1 : f64
// CHECK-NEXT:     %6 = divf %4, %5 : f64
// CHECK-NEXT:     %7 = mulf %cst, %3 : f64
// CHECK-NEXT:     %8 = divf %7, %5 : f64
// CHECK-NEXT:     %9 = negf %6 : f64
// CHECK-NEXT:     %10 = divf %9, %cst_0 : f64
// CHECK-NEXT:     %11 = addf %cst, %6 : f64
// CHECK-NEXT:     %12 = negf %8 : f64
// CHECK-NEXT:     %13 = divf %12, %cst_0 : f64
// CHECK-NEXT:     %14 = addf %cst, %8 : f64
// CHECK-NEXT:     br ^bb1(%c1_i32 : i32)
// CHECK-NEXT:   ^bb1(%15: i32):  // 2 preds: ^bb0, ^bb13
// CHECK-NEXT:     %16 = cmpi "sle", %15, %arg0 : i32
// CHECK-NEXT:     cond_br %16, ^bb3(%c1_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%17: i32):  // 2 preds: ^bb1, ^bb10
// CHECK-NEXT:     %18 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     %19 = cmpi "slt", %17, %18 : i32
// CHECK-NEXT:     cond_br %19, ^bb4, ^bb11(%c1_i32 : i32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %20 = index_cast %17 : i32 to index
// CHECK-NEXT:     store %cst, %arg3[%c0, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     store %cst_1, %arg4[%20, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     %21 = load %arg3[%c0, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     store %21, %arg5[%20, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     br ^bb5(%c1_i32 : i32)
// CHECK-NEXT:   ^bb5(%22: i32):  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:     %23 = cmpi "slt", %22, %18 : i32
// CHECK-NEXT:     cond_br %23, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %24 = index_cast %22 : i32 to index
// CHECK-NEXT:     %25 = negf %10 : f64
// CHECK-NEXT:     %26 = subi %22, %c1_i32 : i32
// CHECK-NEXT:     %27 = index_cast %26 : i32 to index
// CHECK-NEXT:     %28 = load %arg4[%20, %27] : memref<1000x1000xf64>
// CHECK-NEXT:     %29 = mulf %10, %28 : f64
// CHECK-NEXT:     %30 = addf %29, %11 : f64
// CHECK-NEXT:     %31 = divf %25, %30 : f64
// CHECK-NEXT:     store %31, %arg4[%20, %24] : memref<1000x1000xf64>
// CHECK-NEXT:     %32 = negf %13 : f64
// CHECK-NEXT:     %33 = subi %17, %c1_i32 : i32
// CHECK-NEXT:     %34 = index_cast %33 : i32 to index
// CHECK-NEXT:     %35 = load %arg2[%24, %34] : memref<1000x1000xf64>
// CHECK-NEXT:     %36 = mulf %32, %35 : f64
// CHECK-NEXT:     %37 = mulf %cst_0, %13 : f64
// CHECK-NEXT:     %38 = addf %cst, %37 : f64
// CHECK-NEXT:     %39 = load %arg2[%24, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     %40 = mulf %38, %39 : f64
// CHECK-NEXT:     %41 = addf %36, %40 : f64
// CHECK-NEXT:     %42 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     %43 = index_cast %42 : i32 to index
// CHECK-NEXT:     %44 = load %arg2[%24, %43] : memref<1000x1000xf64>
// CHECK-NEXT:     %45 = mulf %13, %44 : f64
// CHECK-NEXT:     %46 = subf %41, %45 : f64
// CHECK-NEXT:     %47 = load %arg5[%20, %27] : memref<1000x1000xf64>
// CHECK-NEXT:     %48 = mulf %10, %47 : f64
// CHECK-NEXT:     %49 = subf %46, %48 : f64
// CHECK-NEXT:     %50 = load %arg4[%20, %27] : memref<1000x1000xf64>
// CHECK-NEXT:     %51 = mulf %10, %50 : f64
// CHECK-NEXT:     %52 = addf %51, %11 : f64
// CHECK-NEXT:     %53 = divf %49, %52 : f64
// CHECK-NEXT:     store %53, %arg5[%20, %24] : memref<1000x1000xf64>
// CHECK-NEXT:     %54 = addi %22, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%54 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %55 = index_cast %18 : i32 to index
// CHECK-NEXT:     store %cst, %arg3[%55, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     %56 = subi %arg1, %c2_i32 : i32
// CHECK-NEXT:     br ^bb8(%56 : i32)
// CHECK-NEXT:   ^bb8(%57: i32):  // 2 preds: ^bb7, ^bb9
// CHECK-NEXT:     %58 = cmpi "sge", %57, %c1_i32 : i32
// CHECK-NEXT:     cond_br %58, ^bb9, ^bb10
// CHECK-NEXT:   ^bb9:  // pred: ^bb8
// CHECK-NEXT:     %59 = index_cast %57 : i32 to index
// CHECK-NEXT:     %60 = load %arg4[%20, %59] : memref<1000x1000xf64>
// CHECK-NEXT:     %61 = addi %57, %c1_i32 : i32
// CHECK-NEXT:     %62 = index_cast %61 : i32 to index
// CHECK-NEXT:     %63 = load %arg3[%62, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     %64 = mulf %60, %63 : f64
// CHECK-NEXT:     %65 = load %arg5[%20, %59] : memref<1000x1000xf64>
// CHECK-NEXT:     %66 = addf %64, %65 : f64
// CHECK-NEXT:     store %66, %arg3[%59, %20] : memref<1000x1000xf64>
// CHECK-NEXT:     %67 = subi %57, %c1_i32 : i32
// CHECK-NEXT:     br ^bb8(%67 : i32)
// CHECK-NEXT:   ^bb10:  // pred: ^bb8
// CHECK-NEXT:     %68 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%68 : i32)
// CHECK-NEXT:   ^bb11(%69: i32):  // 2 preds: ^bb3, ^bb19
// CHECK-NEXT:     %70 = cmpi "slt", %69, %18 : i32
// CHECK-NEXT:     cond_br %70, ^bb12, ^bb13
// CHECK-NEXT:   ^bb12:  // pred: ^bb11
// CHECK-NEXT:     %71 = index_cast %69 : i32 to index
// CHECK-NEXT:     store %cst, %arg2[%71, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     store %cst_1, %arg4[%71, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     %72 = load %arg2[%71, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     store %72, %arg5[%71, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:     br ^bb14(%c1_i32 : i32)
// CHECK-NEXT:   ^bb13:  // pred: ^bb11
// CHECK-NEXT:     %73 = addi %15, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%73 : i32)
// CHECK-NEXT:   ^bb14(%74: i32):  // 2 preds: ^bb12, ^bb15
// CHECK-NEXT:     %75 = cmpi "slt", %74, %18 : i32
// CHECK-NEXT:     cond_br %75, ^bb15, ^bb16
// CHECK-NEXT:   ^bb15:  // pred: ^bb14
// CHECK-NEXT:     %76 = index_cast %74 : i32 to index
// CHECK-NEXT:     %77 = negf %13 : f64
// CHECK-NEXT:     %78 = subi %74, %c1_i32 : i32
// CHECK-NEXT:     %79 = index_cast %78 : i32 to index
// CHECK-NEXT:     %80 = load %arg4[%71, %79] : memref<1000x1000xf64>
// CHECK-NEXT:     %81 = mulf %13, %80 : f64
// CHECK-NEXT:     %82 = addf %81, %14 : f64
// CHECK-NEXT:     %83 = divf %77, %82 : f64
// CHECK-NEXT:     store %83, %arg4[%71, %76] : memref<1000x1000xf64>
// CHECK-NEXT:     %84 = negf %10 : f64
// CHECK-NEXT:     %85 = subi %69, %c1_i32 : i32
// CHECK-NEXT:     %86 = index_cast %85 : i32 to index
// CHECK-NEXT:     %87 = load %arg3[%86, %76] : memref<1000x1000xf64>
// CHECK-NEXT:     %88 = mulf %84, %87 : f64
// CHECK-NEXT:     %89 = mulf %cst_0, %10 : f64
// CHECK-NEXT:     %90 = addf %cst, %89 : f64
// CHECK-NEXT:     %91 = load %arg3[%71, %76] : memref<1000x1000xf64>
// CHECK-NEXT:     %92 = mulf %90, %91 : f64
// CHECK-NEXT:     %93 = addf %88, %92 : f64
// CHECK-NEXT:     %94 = addi %69, %c1_i32 : i32
// CHECK-NEXT:     %95 = index_cast %94 : i32 to index
// CHECK-NEXT:     %96 = load %arg3[%95, %76] : memref<1000x1000xf64>
// CHECK-NEXT:     %97 = mulf %10, %96 : f64
// CHECK-NEXT:     %98 = subf %93, %97 : f64
// CHECK-NEXT:     %99 = load %arg5[%71, %79] : memref<1000x1000xf64>
// CHECK-NEXT:     %100 = mulf %13, %99 : f64
// CHECK-NEXT:     %101 = subf %98, %100 : f64
// CHECK-NEXT:     %102 = load %arg4[%71, %79] : memref<1000x1000xf64>
// CHECK-NEXT:     %103 = mulf %13, %102 : f64
// CHECK-NEXT:     %104 = addf %103, %14 : f64
// CHECK-NEXT:     %105 = divf %101, %104 : f64
// CHECK-NEXT:     store %105, %arg5[%71, %76] : memref<1000x1000xf64>
// CHECK-NEXT:     %106 = addi %74, %c1_i32 : i32
// CHECK-NEXT:     br ^bb14(%106 : i32)
// CHECK-NEXT:   ^bb16:  // pred: ^bb14
// CHECK-NEXT:     %107 = index_cast %18 : i32 to index
// CHECK-NEXT:     store %cst, %arg2[%71, %107] : memref<1000x1000xf64>
// CHECK-NEXT:     %108 = subi %arg1, %c2_i32 : i32
// CHECK-NEXT:     br ^bb17(%108 : i32)
// CHECK-NEXT:   ^bb17(%109: i32):  // 2 preds: ^bb16, ^bb18
// CHECK-NEXT:     %110 = cmpi "sge", %109, %c1_i32 : i32
// CHECK-NEXT:     cond_br %110, ^bb18, ^bb19
// CHECK-NEXT:   ^bb18:  // pred: ^bb17
// CHECK-NEXT:     %111 = index_cast %109 : i32 to index
// CHECK-NEXT:     %112 = load %arg4[%71, %111] : memref<1000x1000xf64>
// CHECK-NEXT:     %113 = addi %109, %c1_i32 : i32
// CHECK-NEXT:     %114 = index_cast %113 : i32 to index
// CHECK-NEXT:     %115 = load %arg2[%71, %114] : memref<1000x1000xf64>
// CHECK-NEXT:     %116 = mulf %112, %115 : f64
// CHECK-NEXT:     %117 = load %arg5[%71, %111] : memref<1000x1000xf64>
// CHECK-NEXT:     %118 = addf %116, %117 : f64
// CHECK-NEXT:     store %118, %arg2[%71, %111] : memref<1000x1000xf64>
// CHECK-NEXT:     %119 = subi %109, %c1_i32 : i32
// CHECK-NEXT:     br ^bb17(%119 : i32)
// CHECK-NEXT:   ^bb19:  // pred: ^bb17
// CHECK-NEXT:     %120 = addi %69, %c1_i32 : i32
// CHECK-NEXT:     br ^bb11(%120 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @[[str0]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @[[str1]] : !llvm.ptr<array<15 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %16 = llvm.load %15 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @[[str5]] : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %18 = llvm.getelementptr %17[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %19 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = llvm.call @fprintf(%16, %18, %20) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %22 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %23 = llvm.load %22 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.mlir.addressof @[[str6]] : !llvm.ptr<array<23 x i8>>
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
// CHECK-NEXT:       %46 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = index_cast %13 : i32 to index
// CHECK-NEXT:     %38 = index_cast %27 : i32 to index
// CHECK-NEXT:     %39 = load %arg1[%37, %38] : memref<1000x1000xf64>
// CHECK-NEXT:     %40 = llvm.mlir.cast %39 : f64 to !llvm.double
// CHECK-NEXT:     %41 = llvm.call @fprintf(%34, %36, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %42 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%42 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %43 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%43 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}