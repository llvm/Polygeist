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
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	tmp[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < _PB_NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
	      alpha, beta,
	      POLYBENCH_ARRAY(tmp),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("D\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c800_i32 = constant 800 : i32
// CHECK-NEXT:     %c900_i32 = constant 900 : i32
// CHECK-NEXT:     %c1100_i32 = constant 1100 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %2 = alloc() : memref<800x900xf64>
// CHECK-NEXT:     %3 = alloc() : memref<800x1100xf64>
// CHECK-NEXT:     %4 = alloc() : memref<1100x900xf64>
// CHECK-NEXT:     %5 = alloc() : memref<900x1200xf64>
// CHECK-NEXT:     %6 = alloc() : memref<800x1200xf64>
// CHECK-NEXT:     %7 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %8 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %9 = memref_cast %3 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?x1100xf64> to memref<800x1100xf64>
// CHECK-NEXT:     %11 = memref_cast %4 : memref<1100x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?x900xf64> to memref<1100x900xf64>
// CHECK-NEXT:     %13 = memref_cast %5 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?x1200xf64> to memref<900x1200xf64>
// CHECK-NEXT:     %15 = memref_cast %6 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?x1200xf64> to memref<800x1200xf64>
// CHECK-NEXT:     call @init_array(%c800_i32, %c900_i32, %c1100_i32, %c1200_i32, %7, %8, %10, %12, %14, %16) : (i32, i32, i32, i32, memref<?xf64>, memref<?xf64>, memref<800x1100xf64>, memref<1100x900xf64>, memref<900x1200xf64>, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     %17 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %18 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %19 = memref_cast %2 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x900xf64> to memref<800x900xf64>
// CHECK-NEXT:     call @kernel_2mm(%c800_i32, %c900_i32, %c1100_i32, %c1200_i32, %17, %18, %20, %10, %12, %14, %16) : (i32, i32, i32, i32, f64, f64, memref<800x900xf64>, memref<800x1100xf64>, memref<1100x900xf64>, memref<900x1200xf64>, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     %21 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %22 = trunci %c0_i32 : i32 to i1
// CHECK-NEXT:     %23 = xor %22, %true : i1
// CHECK-NEXT:     %24 = and %21, %23 : i1
// CHECK-NEXT:     scf.if %24 {
// CHECK-NEXT:       call @print_array(%c800_i32, %c1200_i32, %16) : (i32, i32, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<800x1100xf64>, %arg7: memref<1100x900xf64>, %arg8: memref<900x1200xf64>, %arg9: memref<800x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     %cst_0 = constant 1.200000e+00 : f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     store %cst, %arg4[%c0] : memref<?xf64>
// CHECK-NEXT:     store %cst_0, %arg5[%c0] : memref<?xf64>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2(%c0_i32 : i32), ^bb5(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2(%2: i32):  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg2 : i32
// CHECK-NEXT:     cond_br %3, ^bb3, ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     %6 = muli %0, %2 : i32
// CHECK-NEXT:     %7 = addi %6, %c1_i32 : i32
// CHECK-NEXT:     %8 = remi_signed %7, %arg0 : i32
// CHECK-NEXT:     %9 = sitofp %8 : i32 to f64
// CHECK-NEXT:     %10 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %11 = divf %9, %10 : f64
// CHECK-NEXT:     store %11, %arg6[%4, %5] : memref<800x1100xf64>
// CHECK-NEXT:     %12 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%12 : i32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %13 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%13 : i32)
// CHECK-NEXT:   ^bb5(%14: i32):  // 2 preds: ^bb1, ^bb8
// CHECK-NEXT:     %15 = cmpi "slt", %14, %arg2 : i32
// CHECK-NEXT:     cond_br %15, ^bb6(%c0_i32 : i32), ^bb9(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6(%16: i32):  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:     %17 = cmpi "slt", %16, %arg1 : i32
// CHECK-NEXT:     cond_br %17, ^bb7, ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %18 = index_cast %14 : i32 to index
// CHECK-NEXT:     %19 = index_cast %16 : i32 to index
// CHECK-NEXT:     %20 = addi %16, %c1_i32 : i32
// CHECK-NEXT:     %21 = muli %14, %20 : i32
// CHECK-NEXT:     %22 = remi_signed %21, %arg1 : i32
// CHECK-NEXT:     %23 = sitofp %22 : i32 to f64
// CHECK-NEXT:     %24 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %25 = divf %23, %24 : f64
// CHECK-NEXT:     store %25, %arg7[%18, %19] : memref<1100x900xf64>
// CHECK-NEXT:     br ^bb6(%20 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-NEXT:     %26 = addi %14, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%26 : i32)
// CHECK-NEXT:   ^bb9(%27: i32):  // 2 preds: ^bb5, ^bb12
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg1 : i32
// CHECK-NEXT:     cond_br %28, ^bb10(%c0_i32 : i32), ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%29: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %30 = cmpi "slt", %29, %arg3 : i32
// CHECK-NEXT:     cond_br %30, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %31 = index_cast %27 : i32 to index
// CHECK-NEXT:     %32 = index_cast %29 : i32 to index
// CHECK-NEXT:     %33 = addi %29, %c3_i32 : i32
// CHECK-NEXT:     %34 = muli %27, %33 : i32
// CHECK-NEXT:     %35 = addi %34, %c1_i32 : i32
// CHECK-NEXT:     %36 = remi_signed %35, %arg3 : i32
// CHECK-NEXT:     %37 = sitofp %36 : i32 to f64
// CHECK-NEXT:     %38 = sitofp %arg3 : i32 to f64
// CHECK-NEXT:     %39 = divf %37, %38 : f64
// CHECK-NEXT:     store %39, %arg8[%31, %32] : memref<900x1200xf64>
// CHECK-NEXT:     %40 = addi %29, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%40 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %41 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb9(%41 : i32)
// CHECK-NEXT:   ^bb13(%42: i32):  // 2 preds: ^bb9, ^bb17
// CHECK-NEXT:     %43 = cmpi "slt", %42, %arg0 : i32
// CHECK-NEXT:     cond_br %43, ^bb15(%c0_i32 : i32), ^bb14
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb15(%44: i32):  // 2 preds: ^bb13, ^bb16
// CHECK-NEXT:     %45 = cmpi "slt", %44, %arg3 : i32
// CHECK-NEXT:     cond_br %45, ^bb16, ^bb17
// CHECK-NEXT:   ^bb16:  // pred: ^bb15
// CHECK-NEXT:     %46 = index_cast %42 : i32 to index
// CHECK-NEXT:     %47 = index_cast %44 : i32 to index
// CHECK-NEXT:     %48 = addi %44, %c2_i32 : i32
// CHECK-NEXT:     %49 = muli %42, %48 : i32
// CHECK-NEXT:     %50 = remi_signed %49, %arg2 : i32
// CHECK-NEXT:     %51 = sitofp %50 : i32 to f64
// CHECK-NEXT:     %52 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %53 = divf %51, %52 : f64
// CHECK-NEXT:     store %53, %arg9[%46, %47] : memref<800x1200xf64>
// CHECK-NEXT:     %54 = addi %44, %c1_i32 : i32
// CHECK-NEXT:     br ^bb15(%54 : i32)
// CHECK-NEXT:   ^bb17:  // pred: ^bb15
// CHECK-NEXT:     %55 = addi %42, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%55 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<800x900xf64>, %arg7: memref<800x1100xf64>, %arg8: memref<1100x900xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<800x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2(%c0_i32 : i32), ^bb8(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2(%2: i32):  // 2 preds: ^bb1, ^bb7
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb3, ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = index_cast %2 : i32 to index
// CHECK-NEXT:     store %cst, %arg6[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     br ^bb5(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %6 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%6 : i32)
// CHECK-NEXT:   ^bb5(%7: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %8 = cmpi "slt", %7, %arg2 : i32
// CHECK-NEXT:     cond_br %8, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %9 = index_cast %7 : i32 to index
// CHECK-NEXT:     %10 = load %arg7[%4, %9] : memref<800x1100xf64>
// CHECK-NEXT:     %11 = mulf %arg4, %10 : f64
// CHECK-NEXT:     %12 = load %arg8[%9, %5] : memref<1100x900xf64>
// CHECK-NEXT:     %13 = mulf %11, %12 : f64
// CHECK-NEXT:     %14 = load %arg6[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     %15 = addf %14, %13 : f64
// CHECK-NEXT:     store %15, %arg6[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     %16 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%16 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %17 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%17 : i32)
// CHECK-NEXT:   ^bb8(%18: i32):  // 2 preds: ^bb1, ^bb12
// CHECK-NEXT:     %19 = cmpi "slt", %18, %arg0 : i32
// CHECK-NEXT:     cond_br %19, ^bb10(%c0_i32 : i32), ^bb9
// CHECK-NEXT:   ^bb9:  // pred: ^bb8
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb10(%20: i32):  // 2 preds: ^bb8, ^bb15
// CHECK-NEXT:     %21 = cmpi "slt", %20, %arg3 : i32
// CHECK-NEXT:     cond_br %21, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %22 = index_cast %18 : i32 to index
// CHECK-NEXT:     %23 = index_cast %20 : i32 to index
// CHECK-NEXT:     %24 = load %arg10[%22, %23] : memref<800x1200xf64>
// CHECK-NEXT:     %25 = mulf %24, %arg5 : f64
// CHECK-NEXT:     store %25, %arg10[%22, %23] : memref<800x1200xf64>
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %26 = addi %18, %c1_i32 : i32
// CHECK-NEXT:     br ^bb8(%26 : i32)
// CHECK-NEXT:   ^bb13(%27: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg1 : i32
// CHECK-NEXT:     cond_br %28, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %29 = index_cast %27 : i32 to index
// CHECK-NEXT:     %30 = load %arg6[%22, %29] : memref<800x900xf64>
// CHECK-NEXT:     %31 = load %arg9[%29, %23] : memref<900x1200xf64>
// CHECK-NEXT:     %32 = mulf %30, %31 : f64
// CHECK-NEXT:     %33 = load %arg10[%22, %23] : memref<800x1200xf64>
// CHECK-NEXT:     %34 = addf %33, %32 : f64
// CHECK-NEXT:     store %34, %arg10[%22, %23] : memref<800x1200xf64>
// CHECK-NEXT:     %35 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%35 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %36 = addi %20, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%36 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<800x1200xf64>) {
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
// CHECK-NEXT:     %10 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
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
// CHECK-NEXT:     %19 = llvm.mlir.addressof @str2 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = llvm.call @fprintf(%16, %18, %20) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %22 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %23 = llvm.load %22 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.mlir.addressof @str6 : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %25 = llvm.getelementptr %24[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %26 = llvm.call @fprintf(%23, %25) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%27: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg1 : i32
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
// CHECK-NEXT:     %39 = load %arg2[%37, %38] : memref<800x1200xf64>
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