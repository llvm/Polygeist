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
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* E := A*B */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	E[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }
  /* F := C*D */
  for (i = 0; i < _PB_NJ; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	F[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NM; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }
  /* G := E*F */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	G[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NJ; ++k)
	  G[i][j] += E[i][k] * F[k][j];
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
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("G\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-NEXT:     %c800_i32 = constant 800 : i32
// CHECK-NEXT:     %c900_i32 = constant 900 : i32
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c1100_i32 = constant 1100 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %0 = alloc() : memref<800x900xf64>
// CHECK-NEXT:     %1 = alloc() : memref<800x1000xf64>
// CHECK-NEXT:     %2 = alloc() : memref<1000x900xf64>
// CHECK-NEXT:     %3 = alloc() : memref<900x1100xf64>
// CHECK-NEXT:     %4 = alloc() : memref<900x1200xf64>
// CHECK-NEXT:     %5 = alloc() : memref<1200x1100xf64>
// CHECK-NEXT:     %6 = alloc() : memref<800x1100xf64>
// CHECK-NEXT:     call @init_array(%c800_i32, %c900_i32, %c1000_i32, %c1100_i32, %c1200_i32, %1, %2, %4, %5) : (i32, i32, i32, i32, i32, memref<800x1000xf64>, memref<1000x900xf64>, memref<900x1200xf64>, memref<1200x1100xf64>) -> ()
// CHECK-NEXT:     call @kernel_3mm(%c800_i32, %c900_i32, %c1000_i32, %c1100_i32, %c1200_i32, %0, %1, %2, %3, %4, %5, %6) : (i32, i32, i32, i32, i32, memref<800x900xf64>, memref<800x1000xf64>, memref<1000x900xf64>, memref<900x1100xf64>, memref<900x1200xf64>, memref<1200x1100xf64>, memref<800x1100xf64>) -> ()
// CHECK-NEXT:     %7 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %8 = trunci %c0_i32 : i32 to i1
// CHECK-NEXT:     %9 = xor %8, %true : i1
// CHECK-NEXT:     %10 = and %7, %9 : i1
// CHECK-NEXT:     scf.if %10 {
// CHECK-NEXT:       call @print_array(%c800_i32, %c1100_i32, %6) : (i32, i32, memref<800x1100xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x1000xf64>, %arg6: memref<1000x900xf64>, %arg7: memref<900x1200xf64>, %arg8: memref<1200x1100xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %c5_i32 = constant 5 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
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
// CHECK-NEXT:     %10 = muli %arg0, %c5_i32 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = divf %9, %11 : f64
// CHECK-NEXT:     store %12, %arg5[%4, %5] : memref<800x1000xf64>
// CHECK-NEXT:     %13 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%13 : i32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %14 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   ^bb5(%15: i32):  // 2 preds: ^bb1, ^bb8
// CHECK-NEXT:     %16 = cmpi "slt", %15, %arg2 : i32
// CHECK-NEXT:     cond_br %16, ^bb6(%c0_i32 : i32), ^bb9(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6(%17: i32):  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg1 : i32
// CHECK-NEXT:     cond_br %18, ^bb7, ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %19 = index_cast %15 : i32 to index
// CHECK-NEXT:     %20 = index_cast %17 : i32 to index
// CHECK-NEXT:     %21 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     %22 = muli %15, %21 : i32
// CHECK-NEXT:     %23 = addi %22, %c2_i32 : i32
// CHECK-NEXT:     %24 = remi_signed %23, %arg1 : i32
// CHECK-NEXT:     %25 = sitofp %24 : i32 to f64
// CHECK-NEXT:     %26 = muli %arg1, %c5_i32 : i32
// CHECK-NEXT:     %27 = sitofp %26 : i32 to f64
// CHECK-NEXT:     %28 = divf %25, %27 : f64
// CHECK-NEXT:     store %28, %arg6[%19, %20] : memref<1000x900xf64>
// CHECK-NEXT:     br ^bb6(%21 : i32)
// CHECK-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-NEXT:     %29 = addi %15, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%29 : i32)
// CHECK-NEXT:   ^bb9(%30: i32):  // 2 preds: ^bb5, ^bb12
// CHECK-NEXT:     %31 = cmpi "slt", %30, %arg1 : i32
// CHECK-NEXT:     cond_br %31, ^bb10(%c0_i32 : i32), ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%32: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %33 = cmpi "slt", %32, %arg4 : i32
// CHECK-NEXT:     cond_br %33, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %34 = index_cast %30 : i32 to index
// CHECK-NEXT:     %35 = index_cast %32 : i32 to index
// CHECK-NEXT:     %36 = addi %32, %c3_i32 : i32
// CHECK-NEXT:     %37 = muli %30, %36 : i32
// CHECK-NEXT:     %38 = remi_signed %37, %arg3 : i32
// CHECK-NEXT:     %39 = sitofp %38 : i32 to f64
// CHECK-NEXT:     %40 = muli %arg3, %c5_i32 : i32
// CHECK-NEXT:     %41 = sitofp %40 : i32 to f64
// CHECK-NEXT:     %42 = divf %39, %41 : f64
// CHECK-NEXT:     store %42, %arg7[%34, %35] : memref<900x1200xf64>
// CHECK-NEXT:     %43 = addi %32, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%43 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %44 = addi %30, %c1_i32 : i32
// CHECK-NEXT:     br ^bb9(%44 : i32)
// CHECK-NEXT:   ^bb13(%45: i32):  // 2 preds: ^bb9, ^bb17
// CHECK-NEXT:     %46 = cmpi "slt", %45, %arg4 : i32
// CHECK-NEXT:     cond_br %46, ^bb15(%c0_i32 : i32), ^bb14
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb15(%47: i32):  // 2 preds: ^bb13, ^bb16
// CHECK-NEXT:     %48 = cmpi "slt", %47, %arg3 : i32
// CHECK-NEXT:     cond_br %48, ^bb16, ^bb17
// CHECK-NEXT:   ^bb16:  // pred: ^bb15
// CHECK-NEXT:     %49 = index_cast %45 : i32 to index
// CHECK-NEXT:     %50 = index_cast %47 : i32 to index
// CHECK-NEXT:     %51 = addi %47, %c2_i32 : i32
// CHECK-NEXT:     %52 = muli %45, %51 : i32
// CHECK-NEXT:     %53 = addi %52, %c2_i32 : i32
// CHECK-NEXT:     %54 = remi_signed %53, %arg2 : i32
// CHECK-NEXT:     %55 = sitofp %54 : i32 to f64
// CHECK-NEXT:     %56 = muli %arg2, %c5_i32 : i32
// CHECK-NEXT:     %57 = sitofp %56 : i32 to f64
// CHECK-NEXT:     %58 = divf %55, %57 : f64
// CHECK-NEXT:     store %58, %arg8[%49, %50] : memref<1200x1100xf64>
// CHECK-NEXT:     %59 = addi %47, %c1_i32 : i32
// CHECK-NEXT:     br ^bb15(%59 : i32)
// CHECK-NEXT:   ^bb17:  // pred: ^bb15
// CHECK-NEXT:     %60 = addi %45, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%60 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) {
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
// CHECK-NEXT:     store %cst, %arg5[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     br ^bb5(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %6 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%6 : i32)
// CHECK-NEXT:   ^bb5(%7: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %8 = cmpi "slt", %7, %arg2 : i32
// CHECK-NEXT:     cond_br %8, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %9 = index_cast %7 : i32 to index
// CHECK-NEXT:     %10 = load %arg6[%4, %9] : memref<800x1000xf64>
// CHECK-NEXT:     %11 = load %arg7[%9, %5] : memref<1000x900xf64>
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = load %arg5[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     %14 = addf %13, %12 : f64
// CHECK-NEXT:     store %14, %arg5[%4, %5] : memref<800x900xf64>
// CHECK-NEXT:     %15 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%15 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %16 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%16 : i32)
// CHECK-NEXT:   ^bb8(%17: i32):  // 2 preds: ^bb1, ^bb11
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg1 : i32
// CHECK-NEXT:     cond_br %18, ^bb9(%c0_i32 : i32), ^bb15(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9(%19: i32):  // 2 preds: ^bb8, ^bb14
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg3 : i32
// CHECK-NEXT:     cond_br %20, ^bb10, ^bb11
// CHECK-NEXT:   ^bb10:  // pred: ^bb9
// CHECK-NEXT:     %21 = index_cast %17 : i32 to index
// CHECK-NEXT:     %22 = index_cast %19 : i32 to index
// CHECK-NEXT:     store %cst, %arg8[%21, %22] : memref<900x1100xf64>
// CHECK-NEXT:     br ^bb12(%c0_i32 : i32)
// CHECK-NEXT:   ^bb11:  // pred: ^bb9
// CHECK-NEXT:     %23 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     br ^bb8(%23 : i32)
// CHECK-NEXT:   ^bb12(%24: i32):  // 2 preds: ^bb10, ^bb13
// CHECK-NEXT:     %25 = cmpi "slt", %24, %arg4 : i32
// CHECK-NEXT:     cond_br %25, ^bb13, ^bb14
// CHECK-NEXT:   ^bb13:  // pred: ^bb12
// CHECK-NEXT:     %26 = index_cast %24 : i32 to index
// CHECK-NEXT:     %27 = load %arg9[%21, %26] : memref<900x1200xf64>
// CHECK-NEXT:     %28 = load %arg10[%26, %22] : memref<1200x1100xf64>
// CHECK-NEXT:     %29 = mulf %27, %28 : f64
// CHECK-NEXT:     %30 = load %arg8[%21, %22] : memref<900x1100xf64>
// CHECK-NEXT:     %31 = addf %30, %29 : f64
// CHECK-NEXT:     store %31, %arg8[%21, %22] : memref<900x1100xf64>
// CHECK-NEXT:     %32 = addi %24, %c1_i32 : i32
// CHECK-NEXT:     br ^bb12(%32 : i32)
// CHECK-NEXT:   ^bb14:  // pred: ^bb12
// CHECK-NEXT:     %33 = addi %19, %c1_i32 : i32
// CHECK-NEXT:     br ^bb9(%33 : i32)
// CHECK-NEXT:   ^bb15(%34: i32):  // 2 preds: ^bb8, ^bb19
// CHECK-NEXT:     %35 = cmpi "slt", %34, %arg0 : i32
// CHECK-NEXT:     cond_br %35, ^bb17(%c0_i32 : i32), ^bb16
// CHECK-NEXT:   ^bb16:  // pred: ^bb15
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb17(%36: i32):  // 2 preds: ^bb15, ^bb22
// CHECK-NEXT:     %37 = cmpi "slt", %36, %arg3 : i32
// CHECK-NEXT:     cond_br %37, ^bb18, ^bb19
// CHECK-NEXT:   ^bb18:  // pred: ^bb17
// CHECK-NEXT:     %38 = index_cast %34 : i32 to index
// CHECK-NEXT:     %39 = index_cast %36 : i32 to index
// CHECK-NEXT:     store %cst, %arg11[%38, %39] : memref<800x1100xf64>
// CHECK-NEXT:     br ^bb20(%c0_i32 : i32)
// CHECK-NEXT:   ^bb19:  // pred: ^bb17
// CHECK-NEXT:     %40 = addi %34, %c1_i32 : i32
// CHECK-NEXT:     br ^bb15(%40 : i32)
// CHECK-NEXT:   ^bb20(%41: i32):  // 2 preds: ^bb18, ^bb21
// CHECK-NEXT:     %42 = cmpi "slt", %41, %arg1 : i32
// CHECK-NEXT:     cond_br %42, ^bb21, ^bb22
// CHECK-NEXT:   ^bb21:  // pred: ^bb20
// CHECK-NEXT:     %43 = index_cast %41 : i32 to index
// CHECK-NEXT:     %44 = load %arg5[%38, %43] : memref<800x900xf64>
// CHECK-NEXT:     %45 = load %arg8[%43, %39] : memref<900x1100xf64>
// CHECK-NEXT:     %46 = mulf %44, %45 : f64
// CHECK-NEXT:     %47 = load %arg11[%38, %39] : memref<800x1100xf64>
// CHECK-NEXT:     %48 = addf %47, %46 : f64
// CHECK-NEXT:     store %48, %arg11[%38, %39] : memref<800x1100xf64>
// CHECK-NEXT:     %49 = addi %41, %c1_i32 : i32
// CHECK-NEXT:     br ^bb20(%49 : i32)
// CHECK-NEXT:   ^bb22:  // pred: ^bb20
// CHECK-NEXT:     %50 = addi %36, %c1_i32 : i32
// CHECK-NEXT:     br ^bb17(%50 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<800x1100xf64>) {
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
// CHECK-NEXT:     %39 = load %arg2[%37, %38] : memref<800x1100xf64>
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