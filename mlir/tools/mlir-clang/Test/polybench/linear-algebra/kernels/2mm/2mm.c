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
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("D")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c800_i32 = constant 800 : i32
// CHECK-NEXT:     %c900_i32 = constant 900 : i32
// CHECK-NEXT:     %c1100_i32 = constant 1100 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloca() : memref<1xf64>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %2 = addi %c800_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = addi %c900_i32, %c0_i32 : i32
// CHECK-NEXT:     %4 = muli %2, %3 : i32
// CHECK-NEXT:     %5 = zexti %4 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %6 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%5, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<800x900xf64>>
// CHECK-NEXT:     %9 = addi %c1100_i32, %c0_i32 : i32
// CHECK-NEXT:     %10 = muli %2, %9 : i32
// CHECK-NEXT:     %11 = zexti %10 : i32 to i64
// CHECK-NEXT:     %12 = call @polybench_alloc_data(%11, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<?xi8> to memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:     %14 = muli %9, %3 : i32
// CHECK-NEXT:     %15 = zexti %14 : i32 to i64
// CHECK-NEXT:     %16 = call @polybench_alloc_data(%15, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?xi8> to memref<?xmemref<1100x900xf64>>
// CHECK-NEXT:     %18 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %19 = muli %3, %18 : i32
// CHECK-NEXT:     %20 = zexti %19 : i32 to i64
// CHECK-NEXT:     %21 = call @polybench_alloc_data(%20, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?xi8> to memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %23 = muli %2, %18 : i32
// CHECK-NEXT:     %24 = zexti %23 : i32 to i64
// CHECK-NEXT:     %25 = call @polybench_alloc_data(%24, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<?xi8> to memref<?xmemref<800x1200xf64>>
// CHECK-NEXT:     %27 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %28 = memref_cast %1 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     %29 = load %13[%c0] : memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<?x1100xf64> to memref<800x1100xf64>
// CHECK-NEXT:     %32 = load %17[%c0] : memref<?xmemref<1100x900xf64>>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<1100x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %34 = memref_cast %33 : memref<?x900xf64> to memref<1100x900xf64>
// CHECK-NEXT:     %35 = load %22[%c0] : memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %37 = memref_cast %36 : memref<?x1200xf64> to memref<900x1200xf64>
// CHECK-NEXT:     %38 = load %26[%c0] : memref<?xmemref<800x1200xf64>>
// CHECK-NEXT:     %39 = memref_cast %38 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %40 = memref_cast %39 : memref<?x1200xf64> to memref<800x1200xf64>
// CHECK-NEXT:     call @init_array(%c800_i32, %c900_i32, %c1100_i32, %c1200_i32, %27, %28, %31, %34, %37, %40) : (i32, i32, i32, i32, memref<?xf64>, memref<?xf64>, memref<800x1100xf64>, memref<1100x900xf64>, memref<900x1200xf64>, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     %41 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %42 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %43 = load %8[%c0] : memref<?xmemref<800x900xf64>>
// CHECK-NEXT:     %44 = memref_cast %43 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %45 = memref_cast %44 : memref<?x900xf64> to memref<800x900xf64>
// CHECK-NEXT:     %46 = load %13[%c0] : memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:     %47 = memref_cast %46 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %48 = memref_cast %47 : memref<?x1100xf64> to memref<800x1100xf64>
// CHECK-NEXT:     %49 = load %17[%c0] : memref<?xmemref<1100x900xf64>>
// CHECK-NEXT:     %50 = memref_cast %49 : memref<1100x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %51 = memref_cast %50 : memref<?x900xf64> to memref<1100x900xf64>
// CHECK-NEXT:     %52 = load %22[%c0] : memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %53 = memref_cast %52 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %54 = memref_cast %53 : memref<?x1200xf64> to memref<900x1200xf64>
// CHECK-NEXT:     %55 = load %26[%c0] : memref<?xmemref<800x1200xf64>>
// CHECK-NEXT:     %56 = memref_cast %55 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %57 = memref_cast %56 : memref<?x1200xf64> to memref<800x1200xf64>
// CHECK-NEXT:     call @kernel_2mm(%c800_i32, %c900_i32, %c1100_i32, %c1200_i32, %41, %42, %45, %48, %51, %54, %57) : (i32, i32, i32, i32, f64, f64, memref<800x900xf64>, memref<800x1100xf64>, memref<1100x900xf64>, memref<900x1200xf64>, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %58 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %59 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %60 = addi %c0, %59 : index
// CHECK-NEXT:     %61 = load %arg1[%60] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %62 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %63 = call @strcmp(%61, %62) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %64 = trunci %63 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %65 = xor %64, %true : i1
// CHECK-NEXT:     %66 = and %58, %65 : i1
// CHECK-NEXT:     scf.if %66 {
// CHECK-NEXT:       %72 = load %26[%c0] : memref<?xmemref<800x1200xf64>>
// CHECK-NEXT:       %73 = memref_cast %72 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %74 = memref_cast %73 : memref<?x1200xf64> to memref<800x1200xf64>
// CHECK-NEXT:       call @print_array(%c800_i32, %c1200_i32, %74) : (i32, i32, memref<800x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %67 = memref_cast %8 : memref<?xmemref<800x900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%67) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %68 = memref_cast %13 : memref<?xmemref<800x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%68) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %69 = memref_cast %17 : memref<?xmemref<1100x900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%69) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %70 = memref_cast %22 : memref<?xmemref<900x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%70) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %71 = memref_cast %26 : memref<?xmemref<800x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%71) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<800x1100xf64>, %arg7: memref<1100x900xf64>, %arg8: memref<900x1200xf64>, %arg9: memref<800x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.500000e+00 : f64
// CHECK-NEXT:     store %cst, %arg4[%c0] : memref<?xf64>
// CHECK-NEXT:     %cst_0 = constant 1.200000e+00 : f64
// CHECK-NEXT:     store %cst_0, %arg5[%c0] : memref<?xf64>
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg2 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg6 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = muli %0, %2 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %10 = addi %9, %c1_i32 : i32
// CHECK-NEXT:     %11 = remi_signed %10, %arg0 : i32
// CHECK-NEXT:     %12 = sitofp %11 : i32 to f64
// CHECK-NEXT:     %13 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %14 = divf %12, %13 : f64
// CHECK-NEXT:     store %14, %6[%5, %8] : memref<?x1100xf64>
// CHECK-NEXT:     %15 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%15 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %0, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%16 : i32)
// CHECK-NEXT:   ^bb7(%17: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %18 = cmpi "slt", %17, %arg2 : i32
// CHECK-NEXT:     cond_br %18, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%19: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %20 = cmpi "slt", %19, %arg1 : i32
// CHECK-NEXT:     cond_br %20, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %21 = index_cast %17 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = memref_cast %arg7 : memref<1100x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %24 = index_cast %19 : i32 to index
// CHECK-NEXT:     %25 = addi %c0, %24 : index
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %19, %c1_i32_2 : i32
// CHECK-NEXT:     %27 = muli %17, %26 : i32
// CHECK-NEXT:     %28 = remi_signed %27, %arg1 : i32
// CHECK-NEXT:     %29 = sitofp %28 : i32 to f64
// CHECK-NEXT:     %30 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %31 = divf %29, %30 : f64
// CHECK-NEXT:     store %31, %23[%22, %25] : memref<?x900xf64>
// CHECK-NEXT:     br ^bb10(%26 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %17, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%32 : i32)
// CHECK-NEXT:   ^bb13(%33: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %34 = cmpi "slt", %33, %arg1 : i32
// CHECK-NEXT:     cond_br %34, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%35: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %36 = cmpi "slt", %35, %arg3 : i32
// CHECK-NEXT:     cond_br %36, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %37 = index_cast %33 : i32 to index
// CHECK-NEXT:     %38 = addi %c0, %37 : index
// CHECK-NEXT:     %39 = memref_cast %arg8 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %40 = index_cast %35 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %42 = addi %35, %c3_i32 : i32
// CHECK-NEXT:     %43 = muli %33, %42 : i32
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %44 = addi %43, %c1_i32_4 : i32
// CHECK-NEXT:     %45 = remi_signed %44, %arg3 : i32
// CHECK-NEXT:     %46 = sitofp %45 : i32 to f64
// CHECK-NEXT:     %47 = sitofp %arg3 : i32 to f64
// CHECK-NEXT:     %48 = divf %46, %47 : f64
// CHECK-NEXT:     store %48, %39[%38, %41] : memref<?x1200xf64>
// CHECK-NEXT:     %49 = addi %35, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb16(%49 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %50 = addi %33, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb13(%50 : i32)
// CHECK-NEXT:   ^bb19(%51: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %52 = cmpi "slt", %51, %arg0 : i32
// CHECK-NEXT:     cond_br %52, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb22(%53: i32):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %54 = cmpi "slt", %53, %arg3 : i32
// CHECK-NEXT:     cond_br %54, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %55 = index_cast %51 : i32 to index
// CHECK-NEXT:     %56 = addi %c0, %55 : index
// CHECK-NEXT:     %57 = memref_cast %arg9 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %58 = index_cast %53 : i32 to index
// CHECK-NEXT:     %59 = addi %c0, %58 : index
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %60 = addi %53, %c2_i32 : i32
// CHECK-NEXT:     %61 = muli %51, %60 : i32
// CHECK-NEXT:     %62 = remi_signed %61, %arg2 : i32
// CHECK-NEXT:     %63 = sitofp %62 : i32 to f64
// CHECK-NEXT:     %64 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %65 = divf %63, %64 : f64
// CHECK-NEXT:     store %65, %57[%56, %59] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %66 = addi %53, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb22(%66 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %67 = addi %51, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb19(%67 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<800x900xf64>, %arg7: memref<800x1100xf64>, %arg8: memref<1100x900xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<800x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg6 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %6[%5, %8] : memref<?x900xf64>
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %9 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%9 : i32)
// CHECK-NEXT:   ^bb7(%10: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %11 = cmpi "slt", %10, %arg2 : i32
// CHECK-NEXT:     cond_br %11, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %12 = memref_cast %arg7 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %13 = index_cast %10 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = load %12[%5, %14] : memref<?x1100xf64>
// CHECK-NEXT:     %16 = mulf %arg4, %15 : f64
// CHECK-NEXT:     %17 = memref_cast %arg8 : memref<1100x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %18 = load %17[%14, %8] : memref<?x900xf64>
// CHECK-NEXT:     %19 = mulf %16, %18 : f64
// CHECK-NEXT:     %20 = load %6[%5, %8] : memref<?x900xf64>
// CHECK-NEXT:     %21 = addf %20, %19 : f64
// CHECK-NEXT:     store %21, %6[%5, %8] : memref<?x900xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %10, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%22 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %23 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%23 : i32)
// CHECK-NEXT:   ^bb10(%24: i32):  // 2 preds: ^bb3, ^bb15
// CHECK-NEXT:     %25 = cmpi "slt", %24, %arg0 : i32
// CHECK-NEXT:     cond_br %25, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb13(%26: i32):  // 2 preds: ^bb11, ^bb18
// CHECK-NEXT:     %27 = cmpi "slt", %26, %arg3 : i32
// CHECK-NEXT:     cond_br %27, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %28 = index_cast %24 : i32 to index
// CHECK-NEXT:     %29 = addi %c0, %28 : index
// CHECK-NEXT:     %30 = memref_cast %arg10 : memref<800x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %31 = index_cast %26 : i32 to index
// CHECK-NEXT:     %32 = addi %c0, %31 : index
// CHECK-NEXT:     %33 = load %30[%29, %32] : memref<?x1200xf64>
// CHECK-NEXT:     %34 = mulf %33, %arg5 : f64
// CHECK-NEXT:     store %34, %30[%29, %32] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %35 = addi %24, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%35 : i32)
// CHECK-NEXT:   ^bb16(%36: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %37 = cmpi "slt", %36, %arg1 : i32
// CHECK-NEXT:     cond_br %37, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %38 = memref_cast %arg6 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %39 = index_cast %36 : i32 to index
// CHECK-NEXT:     %40 = addi %c0, %39 : index
// CHECK-NEXT:     %41 = load %38[%29, %40] : memref<?x900xf64>
// CHECK-NEXT:     %42 = memref_cast %arg9 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %43 = load %42[%40, %32] : memref<?x1200xf64>
// CHECK-NEXT:     %44 = mulf %41, %43 : f64
// CHECK-NEXT:     %45 = load %30[%29, %32] : memref<?x1200xf64>
// CHECK-NEXT:     %46 = addf %45, %44 : f64
// CHECK-NEXT:     store %46, %30[%29, %32] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %47 = addi %36, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb16(%47 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %48 = addi %26, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb13(%48 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<800x1200xf64>) {
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
// CHECK-NEXT:   ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg0 : i32
// CHECK-NEXT:     cond_br %12, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %13 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %14 = llvm.mlir.addressof @str5 : !llvm.ptr<array<16 x i8>>
// CHECK-NEXT:     %15 = llvm.getelementptr %14[%2, %2] : (!llvm.ptr<array<16 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str2 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %18 = llvm.call @fprintf(%13, %15, %17) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %19 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %20 = llvm.mlir.addressof @str6 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %21 = llvm.getelementptr %20[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %22 = llvm.call @fprintf(%19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%23: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg1 : i32
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
// CHECK-NEXT:     %34 = memref_cast %arg2 : memref<800x1200xf64> to memref<?x1200xf64>
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