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
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2lf ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("G")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c800_i32 = constant 800 : i32
// CHECK-NEXT:     %c900_i32 = constant 900 : i32
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c1100_i32 = constant 1100 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c800_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c900_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<800x900xf64>>
// CHECK-NEXT:     %7 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %8 = muli %0, %7 : i32
// CHECK-NEXT:     %9 = zexti %8 : i32 to i64
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%9, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<800x1000xf64>>
// CHECK-NEXT:     %12 = muli %7, %1 : i32
// CHECK-NEXT:     %13 = zexti %12 : i32 to i64
// CHECK-NEXT:     %14 = call @polybench_alloc_data(%13, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?xi8> to memref<?xmemref<1000x900xf64>>
// CHECK-NEXT:     %16 = addi %c1100_i32, %c0_i32 : i32
// CHECK-NEXT:     %17 = muli %1, %16 : i32
// CHECK-NEXT:     %18 = zexti %17 : i32 to i64
// CHECK-NEXT:     %19 = call @polybench_alloc_data(%18, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?xi8> to memref<?xmemref<900x1100xf64>>
// CHECK-NEXT:     %21 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %22 = muli %1, %21 : i32
// CHECK-NEXT:     %23 = zexti %22 : i32 to i64
// CHECK-NEXT:     %24 = call @polybench_alloc_data(%23, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?xi8> to memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %26 = muli %21, %16 : i32
// CHECK-NEXT:     %27 = zexti %26 : i32 to i64
// CHECK-NEXT:     %28 = call @polybench_alloc_data(%27, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<?xi8> to memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %30 = muli %0, %16 : i32
// CHECK-NEXT:     %31 = zexti %30 : i32 to i64
// CHECK-NEXT:     %32 = call @polybench_alloc_data(%31, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?xi8> to memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:     %34 = load %11[%c0] : memref<?xmemref<800x1000xf64>>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<800x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %36 = memref_cast %35 : memref<?x1000xf64> to memref<800x1000xf64>
// CHECK-NEXT:     %37 = load %15[%c0] : memref<?xmemref<1000x900xf64>>
// CHECK-NEXT:     %38 = memref_cast %37 : memref<1000x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %39 = memref_cast %38 : memref<?x900xf64> to memref<1000x900xf64>
// CHECK-NEXT:     %40 = load %25[%c0] : memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %41 = memref_cast %40 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %42 = memref_cast %41 : memref<?x1200xf64> to memref<900x1200xf64>
// CHECK-NEXT:     %43 = load %29[%c0] : memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %44 = memref_cast %43 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %45 = memref_cast %44 : memref<?x1100xf64> to memref<1200x1100xf64>
// CHECK-NEXT:     call @init_array(%c800_i32, %c900_i32, %c1000_i32, %c1100_i32, %c1200_i32, %36, %39, %42, %45) : (i32, i32, i32, i32, i32, memref<800x1000xf64>, memref<1000x900xf64>, memref<900x1200xf64>, memref<1200x1100xf64>) -> ()
// CHECK-NEXT:     %46 = load %6[%c0] : memref<?xmemref<800x900xf64>>
// CHECK-NEXT:     %47 = memref_cast %46 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %48 = memref_cast %47 : memref<?x900xf64> to memref<800x900xf64>
// CHECK-NEXT:     %49 = load %11[%c0] : memref<?xmemref<800x1000xf64>>
// CHECK-NEXT:     %50 = memref_cast %49 : memref<800x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %51 = memref_cast %50 : memref<?x1000xf64> to memref<800x1000xf64>
// CHECK-NEXT:     %52 = load %15[%c0] : memref<?xmemref<1000x900xf64>>
// CHECK-NEXT:     %53 = memref_cast %52 : memref<1000x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %54 = memref_cast %53 : memref<?x900xf64> to memref<1000x900xf64>
// CHECK-NEXT:     %55 = load %20[%c0] : memref<?xmemref<900x1100xf64>>
// CHECK-NEXT:     %56 = memref_cast %55 : memref<900x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %57 = memref_cast %56 : memref<?x1100xf64> to memref<900x1100xf64>
// CHECK-NEXT:     %58 = load %25[%c0] : memref<?xmemref<900x1200xf64>>
// CHECK-NEXT:     %59 = memref_cast %58 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %60 = memref_cast %59 : memref<?x1200xf64> to memref<900x1200xf64>
// CHECK-NEXT:     %61 = load %29[%c0] : memref<?xmemref<1200x1100xf64>>
// CHECK-NEXT:     %62 = memref_cast %61 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %63 = memref_cast %62 : memref<?x1100xf64> to memref<1200x1100xf64>
// CHECK-NEXT:     %64 = load %33[%c0] : memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:     %65 = memref_cast %64 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %66 = memref_cast %65 : memref<?x1100xf64> to memref<800x1100xf64>
// CHECK-NEXT:     call @kernel_3mm(%c800_i32, %c900_i32, %c1000_i32, %c1100_i32, %c1200_i32, %48, %51, %54, %57, %60, %63, %66) : (i32, i32, i32, i32, i32, memref<800x900xf64>, memref<800x1000xf64>, memref<1000x900xf64>, memref<900x1100xf64>, memref<900x1200xf64>, memref<1200x1100xf64>, memref<800x1100xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %67 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %68 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %69 = addi %c0, %68 : index
// CHECK-NEXT:     %70 = load %arg1[%69] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %71 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %72 = call @strcmp(%70, %71) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %73 = trunci %72 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %74 = xor %73, %true : i1
// CHECK-NEXT:     %75 = and %67, %74 : i1
// CHECK-NEXT:     scf.if %75 {
// CHECK-NEXT:       %83 = load %33[%c0] : memref<?xmemref<800x1100xf64>>
// CHECK-NEXT:       %84 = memref_cast %83 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:       %85 = memref_cast %84 : memref<?x1100xf64> to memref<800x1100xf64>
// CHECK-NEXT:       call @print_array(%c800_i32, %c1100_i32, %85) : (i32, i32, memref<800x1100xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %76 = memref_cast %6 : memref<?xmemref<800x900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%76) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %77 = memref_cast %11 : memref<?xmemref<800x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%77) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %78 = memref_cast %15 : memref<?xmemref<1000x900xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%78) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %79 = memref_cast %20 : memref<?xmemref<900x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%79) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %80 = memref_cast %25 : memref<?xmemref<900x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%80) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %81 = memref_cast %29 : memref<?xmemref<1200x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%81) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %82 = memref_cast %33 : memref<?xmemref<800x1100xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%82) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x1000xf64>, %arg6: memref<1000x900xf64>, %arg7: memref<900x1200xf64>, %arg8: memref<1200x1100xf64>) {
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
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg2 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg5 : memref<800x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = muli %0, %2 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %10 = addi %9, %c1_i32 : i32
// CHECK-NEXT:     %11 = remi_signed %10, %arg0 : i32
// CHECK-NEXT:     %12 = sitofp %11 : i32 to f64
// CHECK-NEXT:     %c5_i32 = constant 5 : i32
// CHECK-NEXT:     %13 = muli %c5_i32, %arg0 : i32
// CHECK-NEXT:     %14 = sitofp %13 : i32 to f64
// CHECK-NEXT:     %15 = divf %12, %14 : f64
// CHECK-NEXT:     store %15, %6[%5, %8] : memref<?x1000xf64>
// CHECK-NEXT:     %16 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%16 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%17 : i32)
// CHECK-NEXT:   ^bb7(%18: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %19 = cmpi "slt", %18, %arg2 : i32
// CHECK-NEXT:     cond_br %19, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%20: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %21 = cmpi "slt", %20, %arg1 : i32
// CHECK-NEXT:     cond_br %21, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %22 = index_cast %18 : i32 to index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = memref_cast %arg6 : memref<1000x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %25 = index_cast %20 : i32 to index
// CHECK-NEXT:     %26 = addi %c0, %25 : index
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %27 = addi %20, %c1_i32_1 : i32
// CHECK-NEXT:     %28 = muli %18, %27 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %29 = addi %28, %c2_i32 : i32
// CHECK-NEXT:     %30 = remi_signed %29, %arg1 : i32
// CHECK-NEXT:     %31 = sitofp %30 : i32 to f64
// CHECK-NEXT:     %c5_i32_2 = constant 5 : i32
// CHECK-NEXT:     %32 = muli %c5_i32_2, %arg1 : i32
// CHECK-NEXT:     %33 = sitofp %32 : i32 to f64
// CHECK-NEXT:     %34 = divf %31, %33 : f64
// CHECK-NEXT:     store %34, %24[%23, %26] : memref<?x900xf64>
// CHECK-NEXT:     br ^bb10(%27 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %35 = addi %18, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%35 : i32)
// CHECK-NEXT:   ^bb13(%36: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %37 = cmpi "slt", %36, %arg1 : i32
// CHECK-NEXT:     cond_br %37, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%38: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %39 = cmpi "slt", %38, %arg4 : i32
// CHECK-NEXT:     cond_br %39, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %40 = index_cast %36 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %42 = memref_cast %arg7 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %43 = index_cast %38 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %45 = addi %38, %c3_i32 : i32
// CHECK-NEXT:     %46 = muli %36, %45 : i32
// CHECK-NEXT:     %47 = remi_signed %46, %arg3 : i32
// CHECK-NEXT:     %48 = sitofp %47 : i32 to f64
// CHECK-NEXT:     %c5_i32_4 = constant 5 : i32
// CHECK-NEXT:     %49 = muli %c5_i32_4, %arg3 : i32
// CHECK-NEXT:     %50 = sitofp %49 : i32 to f64
// CHECK-NEXT:     %51 = divf %48, %50 : f64
// CHECK-NEXT:     store %51, %42[%41, %44] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %52 = addi %38, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb16(%52 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %53 = addi %36, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb13(%53 : i32)
// CHECK-NEXT:   ^bb19(%54: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %55 = cmpi "slt", %54, %arg4 : i32
// CHECK-NEXT:     cond_br %55, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb22(%56: i32):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %57 = cmpi "slt", %56, %arg3 : i32
// CHECK-NEXT:     cond_br %57, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %58 = index_cast %54 : i32 to index
// CHECK-NEXT:     %59 = addi %c0, %58 : index
// CHECK-NEXT:     %60 = memref_cast %arg8 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %61 = index_cast %56 : i32 to index
// CHECK-NEXT:     %62 = addi %c0, %61 : index
// CHECK-NEXT:     %c2_i32_7 = constant 2 : i32
// CHECK-NEXT:     %63 = addi %56, %c2_i32_7 : i32
// CHECK-NEXT:     %64 = muli %54, %63 : i32
// CHECK-NEXT:     %65 = addi %64, %c2_i32_7 : i32
// CHECK-NEXT:     %66 = remi_signed %65, %arg2 : i32
// CHECK-NEXT:     %67 = sitofp %66 : i32 to f64
// CHECK-NEXT:     %c5_i32_8 = constant 5 : i32
// CHECK-NEXT:     %68 = muli %c5_i32_8, %arg2 : i32
// CHECK-NEXT:     %69 = sitofp %68 : i32 to f64
// CHECK-NEXT:     %70 = divf %67, %69 : f64
// CHECK-NEXT:     store %70, %60[%59, %62] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32_9 = constant 1 : i32
// CHECK-NEXT:     %71 = addi %56, %c1_i32_9 : i32
// CHECK-NEXT:     br ^bb22(%71 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %c1_i32_10 = constant 1 : i32
// CHECK-NEXT:     %72 = addi %54, %c1_i32_10 : i32
// CHECK-NEXT:     br ^bb19(%72 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) {
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
// CHECK-NEXT:     %6 = memref_cast %arg5 : memref<800x900xf64> to memref<?x900xf64>
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
// CHECK-NEXT:     %12 = memref_cast %arg6 : memref<800x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %13 = index_cast %10 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = load %12[%5, %14] : memref<?x1000xf64>
// CHECK-NEXT:     %16 = memref_cast %arg7 : memref<1000x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %17 = load %16[%14, %8] : memref<?x900xf64>
// CHECK-NEXT:     %18 = mulf %15, %17 : f64
// CHECK-NEXT:     %19 = load %6[%5, %8] : memref<?x900xf64>
// CHECK-NEXT:     %20 = addf %19, %18 : f64
// CHECK-NEXT:     store %20, %6[%5, %8] : memref<?x900xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %21 = addi %10, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%21 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%22 : i32)
// CHECK-NEXT:   ^bb10(%23: i32):  // 2 preds: ^bb3, ^bb15
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg1 : i32
// CHECK-NEXT:     cond_br %24, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb13(%25: i32):  // 2 preds: ^bb11, ^bb18
// CHECK-NEXT:     %26 = cmpi "slt", %25, %arg3 : i32
// CHECK-NEXT:     cond_br %26, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %27 = index_cast %23 : i32 to index
// CHECK-NEXT:     %28 = addi %c0, %27 : index
// CHECK-NEXT:     %29 = memref_cast %arg8 : memref<900x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %30 = index_cast %25 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %cst_2 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_2, %29[%28, %31] : memref<?x1100xf64>
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %23, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb10(%32 : i32)
// CHECK-NEXT:   ^bb16(%33: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %34 = cmpi "slt", %33, %arg4 : i32
// CHECK-NEXT:     cond_br %34, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %35 = memref_cast %arg9 : memref<900x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %36 = index_cast %33 : i32 to index
// CHECK-NEXT:     %37 = addi %c0, %36 : index
// CHECK-NEXT:     %38 = load %35[%28, %37] : memref<?x1200xf64>
// CHECK-NEXT:     %39 = memref_cast %arg10 : memref<1200x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %40 = load %39[%37, %31] : memref<?x1100xf64>
// CHECK-NEXT:     %41 = mulf %38, %40 : f64
// CHECK-NEXT:     %42 = load %29[%28, %31] : memref<?x1100xf64>
// CHECK-NEXT:     %43 = addf %42, %41 : f64
// CHECK-NEXT:     store %43, %29[%28, %31] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32_4 = constant 1 : i32
// CHECK-NEXT:     %44 = addi %33, %c1_i32_4 : i32
// CHECK-NEXT:     br ^bb16(%44 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %45 = addi %25, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb13(%45 : i32)
// CHECK-NEXT:   ^bb19(%46: i32):  // 2 preds: ^bb12, ^bb24
// CHECK-NEXT:     %47 = cmpi "slt", %46, %arg0 : i32
// CHECK-NEXT:     cond_br %47, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb22(%48: i32):  // 2 preds: ^bb20, ^bb27
// CHECK-NEXT:     %49 = cmpi "slt", %48, %arg3 : i32
// CHECK-NEXT:     cond_br %49, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %50 = index_cast %46 : i32 to index
// CHECK-NEXT:     %51 = addi %c0, %50 : index
// CHECK-NEXT:     %52 = memref_cast %arg11 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %53 = index_cast %48 : i32 to index
// CHECK-NEXT:     %54 = addi %c0, %53 : index
// CHECK-NEXT:     %cst_6 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_6, %52[%51, %54] : memref<?x1100xf64>
// CHECK-NEXT:     br ^bb25(%c0_i32 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %c1_i32_7 = constant 1 : i32
// CHECK-NEXT:     %55 = addi %46, %c1_i32_7 : i32
// CHECK-NEXT:     br ^bb19(%55 : i32)
// CHECK-NEXT:   ^bb25(%56: i32):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:     %57 = cmpi "slt", %56, %arg1 : i32
// CHECK-NEXT:     cond_br %57, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %58 = memref_cast %arg5 : memref<800x900xf64> to memref<?x900xf64>
// CHECK-NEXT:     %59 = index_cast %56 : i32 to index
// CHECK-NEXT:     %60 = addi %c0, %59 : index
// CHECK-NEXT:     %61 = load %58[%51, %60] : memref<?x900xf64>
// CHECK-NEXT:     %62 = memref_cast %arg8 : memref<900x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %63 = load %62[%60, %54] : memref<?x1100xf64>
// CHECK-NEXT:     %64 = mulf %61, %63 : f64
// CHECK-NEXT:     %65 = load %52[%51, %54] : memref<?x1100xf64>
// CHECK-NEXT:     %66 = addf %65, %64 : f64
// CHECK-NEXT:     store %66, %52[%51, %54] : memref<?x1100xf64>
// CHECK-NEXT:     %c1_i32_8 = constant 1 : i32
// CHECK-NEXT:     %67 = addi %56, %c1_i32_8 : i32
// CHECK-NEXT:     br ^bb25(%67 : i32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %c1_i32_9 = constant 1 : i32
// CHECK-NEXT:     %68 = addi %48, %c1_i32_9 : i32
// CHECK-NEXT:     br ^bb22(%68 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<800x1100xf64>) {
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
// CHECK-NEXT:     %34 = memref_cast %arg2 : memref<800x1100xf64> to memref<?x1100xf64>
// CHECK-NEXT:     %35 = index_cast %23 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %34[%33, %36] : memref<?x1100xf64>
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