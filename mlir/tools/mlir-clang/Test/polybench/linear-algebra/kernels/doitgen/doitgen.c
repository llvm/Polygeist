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
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++)  {
      for (p = 0; p < _PB_NP; p++)  {
	sum[p] = SCALAR_VAL(0.0);
	for (s = 0; s < _PB_NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c150_i32 = constant 150 : i32
// CHECK-NEXT:     %c140_i32 = constant 140 : i32
// CHECK-NEXT:     %c160_i32 = constant 160 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c150_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c140_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = addi %c160_i32, %c0_i32 : i32
// CHECK-NEXT:     %4 = muli %2, %3 : i32
// CHECK-NEXT:     %5 = zexti %4 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %6 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%5, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<150x140x160xf64>>
// CHECK-NEXT:     %9 = zexti %3 : i32 to i64
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%9, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<160xf64>>
// CHECK-NEXT:     %12 = muli %3, %3 : i32
// CHECK-NEXT:     %13 = zexti %12 : i32 to i64
// CHECK-NEXT:     %14 = call @polybench_alloc_data(%13, %6) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?xi8> to memref<?xmemref<160x160xf64>>
// CHECK-NEXT:     %16 = load %8[%c0] : memref<?xmemref<150x140x160xf64>>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?x140x160xf64> to memref<150x140x160xf64>
// CHECK-NEXT:     %19 = load %15[%c0] : memref<?xmemref<160x160xf64>>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<160x160xf64> to memref<?x160xf64>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<?x160xf64> to memref<160x160xf64>
// CHECK-NEXT:     call @init_array(%c150_i32, %c140_i32, %c160_i32, %18, %21) : (i32, i32, i32, memref<150x140x160xf64>, memref<160x160xf64>) -> ()
// CHECK-NEXT:     %22 = load %8[%c0] : memref<?xmemref<150x140x160xf64>>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?x140x160xf64> to memref<150x140x160xf64>
// CHECK-NEXT:     %25 = load %15[%c0] : memref<?xmemref<160x160xf64>>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<160x160xf64> to memref<?x160xf64>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<?x160xf64> to memref<160x160xf64>
// CHECK-NEXT:     %28 = load %11[%c0] : memref<?xmemref<160xf64>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<160xf64> to memref<?xf64>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?xf64> to memref<160xf64>
// CHECK-NEXT:     call @kernel_doitgen(%c150_i32, %c140_i32, %c160_i32, %24, %27, %30) : (i32, i32, i32, memref<150x140x160xf64>, memref<160x160xf64>, memref<160xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %31 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %32 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = load %arg1[%33] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %35 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %36 = call @strcmp(%34, %35) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %37 = trunci %36 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %38 = xor %37, %true : i1
// CHECK-NEXT:     %39 = and %31, %38 : i1
// CHECK-NEXT:     scf.if %39 {
// CHECK-NEXT:       %43 = load %8[%c0] : memref<?xmemref<150x140x160xf64>>
// CHECK-NEXT:       %44 = memref_cast %43 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:       %45 = memref_cast %44 : memref<?x140x160xf64> to memref<150x140x160xf64>
// CHECK-NEXT:       call @print_array(%c150_i32, %c140_i32, %c160_i32, %45) : (i32, i32, i32, memref<150x140x160xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %40 = memref_cast %8 : memref<?xmemref<150x140x160xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%40) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %41 = memref_cast %11 : memref<?xmemref<160xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%41) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %42 = memref_cast %15 : memref<?xmemref<160x160xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%42) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<150x140x160xf64>, %arg4: memref<160x160xf64>) {
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
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg2 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = index_cast %0 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = memref_cast %arg3 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %10 = index_cast %2 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %9 : memref<?x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %13 = index_cast %5 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = muli %0, %2 : i32
// CHECK-NEXT:     %16 = addi %15, %5 : i32
// CHECK-NEXT:     %17 = remi_signed %16, %arg2 : i32
// CHECK-NEXT:     %18 = sitofp %17 : i32 to f64
// CHECK-NEXT:     %19 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %20 = divf %18, %19 : f64
// CHECK-NEXT:     store %20, %12[%8, %11, %14] : memref<?x140x160xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %21 = addi %5, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%21 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %22 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%22 : i32)
// CHECK-NEXT:   ^bb10(%23: i32):  // 2 preds: ^bb3, ^bb15
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg2 : i32
// CHECK-NEXT:     cond_br %24, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb13(%25: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:     %26 = cmpi "slt", %25, %arg2 : i32
// CHECK-NEXT:     cond_br %26, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %27 = index_cast %23 : i32 to index
// CHECK-NEXT:     %28 = addi %c0, %27 : index
// CHECK-NEXT:     %29 = memref_cast %arg4 : memref<160x160xf64> to memref<?x160xf64>
// CHECK-NEXT:     %30 = index_cast %25 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = muli %23, %25 : i32
// CHECK-NEXT:     %33 = remi_signed %32, %arg2 : i32
// CHECK-NEXT:     %34 = sitofp %33 : i32 to f64
// CHECK-NEXT:     %35 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %36 = divf %34, %35 : f64
// CHECK-NEXT:     store %36, %29[%28, %31] : memref<?x160xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %37 = addi %25, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb13(%37 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %38 = addi %23, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb10(%38 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<150x140x160xf64>, %arg4: memref<160x160xf64>, %arg5: memref<160xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb15
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb12
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg2 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = index_cast %5 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst, %arg5[%8] : memref<160xf64>
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%9: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg2 : i32
// CHECK-NEXT:     cond_br %10, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %11 = index_cast %0 : i32 to index
// CHECK-NEXT:     %12 = addi %c0, %11 : index
// CHECK-NEXT:     %13 = memref_cast %arg3 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %14 = index_cast %2 : i32 to index
// CHECK-NEXT:     %15 = addi %c0, %14 : index
// CHECK-NEXT:     %16 = memref_cast %13 : memref<?x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %17 = index_cast %9 : i32 to index
// CHECK-NEXT:     %18 = addi %c0, %17 : index
// CHECK-NEXT:     %19 = load %16[%12, %15, %18] : memref<?x140x160xf64>
// CHECK-NEXT:     %20 = memref_cast %arg4 : memref<160x160xf64> to memref<?x160xf64>
// CHECK-NEXT:     %21 = load %20[%18, %8] : memref<?x160xf64>
// CHECK-NEXT:     %22 = mulf %19, %21 : f64
// CHECK-NEXT:     %23 = load %arg5[%8] : memref<160xf64>
// CHECK-NEXT:     %24 = addf %23, %22 : f64
// CHECK-NEXT:     store %24, %arg5[%8] : memref<160xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %25 = addi %9, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb10(%25 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %5, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb7(%26 : i32)
// CHECK-NEXT:   ^bb13(%27: i32):  // 2 preds: ^bb9, ^bb14
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg2 : i32
// CHECK-NEXT:     cond_br %28, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %29 = index_cast %0 : i32 to index
// CHECK-NEXT:     %30 = addi %c0, %29 : index
// CHECK-NEXT:     %31 = memref_cast %arg3 : memref<150x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %32 = index_cast %2 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = memref_cast %31 : memref<?x140x160xf64> to memref<?x140x160xf64>
// CHECK-NEXT:     %35 = index_cast %27 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %arg5[%36] : memref<160xf64>
// CHECK-NEXT:     store %37, %34[%30, %33, %36] : memref<?x140x160xf64>
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %38 = addi %27, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb13(%38 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %39 = addi %2, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb4(%39 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<150x140x160xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb7(%5: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %6 = cmpi "slt", %5, %arg2 : i32
// CHECK-NEXT:     cond_br %6, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %7 = muli %0, %arg1 : i32
// CHECK-NEXT:     %8 = muli %7, %arg2 : i32
// CHECK-NEXT:     %9 = muli %2, %arg2 : i32
// CHECK-NEXT:     %10 = addi %8, %9 : i32
// CHECK-NEXT:     %11 = addi %10, %5 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %12 = remi_signed %11, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %13 = addi %5, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%13 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %2, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%14 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }