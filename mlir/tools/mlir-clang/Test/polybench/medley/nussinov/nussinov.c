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
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* RNA bases represented as chars, range is [0,3] */
typedef char base;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  //base is AGCT/0..3
  for (i=0; i <n; i++) {
     seq[i] = (base)((i+1)%4);
  }

  for (i=0; i <n; i++)
     for (j=0; j <n; j++)
       table[i][j] = 0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))

{
  int i, j;
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
      t++;
    }
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov(int n, base POLYBENCH_1D(seq,N,n),
			   DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

#pragma scop
 for (i = _PB_N-1; i >= 0; i--) {
  for (j=i+1; j<_PB_N; j++) {

   if (j-1>=0)
      table[i][j] = max_score(table[i][j], table[i][j-1]);
   if (i+1<_PB_N)
      table[i][j] = max_score(table[i][j], table[i+1][j]);

   if (j-1>=0 && i+1<_PB_N) {
     /* don't allow adjacent elements to bond */
     if (i<j-1)
        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
     else
        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
 }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_nussinov (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2500_i32 = constant 2500 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2500_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = zexti %0 : i32 to i64
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %2 = trunci %c1_i64 : i64 to i32
// CHECK-NEXT:     %3 = call @polybench_alloc_data(%1, %2) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %4 = memref_cast %3 : memref<?xi8> to memref<?xmemref<2500xi8>>
// CHECK-NEXT:     %5 = muli %0, %0 : i32
// CHECK-NEXT:     %6 = zexti %5 : i32 to i64
// CHECK-NEXT:     %c4_i64 = constant 4 : i64
// CHECK-NEXT:     %7 = trunci %c4_i64 : i64 to i32
// CHECK-NEXT:     %8 = call @polybench_alloc_data(%6, %7) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xi8> to memref<?xmemref<2500x2500xi32>>
// CHECK-NEXT:     %10 = load %4[%c0] : memref<?xmemref<2500xi8>>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<2500xi8> to memref<?xi8>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<?xi8> to memref<2500xi8>
// CHECK-NEXT:     %13 = load %9[%c0] : memref<?xmemref<2500x2500xi32>>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<?x2500xi32> to memref<2500x2500xi32>
// CHECK-NEXT:     call @init_array(%c2500_i32, %12, %15) : (i32, memref<2500xi8>, memref<2500x2500xi32>) -> ()
// CHECK-NEXT:     %16 = load %4[%c0] : memref<?xmemref<2500xi8>>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<2500xi8> to memref<?xi8>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<?xi8> to memref<2500xi8>
// CHECK-NEXT:     %19 = load %9[%c0] : memref<?xmemref<2500x2500xi32>>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<?x2500xi32> to memref<2500x2500xi32>
// CHECK-NEXT:     call @kernel_nussinov(%c2500_i32, %18, %21) : (i32, memref<2500xi8>, memref<2500x2500xi32>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %22 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %23 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     %25 = load %arg1[%24] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %26 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %27 = call @strcmp(%25, %26) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %28 = trunci %27 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %29 = xor %28, %true : i1
// CHECK-NEXT:     %30 = and %22, %29 : i1
// CHECK-NEXT:     scf.if %30 {
// CHECK-NEXT:       %33 = load %9[%c0] : memref<?xmemref<2500x2500xi32>>
// CHECK-NEXT:       %34 = memref_cast %33 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:       %35 = memref_cast %34 : memref<?x2500xi32> to memref<2500x2500xi32>
// CHECK-NEXT:       call @print_array(%c2500_i32, %35) : (i32, memref<2500x2500xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %31 = memref_cast %4 : memref<?xmemref<2500xi8>> to memref<?xi8>
// CHECK-NEXT:     call @free(%31) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %32 = memref_cast %9 : memref<?xmemref<2500x2500xi32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%32) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %4 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %5 = remi_signed %4, %c4_i32 : i32
// CHECK-NEXT:     %6 = trunci %5 : i32 to i8
// CHECK-NEXT:     store %6, %arg1[%3] : memref<2500xi8>
// CHECK-NEXT:     br ^bb1(%4 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%7: i32):  // 2 preds: ^bb3, ^bb9
// CHECK-NEXT:     %8 = cmpi "slt", %7, %arg0 : i32
// CHECK-NEXT:     cond_br %8, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%9: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg0 : i32
// CHECK-NEXT:     cond_br %10, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %11 = index_cast %7 : i32 to index
// CHECK-NEXT:     %12 = addi %c0, %11 : index
// CHECK-NEXT:     %13 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:     %14 = index_cast %9 : i32 to index
// CHECK-NEXT:     %15 = addi %c0, %14 : index
// CHECK-NEXT:     store %c0_i32, %13[%12, %15] : memref<?x2500xi32>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %9, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%16 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %7, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%17 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_nussinov(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = alloca() : memref<1xi32>
// CHECK-NEXT:     %1 = alloca() : memref<1xi32>
// CHECK-NEXT:     %2 = alloca() : memref<1xi32>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %3 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     store %3, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%3 : i32)
// CHECK-NEXT:   ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %5 = cmpi "sge", %4, %c0_i32 : i32
// CHECK-NEXT:     cond_br %5, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %6 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     store %6, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%6 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%7: i32):  // 2 preds: ^bb2, ^bb9
// CHECK-NEXT:     %8 = cmpi "slt", %7, %arg0 : i32
// CHECK-NEXT:     cond_br %8, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %9 = subi %7, %c1_i32 : i32
// CHECK-NEXT:     %10 = cmpi "sge", %9, %c0_i32 : i32
// CHECK-NEXT:     scf.if %10 {
// CHECK-NEXT:       %33 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %34 = index_cast %33 : i32 to index
// CHECK-NEXT:       %35 = addi %c0, %34 : index
// CHECK-NEXT:       %36 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:       %37 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %38 = index_cast %37 : i32 to index
// CHECK-NEXT:       %39 = addi %c0, %38 : index
// CHECK-NEXT:       %40 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %41 = index_cast %40 : i32 to index
// CHECK-NEXT:       %42 = addi %c0, %41 : index
// CHECK-NEXT:       %43 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %44 = index_cast %43 : i32 to index
// CHECK-NEXT:       %45 = addi %c0, %44 : index
// CHECK-NEXT:       %46 = load %36[%42, %45] : memref<?x2500xi32>
// CHECK-NEXT:       %47 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %48 = index_cast %47 : i32 to index
// CHECK-NEXT:       %49 = addi %c0, %48 : index
// CHECK-NEXT:       %50 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %51 = subi %50, %c1_i32 : i32
// CHECK-NEXT:       %52 = index_cast %51 : i32 to index
// CHECK-NEXT:       %53 = addi %c0, %52 : index
// CHECK-NEXT:       %54 = load %36[%49, %53] : memref<?x2500xi32>
// CHECK-NEXT:       %55 = cmpi "sge", %46, %54 : i32
// CHECK-NEXT:       %56 = scf.if %55 -> (i32) {
// CHECK-NEXT:         %57 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %58 = index_cast %57 : i32 to index
// CHECK-NEXT:         %59 = addi %c0, %58 : index
// CHECK-NEXT:         %60 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %61 = index_cast %60 : i32 to index
// CHECK-NEXT:         %62 = addi %c0, %61 : index
// CHECK-NEXT:         %63 = load %36[%59, %62] : memref<?x2500xi32>
// CHECK-NEXT:         scf.yield %63 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %57 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %58 = index_cast %57 : i32 to index
// CHECK-NEXT:         %59 = addi %c0, %58 : index
// CHECK-NEXT:         %60 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %61 = subi %60, %c1_i32 : i32
// CHECK-NEXT:         %62 = index_cast %61 : i32 to index
// CHECK-NEXT:         %63 = addi %c0, %62 : index
// CHECK-NEXT:         %64 = load %36[%59, %63] : memref<?x2500xi32>
// CHECK-NEXT:         scf.yield %64 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       store %56, %36[%35, %39] : memref<?x2500xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %11 = cmpi "slt", %6, %arg0 : i32
// CHECK-NEXT:     scf.if %11 {
// CHECK-NEXT:       %33 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %34 = index_cast %33 : i32 to index
// CHECK-NEXT:       %35 = addi %c0, %34 : index
// CHECK-NEXT:       %36 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:       %37 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %38 = index_cast %37 : i32 to index
// CHECK-NEXT:       %39 = addi %c0, %38 : index
// CHECK-NEXT:       %40 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %41 = index_cast %40 : i32 to index
// CHECK-NEXT:       %42 = addi %c0, %41 : index
// CHECK-NEXT:       %43 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %44 = index_cast %43 : i32 to index
// CHECK-NEXT:       %45 = addi %c0, %44 : index
// CHECK-NEXT:       %46 = load %36[%42, %45] : memref<?x2500xi32>
// CHECK-NEXT:       %47 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %48 = addi %47, %c1_i32 : i32
// CHECK-NEXT:       %49 = index_cast %48 : i32 to index
// CHECK-NEXT:       %50 = addi %c0, %49 : index
// CHECK-NEXT:       %51 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %52 = index_cast %51 : i32 to index
// CHECK-NEXT:       %53 = addi %c0, %52 : index
// CHECK-NEXT:       %54 = load %36[%50, %53] : memref<?x2500xi32>
// CHECK-NEXT:       %55 = cmpi "sge", %46, %54 : i32
// CHECK-NEXT:       %56 = scf.if %55 -> (i32) {
// CHECK-NEXT:         %57 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %58 = index_cast %57 : i32 to index
// CHECK-NEXT:         %59 = addi %c0, %58 : index
// CHECK-NEXT:         %60 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %61 = index_cast %60 : i32 to index
// CHECK-NEXT:         %62 = addi %c0, %61 : index
// CHECK-NEXT:         %63 = load %36[%59, %62] : memref<?x2500xi32>
// CHECK-NEXT:         scf.yield %63 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %57 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %58 = addi %57, %c1_i32 : i32
// CHECK-NEXT:         %59 = index_cast %58 : i32 to index
// CHECK-NEXT:         %60 = addi %c0, %59 : index
// CHECK-NEXT:         %61 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %62 = index_cast %61 : i32 to index
// CHECK-NEXT:         %63 = addi %c0, %62 : index
// CHECK-NEXT:         %64 = load %36[%60, %63] : memref<?x2500xi32>
// CHECK-NEXT:         scf.yield %64 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       store %56, %36[%35, %39] : memref<?x2500xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %12 = and %10, %11 : i1
// CHECK-NEXT:     scf.if %12 {
// CHECK-NEXT:       %33 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %34 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %35 = subi %34, %c1_i32 : i32
// CHECK-NEXT:       %36 = cmpi "slt", %33, %35 : i32
// CHECK-NEXT:       scf.if %36 {
// CHECK-NEXT:         %37 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %38 = index_cast %37 : i32 to index
// CHECK-NEXT:         %39 = addi %c0, %38 : index
// CHECK-NEXT:         %40 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:         %41 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %42 = index_cast %41 : i32 to index
// CHECK-NEXT:         %43 = addi %c0, %42 : index
// CHECK-NEXT:         %44 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %45 = index_cast %44 : i32 to index
// CHECK-NEXT:         %46 = addi %c0, %45 : index
// CHECK-NEXT:         %47 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %48 = index_cast %47 : i32 to index
// CHECK-NEXT:         %49 = addi %c0, %48 : index
// CHECK-NEXT:         %50 = load %40[%46, %49] : memref<?x2500xi32>
// CHECK-NEXT:         %51 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %52 = addi %51, %c1_i32 : i32
// CHECK-NEXT:         %53 = index_cast %52 : i32 to index
// CHECK-NEXT:         %54 = addi %c0, %53 : index
// CHECK-NEXT:         %55 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %56 = subi %55, %c1_i32 : i32
// CHECK-NEXT:         %57 = index_cast %56 : i32 to index
// CHECK-NEXT:         %58 = addi %c0, %57 : index
// CHECK-NEXT:         %59 = load %40[%54, %58] : memref<?x2500xi32>
// CHECK-NEXT:         %60 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %61 = index_cast %60 : i32 to index
// CHECK-NEXT:         %62 = addi %c0, %61 : index
// CHECK-NEXT:         %63 = load %arg1[%62] : memref<2500xi8>
// CHECK-NEXT:         %64 = sexti %63 : i8 to i32
// CHECK-NEXT:         %65 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %66 = index_cast %65 : i32 to index
// CHECK-NEXT:         %67 = addi %c0, %66 : index
// CHECK-NEXT:         %68 = load %arg1[%67] : memref<2500xi8>
// CHECK-NEXT:         %69 = sexti %68 : i8 to i32
// CHECK-NEXT:         %70 = addi %64, %69 : i32
// CHECK-NEXT:         %c3_i32 = constant 3 : i32
// CHECK-NEXT:         %71 = cmpi "eq", %70, %c3_i32 : i32
// CHECK-NEXT:         %72 = scf.if %71 -> (i32) {
// CHECK-NEXT:           scf.yield %c1_i32 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %c0_i32 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %73 = addi %59, %72 : i32
// CHECK-NEXT:         %74 = cmpi "sge", %50, %73 : i32
// CHECK-NEXT:         %75 = scf.if %74 -> (i32) {
// CHECK-NEXT:           %76 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:           %77 = index_cast %76 : i32 to index
// CHECK-NEXT:           %78 = addi %c0, %77 : index
// CHECK-NEXT:           %79 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:           %80 = index_cast %79 : i32 to index
// CHECK-NEXT:           %81 = addi %c0, %80 : index
// CHECK-NEXT:           %82 = load %40[%78, %81] : memref<?x2500xi32>
// CHECK-NEXT:           scf.yield %82 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %76 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:           %77 = addi %76, %c1_i32 : i32
// CHECK-NEXT:           %78 = index_cast %77 : i32 to index
// CHECK-NEXT:           %79 = addi %c0, %78 : index
// CHECK-NEXT:           %80 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:           %81 = subi %80, %c1_i32 : i32
// CHECK-NEXT:           %82 = index_cast %81 : i32 to index
// CHECK-NEXT:           %83 = addi %c0, %82 : index
// CHECK-NEXT:           %84 = load %40[%79, %83] : memref<?x2500xi32>
// CHECK-NEXT:           %85 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:           %86 = index_cast %85 : i32 to index
// CHECK-NEXT:           %87 = addi %c0, %86 : index
// CHECK-NEXT:           %88 = load %arg1[%87] : memref<2500xi8>
// CHECK-NEXT:           %89 = sexti %88 : i8 to i32
// CHECK-NEXT:           %90 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:           %91 = index_cast %90 : i32 to index
// CHECK-NEXT:           %92 = addi %c0, %91 : index
// CHECK-NEXT:           %93 = load %arg1[%92] : memref<2500xi8>
// CHECK-NEXT:           %94 = sexti %93 : i8 to i32
// CHECK-NEXT:           %95 = addi %89, %94 : i32
// CHECK-NEXT:           %96 = cmpi "eq", %95, %c3_i32 : i32
// CHECK-NEXT:           %97 = scf.if %96 -> (i32) {
// CHECK-NEXT:             scf.yield %c1_i32 : i32
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %c0_i32 : i32
// CHECK-NEXT:           }
// CHECK-NEXT:           %98 = addi %84, %97 : i32
// CHECK-NEXT:           scf.yield %98 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         store %75, %40[%39, %43] : memref<?x2500xi32>
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %37 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %38 = index_cast %37 : i32 to index
// CHECK-NEXT:         %39 = addi %c0, %38 : index
// CHECK-NEXT:         %40 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:         %41 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %42 = index_cast %41 : i32 to index
// CHECK-NEXT:         %43 = addi %c0, %42 : index
// CHECK-NEXT:         %44 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %45 = index_cast %44 : i32 to index
// CHECK-NEXT:         %46 = addi %c0, %45 : index
// CHECK-NEXT:         %47 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %48 = index_cast %47 : i32 to index
// CHECK-NEXT:         %49 = addi %c0, %48 : index
// CHECK-NEXT:         %50 = load %40[%46, %49] : memref<?x2500xi32>
// CHECK-NEXT:         %51 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:         %52 = addi %51, %c1_i32 : i32
// CHECK-NEXT:         %53 = index_cast %52 : i32 to index
// CHECK-NEXT:         %54 = addi %c0, %53 : index
// CHECK-NEXT:         %55 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:         %56 = subi %55, %c1_i32 : i32
// CHECK-NEXT:         %57 = index_cast %56 : i32 to index
// CHECK-NEXT:         %58 = addi %c0, %57 : index
// CHECK-NEXT:         %59 = load %40[%54, %58] : memref<?x2500xi32>
// CHECK-NEXT:         %60 = cmpi "sge", %50, %59 : i32
// CHECK-NEXT:         %61 = scf.if %60 -> (i32) {
// CHECK-NEXT:           %62 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:           %63 = index_cast %62 : i32 to index
// CHECK-NEXT:           %64 = addi %c0, %63 : index
// CHECK-NEXT:           %65 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:           %66 = index_cast %65 : i32 to index
// CHECK-NEXT:           %67 = addi %c0, %66 : index
// CHECK-NEXT:           %68 = load %40[%64, %67] : memref<?x2500xi32>
// CHECK-NEXT:           scf.yield %68 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %62 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:           %63 = addi %62, %c1_i32 : i32
// CHECK-NEXT:           %64 = index_cast %63 : i32 to index
// CHECK-NEXT:           %65 = addi %c0, %64 : index
// CHECK-NEXT:           %66 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:           %67 = subi %66, %c1_i32 : i32
// CHECK-NEXT:           %68 = index_cast %67 : i32 to index
// CHECK-NEXT:           %69 = addi %c0, %68 : index
// CHECK-NEXT:           %70 = load %40[%65, %69] : memref<?x2500xi32>
// CHECK-NEXT:           scf.yield %70 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         store %61, %40[%39, %43] : memref<?x2500xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     store %6, %2[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%6 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %13 = subi %4, %c1_i32 : i32
// CHECK-NEXT:     store %13, %0[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb1(%13 : i32)
// CHECK-NEXT:   ^bb7(%14: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %15 = cmpi "slt", %14, %7 : i32
// CHECK-NEXT:     cond_br %15, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %16 = index_cast %4 : i32 to index
// CHECK-NEXT:     %17 = addi %c0, %16 : index
// CHECK-NEXT:     %18 = memref_cast %arg2 : memref<2500x2500xi32> to memref<?x2500xi32>
// CHECK-NEXT:     %19 = index_cast %7 : i32 to index
// CHECK-NEXT:     %20 = addi %c0, %19 : index
// CHECK-NEXT:     %21 = load %18[%17, %20] : memref<?x2500xi32>
// CHECK-NEXT:     %22 = index_cast %14 : i32 to index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = load %18[%17, %23] : memref<?x2500xi32>
// CHECK-NEXT:     %25 = addi %14, %c1_i32 : i32
// CHECK-NEXT:     %26 = index_cast %25 : i32 to index
// CHECK-NEXT:     %27 = addi %c0, %26 : index
// CHECK-NEXT:     %28 = load %18[%27, %20] : memref<?x2500xi32>
// CHECK-NEXT:     %29 = addi %24, %28 : i32
// CHECK-NEXT:     %30 = cmpi "sge", %21, %29 : i32
// CHECK-NEXT:     %31 = scf.if %30 -> (i32) {
// CHECK-NEXT:       %33 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %34 = index_cast %33 : i32 to index
// CHECK-NEXT:       %35 = addi %c0, %34 : index
// CHECK-NEXT:       %36 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %37 = index_cast %36 : i32 to index
// CHECK-NEXT:       %38 = addi %c0, %37 : index
// CHECK-NEXT:       %39 = load %18[%35, %38] : memref<?x2500xi32>
// CHECK-NEXT:       scf.yield %39 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %33 = load %0[%c0] : memref<1xi32>
// CHECK-NEXT:       %34 = index_cast %33 : i32 to index
// CHECK-NEXT:       %35 = addi %c0, %34 : index
// CHECK-NEXT:       %36 = load %2[%c0] : memref<1xi32>
// CHECK-NEXT:       %37 = index_cast %36 : i32 to index
// CHECK-NEXT:       %38 = addi %c0, %37 : index
// CHECK-NEXT:       %39 = load %18[%35, %38] : memref<?x2500xi32>
// CHECK-NEXT:       %40 = load %2[%c0] : memref<1xi32>
// CHECK-NEXT:       %41 = addi %40, %c1_i32 : i32
// CHECK-NEXT:       %42 = index_cast %41 : i32 to index
// CHECK-NEXT:       %43 = addi %c0, %42 : index
// CHECK-NEXT:       %44 = load %1[%c0] : memref<1xi32>
// CHECK-NEXT:       %45 = index_cast %44 : i32 to index
// CHECK-NEXT:       %46 = addi %c0, %45 : index
// CHECK-NEXT:       %47 = load %18[%43, %46] : memref<?x2500xi32>
// CHECK-NEXT:       %48 = addi %39, %47 : i32
// CHECK-NEXT:       scf.yield %48 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     store %31, %18[%17, %20] : memref<?x2500xi32>
// CHECK-NEXT:     store %25, %2[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb7(%25 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %32 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     store %32, %1[%c0] : memref<1xi32>
// CHECK-NEXT:     br ^bb4(%32 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2500x2500xi32>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32, %c0_i32 : i32, i32)
// CHECK-NEXT:   ^bb1(%0: i32, %1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %2 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%0, %1 : i32, i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%3: i32, %4: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %5 = cmpi "slt", %3, %arg0 : i32
// CHECK-NEXT:     cond_br %5, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %6 = remi_signed %4, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     %8 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%8, %7 : i32, i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %9 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%9, %4 : i32, i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }