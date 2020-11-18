// TODO: mlir-clang %s %stdinclude | FileCheck %s
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

static void subfn(int n, int i, DATA_TYPE POLYBENCH_2D(table,N,N,n,n)) {
   int j, k;
  for (j=i+1; j<_PB_N; j++) {

   //if (j-1>=0)
   //   table[i][j] = max_score(table[i][j], table[i][j-1]);
   //if (i+1<_PB_N)
   //   table[i][j] = max_score(table[i][j], table[i+1][j]);

   //if (j-1>=0 && i+1<_PB_N) {
     /* don't allow adjacent elements to bond */
   //  if (i<j-1)
   //     table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
  //   else
   //     table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   //}

   for (k=0; k<_PB_N; k++) {
   //for (k=i+1; k<_PB_N; k++) {
      //table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
      table[i][j] = table[i][j]+1;//max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
}

static
void kernel_nussinov(int n, base POLYBENCH_1D(seq,N,n),
			   DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

#pragma scop
 //for (i = _PB_N-1; i >= 0; i--) {
 for (i = 0; i < _PB_N; i++) {
    //subfn(n, i, table);
    
  for (j=i+1; j<_PB_N; j++) {

   //if (j-1>=0)
   //   table[i][j] = max_score(table[i][j], table[i][j-1]);
   //if (i+1<_PB_N)
   //   table[i][j] = max_score(table[i][j], table[i+1][j]);

   //if (j-1>=0 && i+1<_PB_N) {
   //  if (i<j-1)
   //     table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
  //   else
   //     table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   //}

   for (k=0; k<_PB_N; k++) {
   //for (k=i+1; k<_PB_N; k++) {
      //table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
      table[i][j] = table[i][j]+1;//max_score(table[i][j], table[i][k] + table[k+1][j]);
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
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%d \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("table\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     %4 = remi_signed %3, %c4_i32 : i32
// CHECK-NEXT:     %5 = trunci %4 : i32 to i8
// CHECK-NEXT:     store %5, %arg1[%2] : memref<2500xi8>
// CHECK-NEXT:     br ^bb1(%3 : i32)
// CHECK-NEXT:   ^bb3(%6: i32):  // 2 preds: ^bb1, ^bb7
// CHECK-NEXT:     %7 = cmpi "slt", %6, %arg0 : i32
// CHECK-NEXT:     cond_br %7, ^bb5(%c0_i32 : i32), ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb5(%8: i32):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:     %9 = cmpi "slt", %8, %arg0 : i32
// CHECK-NEXT:     cond_br %9, ^bb6, ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %10 = index_cast %6 : i32 to index
// CHECK-NEXT:     %11 = index_cast %8 : i32 to index
// CHECK-NEXT:     store %c0_i32, %arg2[%10, %11] : memref<2500x2500xi32>
// CHECK-NEXT:     %12 = addi %8, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%12 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %13 = addi %6, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%13 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_nussinov(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %false = constant false
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %0 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%0 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %2 = cmpi "sge", %1, %c0_i32 : i32
// CHECK-NEXT:     cond_br %2, ^bb2(%1 : i32), ^bb3
// CHECK-NEXT:   ^bb2(%3: i32):  // 2 preds: ^bb1, ^bb6
// CHECK-NEXT:     %4 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     %5 = cmpi "slt", %4, %arg0 : i32
// CHECK-NEXT:     cond_br %5, ^bb4, ^bb5
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %6 = subi %4, %c1_i32 : i32
// CHECK-NEXT:     %7 = cmpi "sge", %6, %c0_i32 : i32
// CHECK-NEXT:     scf.if %7 {
// CHECK-NEXT:       %24 = index_cast %1 : i32 to index
// CHECK-NEXT:       %25 = index_cast %4 : i32 to index
// CHECK-NEXT:       %26 = load %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:       %27 = index_cast %6 : i32 to index
// CHECK-NEXT:       %28 = load %arg2[%24, %27] : memref<2500x2500xi32>
// CHECK-NEXT:       %29 = cmpi "sge", %26, %28 : i32
// CHECK-NEXT:       %30 = scf.if %29 -> (i32) {
// CHECK-NEXT:         %31 = load %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:         scf.yield %31 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %31 = load %arg2[%24, %27] : memref<2500x2500xi32>
// CHECK-NEXT:         scf.yield %31 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       store %30, %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %8 = cmpi "slt", %4, %arg0 : i32
// CHECK-NEXT:     scf.if %8 {
// CHECK-NEXT:       %24 = index_cast %1 : i32 to index
// CHECK-NEXT:       %25 = index_cast %4 : i32 to index
// CHECK-NEXT:       %26 = load %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:       %27 = index_cast %4 : i32 to index
// CHECK-NEXT:       %28 = load %arg2[%27, %25] : memref<2500x2500xi32>
// CHECK-NEXT:       %29 = cmpi "sge", %26, %28 : i32
// CHECK-NEXT:       %30 = scf.if %29 -> (i32) {
// CHECK-NEXT:         %31 = load %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:         scf.yield %31 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %31 = load %arg2[%27, %25] : memref<2500x2500xi32>
// CHECK-NEXT:         scf.yield %31 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       store %30, %arg2[%24, %25] : memref<2500x2500xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %9 = scf.if %7 -> (i1) {
// CHECK-NEXT:       scf.yield %8 : i1
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %false : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.if %9 {
// CHECK-NEXT:       %24 = cmpi "slt", %1, %6 : i32
// CHECK-NEXT:       scf.if %24 {
// CHECK-NEXT:         %25 = index_cast %1 : i32 to index
// CHECK-NEXT:         %26 = index_cast %4 : i32 to index
// CHECK-NEXT:         %27 = load %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:         %28 = index_cast %4 : i32 to index
// CHECK-NEXT:         %29 = index_cast %6 : i32 to index
// CHECK-NEXT:         %30 = load %arg2[%28, %29] : memref<2500x2500xi32>
// CHECK-NEXT:         %31 = load %arg1[%25] : memref<2500xi8>
// CHECK-NEXT:         %32 = sexti %31 : i8 to i32
// CHECK-NEXT:         %33 = load %arg1[%26] : memref<2500xi8>
// CHECK-NEXT:         %34 = sexti %33 : i8 to i32
// CHECK-NEXT:         %35 = addi %32, %34 : i32
// CHECK-NEXT:         %36 = cmpi "eq", %35, %c3_i32 : i32
// CHECK-NEXT:         %37 = scf.if %36 -> (i32) {
// CHECK-NEXT:           scf.yield %c1_i32 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %c0_i32 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %38 = addi %30, %37 : i32
// CHECK-NEXT:         %39 = cmpi "sge", %27, %38 : i32
// CHECK-NEXT:         %40 = scf.if %39 -> (i32) {
// CHECK-NEXT:           %41 = load %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:           scf.yield %41 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %41 = load %arg2[%28, %29] : memref<2500x2500xi32>
// CHECK-NEXT:           %42 = load %arg1[%25] : memref<2500xi8>
// CHECK-NEXT:           %43 = sexti %42 : i8 to i32
// CHECK-NEXT:           %44 = load %arg1[%26] : memref<2500xi8>
// CHECK-NEXT:           %45 = sexti %44 : i8 to i32
// CHECK-NEXT:           %46 = addi %43, %45 : i32
// CHECK-NEXT:           %47 = cmpi "eq", %46, %c3_i32 : i32
// CHECK-NEXT:           %48 = scf.if %47 -> (i32) {
// CHECK-NEXT:             scf.yield %c1_i32 : i32
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %c0_i32 : i32
// CHECK-NEXT:           }
// CHECK-NEXT:           %49 = addi %41, %48 : i32
// CHECK-NEXT:           scf.yield %49 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         store %40, %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %25 = index_cast %1 : i32 to index
// CHECK-NEXT:         %26 = index_cast %4 : i32 to index
// CHECK-NEXT:         %27 = load %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:         %28 = index_cast %4 : i32 to index
// CHECK-NEXT:         %29 = index_cast %6 : i32 to index
// CHECK-NEXT:         %30 = load %arg2[%28, %29] : memref<2500x2500xi32>
// CHECK-NEXT:         %31 = cmpi "sge", %27, %30 : i32
// CHECK-NEXT:         %32 = scf.if %31 -> (i32) {
// CHECK-NEXT:           %33 = load %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:           scf.yield %33 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %33 = load %arg2[%28, %29] : memref<2500x2500xi32>
// CHECK-NEXT:           scf.yield %33 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         store %32, %arg2[%25, %26] : memref<2500x2500xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     br ^bb6(%4 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb2
// CHECK-NEXT:     %10 = subi %1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%10 : i32)
// CHECK-NEXT:   ^bb6(%11: i32):  // 2 preds: ^bb4, ^bb7
// CHECK-NEXT:     %12 = cmpi "slt", %11, %4 : i32
// CHECK-NEXT:     cond_br %12, ^bb7, ^bb2(%4 : i32)
// CHECK-NEXT:   ^bb7:  // pred: ^bb6
// CHECK-NEXT:     %13 = index_cast %1 : i32 to index
// CHECK-NEXT:     %14 = index_cast %4 : i32 to index
// CHECK-NEXT:     %15 = load %arg2[%13, %14] : memref<2500x2500xi32>
// CHECK-NEXT:     %16 = index_cast %11 : i32 to index
// CHECK-NEXT:     %17 = load %arg2[%13, %16] : memref<2500x2500xi32>
// CHECK-NEXT:     %18 = addi %11, %c1_i32 : i32
// CHECK-NEXT:     %19 = index_cast %18 : i32 to index
// CHECK-NEXT:     %20 = load %arg2[%19, %14] : memref<2500x2500xi32>
// CHECK-NEXT:     %21 = addi %17, %20 : i32
// CHECK-NEXT:     %22 = cmpi "sge", %15, %21 : i32
// CHECK-NEXT:     %23 = scf.if %22 -> (i32) {
// CHECK-NEXT:       %24 = load %arg2[%13, %14] : memref<2500x2500xi32>
// CHECK-NEXT:       scf.yield %24 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %24 = load %arg2[%13, %16] : memref<2500x2500xi32>
// CHECK-NEXT:       %25 = load %arg2[%19, %14] : memref<2500x2500xi32>
// CHECK-NEXT:       %26 = addi %24, %25 : i32
// CHECK-NEXT:       scf.yield %26 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     store %23, %arg2[%13, %14] : memref<2500x2500xi32>
// CHECK-NEXT:     br ^bb6(%18 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2500x2500xi32>) {
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
// CHECK-NEXT:     %10 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<6 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32, %c0_i32 : i32, i32)
// CHECK-NEXT:   ^bb1(%13: i32, %14: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %15 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %15, ^bb3(%13, %14 : i32, i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %16 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.load %16 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %18 = llvm.mlir.addressof @[[str5]] : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %20 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<6 x i8>>
// CHECK-NEXT:     %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %22 = llvm.call @fprintf(%17, %19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %23 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.load %23 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %25 = llvm.mlir.addressof @[[str6]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %26 = llvm.getelementptr %25[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %27 = llvm.call @fprintf(%24, %26) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%28: i32, %29: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %30 = cmpi "slt", %28, %arg0 : i32
// CHECK-NEXT:     cond_br %30, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %31 = remi_signed %29, %c20_i32 : i32
// CHECK-NEXT:     %32 = cmpi "eq", %31, %c0_i32 : i32
// CHECK-NEXT:     scf.if %32 {
// CHECK-NEXT:       %45 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %46 = llvm.load %45 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %47 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %48 = llvm.getelementptr %47[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %49 = llvm.call @fprintf(%46, %48) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = index_cast %13 : i32 to index
// CHECK-NEXT:     %38 = index_cast %28 : i32 to index
// CHECK-NEXT:     %39 = load %arg1[%37, %38] : memref<2500x2500xi32>
// CHECK-NEXT:     %40 = llvm.mlir.cast %39 : i32 to !llvm.i32
// CHECK-NEXT:     %41 = llvm.call @fprintf(%34, %36, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
// CHECK-NEXT:     %42 = addi %29, %c1_i32 : i32
// CHECK-NEXT:     %43 = addi %28, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%43, %42 : i32, i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %44 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%44, %29 : i32, i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}