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
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c2000_i32 = constant 2000 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c2000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = zexti %1 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %3 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %4 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %5 = memref_cast %4 : memref<?xi8> to memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %6 = zexti %0 : i32 to i64
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %9 = call @polybench_alloc_data(%6, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?xi8> to memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %11 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %12 = memref_cast %11 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %14 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %15 = memref_cast %14 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %17 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %18 = memref_cast %17 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @init_array(%c2000_i32, %13, %16, %19) : (i32, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %20 = load %5[%c0] : memref<?xmemref<2000x2000xf64>>
// CHECK-NEXT:     %21 = memref_cast %20 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<?x2000xf64> to memref<2000x2000xf64>
// CHECK-NEXT:     %23 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     %26 = load %10[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:     call @kernel_trisolv(%c2000_i32, %22, %25, %28) : (i32, memref<2000x2000xf64>, memref<2000xf64>, memref<2000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %29 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %30 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = load %arg1[%31] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %33 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %34 = call @strcmp(%32, %33) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %35 = trunci %34 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %36 = xor %35, %true : i1
// CHECK-NEXT:     %37 = and %29, %36 : i1
// CHECK-NEXT:     scf.if %37 {
// CHECK-NEXT:       %41 = load %8[%c0] : memref<?xmemref<2000xf64>>
// CHECK-NEXT:       %42 = memref_cast %41 : memref<2000xf64> to memref<?xf64>
// CHECK-NEXT:       %43 = memref_cast %42 : memref<?xf64> to memref<2000xf64>
// CHECK-NEXT:       call @print_array(%c2000_i32, %43) : (i32, memref<2000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %38 = memref_cast %5 : memref<?xmemref<2000x2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%38) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %39 = memref_cast %8 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%39) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %40 = memref_cast %10 : memref<?xmemref<2000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%40) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %c999_i32 = constant 999 : i32
// CHECK-NEXT:     %4 = subi %c0_i32, %c999_i32 : i32
// CHECK-NEXT:     %5 = sitofp %4 : i32 to f64
// CHECK-NEXT:     store %5, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %6 = sitofp %0 : i32 to f64
// CHECK-NEXT:     store %6, %arg3[%3] : memref<2000xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%7: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %8 = cmpi "sle", %7, %0 : i32
// CHECK-NEXT:     cond_br %8, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %9 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %10 = index_cast %7 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = addi %0, %arg0 : i32
// CHECK-NEXT:     %13 = subi %12, %7 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     %15 = sitofp %14 : i32 to f64
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %16 = sitofp %c2_i32 : i32 to f64
// CHECK-NEXT:     %17 = mulf %15, %16 : f64
// CHECK-NEXT:     %18 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %19 = divf %17, %18 : f64
// CHECK-NEXT:     store %19, %9[%3, %11] : memref<?x2000xf64>
// CHECK-NEXT:     %20 = addi %7, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%20 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %21 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%21 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_trisolv(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = load %arg3[%3] : memref<2000xf64>
// CHECK-NEXT:     store %4, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%5: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %6 = cmpi "slt", %5, %0 : i32
// CHECK-NEXT:     cond_br %6, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %7 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %8 = index_cast %5 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %10 = load %7[%3, %9] : memref<?x2000xf64>
// CHECK-NEXT:     %11 = load %arg2[%9] : memref<2000xf64>
// CHECK-NEXT:     %12 = mulf %10, %11 : f64
// CHECK-NEXT:     %13 = load %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %14 = subf %13, %12 : f64
// CHECK-NEXT:     store %14, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %15 = addi %5, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%15 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %16 = load %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %17 = memref_cast %arg1 : memref<2000x2000xf64> to memref<?x2000xf64>
// CHECK-NEXT:     %18 = load %17[%3, %3] : memref<?x2000xf64>
// CHECK-NEXT:     %19 = divf %16, %18 : f64
// CHECK-NEXT:     store %19, %arg2[%3] : memref<2000xf64>
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %20 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%20 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<2000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %2 = remi_signed %0, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %3 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%3 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }