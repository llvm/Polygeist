// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - | FileCheck %s --check-prefix EXEC
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "template-for-new-benchmark.h"


/* Array initialization. */
static
void init_array(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = 42;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_template(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      C[i][j] += 42;
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(C));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_template (n, POLYBENCH_ARRAY(C));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}