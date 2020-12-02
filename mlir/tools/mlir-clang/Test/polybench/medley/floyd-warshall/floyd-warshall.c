// TODO: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      for(i = 0; i < _PB_N; i++)
	for (j = 0; j < _PB_N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}

// CHECK:func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<2800x2800xi32>) {
//CHECK-NEXT:    %0 = index_cast %arg0 : i32 to index
//CHECK-NEXT:    affine.for %arg2 = 0 to %0 {
//CHECK-NEXT:      affine.for %arg3 = 0 to %0 {
//CHECK-NEXT:        %1 = affine.load %arg1[%arg3, %arg2] : memref<2800x2800xi32>
//CHECK-NEXT:        affine.for %arg4 = 0 to %0 {
//CHECK-NEXT:          %2 = affine.load %arg1[%arg3, %arg4] : memref<2800x2800xi32>
//CHECK-NEXT:         %3 = affine.load %arg1[%arg2, %arg4] : memref<2800x2800xi32>
//CHECK-NEXT:          %4 = addi %1, %3 : i32
//CHECK-NEXT:          %5 = cmpi "slt", %2, %4 : i32
//CHECK-NEXT:          %6 = scf.if %5 -> (i32) {
//CHECK-NEXT:            %7 = affine.load %arg1[%arg3, %arg4] : memref<2800x2800xi32>
//CHECK-NEXT:            scf.yield %7 : i32
//CHECK-NEXT:          } else {
//CHECK-NEXT:            %7 = affine.load %arg1[%arg3, %arg2] : memref<2800x2800xi32>
//CHECK-NEXT:            %8 = affine.load %arg1[%arg2, %arg4] : memref<2800x2800xi32>
//CHECK-NEXT:            %9 = addi %7, %8 : i32
//CHECK-NEXT:            scf.yield %9 : i32
//CHECK-NEXT:          }
//CHECK-NEXT:          affine.store %6, %arg1[%arg3, %arg4] : memref<2800x2800xi32>
//CHECK-NEXT:       }
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:    return

// EXEC: {{[0-9]\.[0-9]+}}
