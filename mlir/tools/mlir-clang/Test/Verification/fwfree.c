// RUN: mlir-clang %s %stdinclude | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

#   define N 2800

/* Array initialization. */
static
void init_array (int path[N])
{
  //path[0][1] = 2;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */

  /* Variable declaration/allocation. */
  //POLYBENCH_1D_ARRAY_DECL(path, int, N, n);
  int (*path)[N];
  //int path[POLYBENCH_C99_SELECT(N,n) + POLYBENCH_PADDING_FACTOR];
  path = (int(*)[N])polybench_alloc_data (N, sizeof(int)) ;

  /* Initialize array(s). */
  init_array (*path);

  POLYBENCH_FREE_ARRAY(path);
  return 0;
}

// CHECK:     func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = alloc() : memref<2800xi32>
// CHECK-NEXT:     %1 = memref_cast %0 : memref<2800xi32> to memref<?xi32>
// CHECK-NEXT:     call @init_array(%1) : (memref<?xi32>) -> ()
// CHECK-NEXT:     dealloc %0 : memref<2800xi32>
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @init_array(%arg0: memref<?xi32>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }