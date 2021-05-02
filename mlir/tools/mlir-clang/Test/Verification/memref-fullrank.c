// RUN: mlir-clang %s -memref-fullrank | FileCheck %s
#include <stdio.h>

int f(int A[10][20]) {
#pragma scop
  int i, j, sum = 0;
  for (i = 0; i < 10; i ++)
    for (j = 0; j < 20; j ++)
      sum += A[i][j];
#pragma endscop
  return sum;
}

// CHECK: func @main() -> i32
int main() {
  int A[10][20];

  // CHECK: %[[VAL:.*]] = memref.alloca() : memref<10x20xi32>
  // CHECK: call @f(%[[VAL]]) : (memref<10x20xi32>) -> i32
  printf("%d\n", f(A));
}

// CHECK: func @f(%arg0: memref<10x20xi32>) -> i32
