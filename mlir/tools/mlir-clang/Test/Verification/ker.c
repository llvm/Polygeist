// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int kernel_deriche(int a[30][40]) {
    a[3][5]++;
    return a[1][2];
}

// CHECK:  func @kernel_deriche(%arg0: memref<?x40xi32>) -> i32 {
// CHECK-NEXT:    %c1_i32 = constant 1 : i32
// CHECK-NEXT:    %c5 = constant 5 : index
// CHECK-NEXT:    %c3 = constant 3 : index
// CHECK-NEXT:    %c2 = constant 2 : index
// CHECK-NEXT:    %c1 = constant 1 : index
// CHECK-NEXT:    %0 = load %arg0[%c3, %c5] : memref<?x40xi32>
// CHECK-NEXT:    %1 = addi %0, %c1_i32 : i32
// CHECK-NEXT:    store %1, %arg0[%c3, %c5] : memref<?x40xi32>
// CHECK-NEXT:    %2 = load %arg0[%c1, %c2] : memref<?x40xi32>
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }