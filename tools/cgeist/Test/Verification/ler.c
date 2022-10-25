// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

int kernel_deriche(int *a) {
    a[3]++;
    return a[1];
}

// CHECK:  func @kernel_deriche(%[[arg0:.+]]: memref<?xi32>) -> i32
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[V0:.+]] = affine.load %[[arg0]][3] : memref<?xi32>
// CHECK-NEXT:    %[[V1:.+]] = arith.addi %[[V0]], %[[c1_i32]] : i32
// CHECK-NEXT:    affine.store %[[V1]], %[[arg0]][3] : memref<?xi32>
// CHECK-NEXT:    %[[V2:.+]] = affine.load %[[arg0]][1] : memref<?xi32>
// CHECK-NEXT:    return %[[V2]] : i32
// CHECK-NEXT:  }
