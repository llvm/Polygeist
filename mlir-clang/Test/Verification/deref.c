// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int deref(int a) {
    return a;
}

void kernel_deriche(int *a) {
    deref(*a);
}

// CHECK:    func @kernel_deriche(%arg0: memref<?xi32>) {
// CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:    %1 = call @deref(%0) : (i32) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @deref(%arg0: i32) -> i32 {
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
