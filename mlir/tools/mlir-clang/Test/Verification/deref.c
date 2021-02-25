// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int deref(int a) {
    return a;
}

void kernel_deriche(int *a) {
    deref(*a);
}

// CHECK:    func @kernel_deriche(%arg0: memref<?xi32>) {
// CHECK-NEXT:    %c0 = constant 0 : index
// CHECK-NEXT:    %0 = load %arg0[%c0] : memref<?xi32>
// CHECK-NEXT:    %1 = call @deref(%0) : (i32) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @deref(%arg0: i32) -> i32 {
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }