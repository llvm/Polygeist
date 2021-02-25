// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

void sub(int a[2]) {
    a[2]++;
}

void kernel_deriche() {
    int a[2];
    sub(a);
}

// CHECK:  func @kernel_deriche() {
// CHECK-NEXT:    %0 = alloca() : memref<2xi32>
// CHECK-NEXT:    %1 = memref_cast %0 : memref<2xi32> to memref<?xi32>
// CHECK-NEXT:    call @sub(%1) : (memref<?xi32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @sub(%arg0: memref<?xi32>) {
// CHECK-NEXT:    %c1_i32 = constant 1 : i32
// CHECK-NEXT:    %c2 = constant 2 : index
// CHECK-NEXT:    %0 = load %arg0[%c2] : memref<?xi32>
// CHECK-NEXT:    %1 = addi %0, %c1_i32 : i32
// CHECK-NEXT:    store %1, %arg0[%c2] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }