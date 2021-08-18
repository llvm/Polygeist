// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

extern "C" {

struct pair {
    int x, y;
};
void sub(pair& a) {
    a.x++;
}

void kernel_deriche() {
    pair a;
    a.x = 32;;
    pair &b = a;
    sub(b);
}

}

// CHECK:   builtin.func @kernel_deriche() {
// CHECK-NEXT:     %c32_i32 = constant 32 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     call @_ZN4pairC1Ev(%0) : (memref<1x2xi32>) -> ()
// CHECK-NEXT:     affine.store %c32_i32, %0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     call @sub(%0) : (memref<1x2xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   builtin.func @_ZN4pairC1Ev(%arg0: memref<1x2xi32>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   builtin.func @sub(%arg0: memref<1x2xi32>) {
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     %1 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     affine.store %1, %arg0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
