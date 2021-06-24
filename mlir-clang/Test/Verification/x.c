// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int kernel_deriche(int x) {
    x++;
    x+=3;
    x*=2;
    return x;
}

// CHECK:  func @kernel_deriche(%arg0: i32) -> i32 {
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %c4_i32 = constant 4 : i32
// CHECK-NEXT:     %0 = addi %arg0, %c4_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %c2_i32 : i32
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:   }