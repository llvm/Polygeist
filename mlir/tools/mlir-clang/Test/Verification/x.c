// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int kernel_deriche(int x) {
    x++;
    x+=1;
    x*=2;
    return x;
}

// CHECK:  func @kernel_deriche(%arg0: i32) -> i32 {
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %0 = addi %arg0, %c1_i32 : i32
// CHECK-NEXT:     %1 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     %2 = muli %1, %c2_i32 : i32
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }