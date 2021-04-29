// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int kernel_deriche(int x) {
    x++;
    x+=1;
    x*=2;
    return x;
}

// CHECK:  func @kernel_deriche(%[[arg:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[cst:.+]] = constant 2 : i32
// CHECK-NEXT:     %[[add:.+]] = addi %[[arg]], %[[cst]] : i32
// CHECK-NEXT:     %[[mul:.+]] = muli %[[add]], %[[cst]] : i32
// CHECK-NEXT:     return %[[mul]] : i32
// CHECK-NEXT:   }
