// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int deref(int a) {
    return a;
}

void kernel_deriche(int in) {
    int* iter = &in;
    for(; *iter != 10; (*iter)++) {
        deref(*iter);
    }
}

// CHECK:    func @kernel_deriche(%arg0: i32) {
// CHECK-NEXT:     %c10_i32 = constant 10 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = scf.while (%arg1 = %arg0) : (i32) -> i32 {
// CHECK-NEXT:       %1 = cmpi ne, %arg1, %c10_i32 : i32
// CHECK-NEXT:       scf.condition(%1) %arg1 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg1: i32):  // no predecessors
// CHECK-NEXT:       %1 = call @deref(%arg1) : (i32) -> i32
// CHECK-NEXT:       %2 = addi %arg1, %c1_i32 : i32
// CHECK-NEXT:       scf.yield %2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:  func @deref(%arg0: i32) -> i32 {
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
