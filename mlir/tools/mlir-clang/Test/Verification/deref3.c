// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

long deref(long a) {
    return a;
}

void kernel_deriche(long* __attribute__((align_value(8192))) in) {
    deref((*in)++);
}

// CHECK:    func @kernel_deriche(%arg0: !llvm.ptr<i64>) {
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %0 = llvm.load %arg0 : !llvm.ptr<i64>
// CHECK-NEXT:     %1 = addi %0, %c1_i64 : i64
// CHECK-NEXT:     llvm.store %1, %arg0 : !llvm.ptr<i64>
// CHECK-NEXT:     %2 = call @deref(%0) : (i64) -> i64
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:  func @deref(%arg0: i64) -> i64 {
// CHECK-NEXT:    return %arg0 : i64
// CHECK-NEXT:  }
