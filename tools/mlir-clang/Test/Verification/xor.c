// RUN: mlir-clang %s --function=foo -S | FileCheck %s

void foo(int A[10], int a) {
  for (int i = 0; i < 10; ++i)
    A[i] ^= (a ^ (A[i] + 1));
}

// CHECK-LABEL: func @foo
// CHECK: scf.for 
// CHECK: %[[VAL0:.*]] = memref.load
// CHECK: %[[VAL1:.*]] = arith.addi %[[VAL0]], %{{.*}}
// CHECK: %[[VAL2:.*]] = arith.xori %{{.*}}, %[[VAL1]]
// CHECK: %[[VAL3:.*]] = memref.load
// CHECK: %[[VAL4:.*]] = arith.xori %[[VAL3]], %[[VAL2]]
// CHECK: memref.store %[[VAL4]]
