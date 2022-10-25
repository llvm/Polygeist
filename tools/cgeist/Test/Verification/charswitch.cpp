// RUN: cgeist %s --function=foo -S | FileCheck %s

extern "C" {
int foo(char t) {
  int n = 10;
  switch (t) {
  case 'a':
    n = 20;
    break;
  case 'A':
    n = 30;
    break;
  default:
    return -1;
  }
  return n;
}
}

// TODO the select should be canonicalized better
// CHECK:   func @foo(%[[arg0:.+]]: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[cm1:.+]] = arith.constant -1 : i32
// CHECK-DAG:     %[[c30_i32:.+]] = arith.constant 30 : i32
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-DAG:     %[[c20_i32:.+]] = arith.constant 20 : i32
// CHECK-DAG:     %[[c10_i32:.+]] = arith.constant 10 : i32
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     %[[V0:.+]] = llvm.mlir.undef : i32
// CHECK-DAG:     %[[V1:.+]] = arith.extsi %[[arg0]] : i8 to i32
// CHECK-NEXT:     switch %[[V1]] : i32, [
// CHECK-NEXT:       default: ^bb1(%[[c10_i32]], %[[false]], %[[cm1]] : i32, i1, i32),
// CHECK-NEXT:       97: ^bb1(%[[c20_i32]], %[[true]], %[[V0]] : i32, i1, i32),
// CHECK-NEXT:       65: ^bb1(%[[c30_i32]], %[[true]], %[[V0]] : i32, i1, i32)
// CHECK-NEXT:     ]
// CHECK-NEXT:   ^bb1(%[[V2:.+]]: i32, %[[V3:.+]]: i1, %[[V4:.+]]: i32):  // 3 preds: ^bb0, ^bb0, ^bb0
// CHECK-NEXT:     %[[V5:.+]] = arith.select %[[V3]], %[[V2]], %[[V4]] : i32
// CHECK-NEXT:     return %[[V5]] : i32
// CHECK-NEXT:   }
