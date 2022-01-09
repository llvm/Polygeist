// RUN: mlir-clang %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK:   func @sum(%arg0: !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = llvm.mlir.null : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %1 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %2 = arith.xori %1, %true : i1
// CHECK-NEXT:     %3 = scf.if %2 -> (f64) {
// CHECK-NEXT:       %4 = llvm.getelementptr %arg0[%c0_i32, 1] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:       %5 = llvm.load %4 : !llvm.ptr<f64>
// CHECK-NEXT:       %6 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32) -> !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %8 = call @sum(%7) : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK-NEXT:       %9 = arith.addf %5, %8 : f64
// CHECK-NEXT:       scf.yield %9 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %3 : f64
// CHECK-NEXT:   }
