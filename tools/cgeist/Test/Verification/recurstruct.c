// RUN: cgeist %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK:   func.func @sum(%[[arg0:.+]]: memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.bitcast %[[V0]] : !llvm.ptr<i8> to !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %[[V3:.+]] = llvm.icmp "eq" %[[V1]], %[[V2]] : !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %[[V4:.+]] = scf.if %[[V3]] -> (f64) {
// CHECK-NEXT:       scf.yield %[[cst]] : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[V5:.+]] = llvm.getelementptr %[[V1]][0, 1] : (!llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:       %[[V6:.+]] = llvm.load %[[V5]] : !llvm.ptr<f64>
// CHECK-NEXT:       %[[V7:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:     %[[V8:.+]] = llvm.load %[[V7]] : !llvm.ptr<memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:     %[[V9:.+]] = func.call @sum(%[[V8:.+]]) : (memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK-NEXT:       %[[V10:.+]] = arith.addf %[[V6]], %[[V9]] : f64
// CHECK-NEXT:       scf.yield %[[V10]] : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V4]] : f64
// CHECK-NEXT:   }
