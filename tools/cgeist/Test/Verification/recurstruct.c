// RUN: cgeist %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK:   func.func @sum(%arg0: memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %2 = llvm.bitcast %0 : !llvm.ptr<i8> to !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %4 = scf.if %3 -> (f64) {
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %5 = llvm.getelementptr %1[0, 1] : (!llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:       %6 = llvm.load %5 : !llvm.ptr<f64>
// CHECK-NEXT:       %7 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr<memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:     %8 = llvm.load %7 : !llvm.ptr<memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:     %9 = func.call @sum(%8) : (memref<?x!llvm.struct<"polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK-NEXT:       %10 = arith.addf %6, %9 : f64
// CHECK-NEXT:       scf.yield %10 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %4 : f64
// CHECK-NEXT:   }
