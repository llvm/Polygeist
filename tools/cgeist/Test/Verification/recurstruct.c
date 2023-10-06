// RUN: cgeist %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK-LABEL:   func.func @sum(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node">>, f64)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.icmp "eq" %[[VAL_3]], %[[VAL_2]] : !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (f64) {
// CHECK:             scf.yield %[[VAL_1]] : f64
// CHECK:           } else {
// CHECK:             %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"opaque@polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node">>, f64)>
// CHECK:             %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> f64
// CHECK:             %[[VAL_8:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node">>, f64)>>
// CHECK:             %[[VAL_9:.*]] = func.call @sum(%[[VAL_8]]) : (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK:             %[[VAL_10:.*]] = arith.addf %[[VAL_7]], %[[VAL_9]] : f64
// CHECK:             scf.yield %[[VAL_10]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_5]] : f64
// CHECK:         }

