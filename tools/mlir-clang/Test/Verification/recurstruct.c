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
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = memref.alloca() : memref<f64>
// CHECK-NEXT:     %1 = affine.load %0[] : memref<f64>
// CHECK-NEXT:     %2 = llvm.mlir.null : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %3 = llvm.icmp "eq" %arg0, %2 : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %4 = arith.xori %3, %true : i1
// CHECK-NEXT:     %5 = scf.if %3 -> (f64) {
// CHECK-NEXT:       affine.store %cst, %0[] : memref<f64>
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %1 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %6 = scf.if %4 -> (f64) {
// CHECK-NEXT:       %7 = llvm.getelementptr %arg0[%c0_i32, %c1_i32] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:       %8 = llvm.load %7 : !llvm.ptr<f64>
// CHECK-NEXT:       %9 = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32, i32) -> !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %10 = llvm.load %9 : !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %11 = call @sum(%10) : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK-NEXT:       %12 = arith.addf %8, %11 : f64
// CHECK-NEXT:       affine.store %12, %0[] : memref<f64>
// CHECK-NEXT:       scf.yield %12 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %6 : f64
// CHECK-NEXT:   }
