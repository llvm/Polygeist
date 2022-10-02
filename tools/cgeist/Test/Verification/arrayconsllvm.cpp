// RUN: cgeist %s --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
    double v;
};

void kern() {
    AIntDivider sizes_[25];
}

// CHECK:   func.func @_Z4kernv() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c25 = arith.constant 25 : index
// CHECK-NEXT:     %0 = memref.alloca() : memref<25x!llvm.struct<(i32, f64)>>
// CHECK-NEXT:     scf.for %arg0 = %c0 to %c25 step %c1 {
// CHECK-NEXT:       %1 = "polygeist.subindex"(%0, %arg0) : (memref<25x!llvm.struct<(i32, f64)>>, index) -> memref<?x!llvm.struct<(i32, f64)>>
// CHECK-NEXT:       func.call @_ZN11AIntDividerC1Ev(%1) : (memref<?x!llvm.struct<(i32, f64)>>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT: return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN11AIntDividerC1Ev(%arg0: memref<?x!llvm.struct<(i32, f64)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c3_i32, %1 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
