// RUN: cgeist %s --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
};

struct Meta {
    AIntDivider sizes_[25];
    double x;
};

void kern() {
    Meta m;
}

// CHECK:   func.func @_Z4kernv() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x!llvm.struct<(array<25 x struct<(i32)>>, f64)>> to memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK-NEXT:     call @_ZN4MetaC1Ev(%1) : (memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN4MetaC1Ev(%arg0: memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c25 = arith.constant 25 : index
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>) -> !llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>) -> memref<25x1xi32>
// CHECK-NEXT:     scf.for %arg1 = %c0 to %c25 step %c1 {
// CHECK-NEXT:       %2 = "polygeist.subindex"(%1, %arg1) : (memref<25x1xi32>, index) -> memref<?x1xi32>
// CHECK-NEXT:       func.call @_ZN11AIntDividerC1Ev(%2) : (memref<?x1xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN11AIntDividerC1Ev(%arg0: memref<?x1xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     affine.store %c3_i32, %arg0[0, 0] : memref<?x1xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
