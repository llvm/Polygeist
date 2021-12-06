// RUN: mlir-clang %s --function=* -S | FileCheck %s

extern void print(char*);

class Root {
public:
    int x;
    Root(int y) : x(y) {
        print("Calling root");
    }
};

class FRoot {
public:
    float f;
    FRoot() : f(2.18) {
        print("Calling froot");
    }
};

class Sub : public Root, public FRoot {
public:
    double d;
    Sub(int i, double y) : Root(i), d(y) {
        print("Calling Sub");
    }
};

void make() {
    Sub s(3, 3.14);
}

// CHECK:   func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %cst = arith.constant 3.140000e+00 : f64
// CHECK-NEXT:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)> : (i64) -> !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>
// CHECK-NEXT:     call @_ZN3SubC1Eid(%0, %c3_i32, %cst) : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>, i32, f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN3SubC1Eid(%arg0: !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>, %arg1: i32, %arg2: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> memref<?x1xi32>
// CHECK-NEXT:     call @_ZN4RootC1Ei(%0, %arg1) : (memref<?x1xi32>, i32) -> ()
// CHECK-NEXT:     %1 = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[%c4_i32] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:     %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<i8>) -> memref<?x1xf32>
// CHECK-NEXT:     call @_ZN5FRootC1Ev(%3) : (memref<?x1xf32>) -> ()
// CHECK-NEXT:     %4 = llvm.getelementptr %arg0[%c0_i32, %c2_i32] : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %arg2, %4 : !llvm.ptr<f64>
// CHECK-NEXT:     %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<12 x i8>>
// CHECK-NEXT:     %6 = llvm.getelementptr %5[%c0_i32, %c0_i32] : (!llvm.ptr<array<12 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%6) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN4RootC1Ei(%arg0: memref<?x1xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     affine.store %arg1, %arg0[0, 0] : memref<?x1xi32>
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str1 : !llvm.ptr<array<13 x i8>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<array<13 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%1) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN5FRootC1Ev(%arg0: memref<?x1xf32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant 2.180000e+00 : f64
// CHECK-NEXT:     %0 = arith.truncf %cst : f64 to f32
// CHECK-NEXT:     affine.store %0, %arg0[0, 0] : memref<?x1xf32>
// CHECK-NEXT:     %1 = llvm.mlir.addressof @str2 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[%c0_i32, %c0_i32] : (!llvm.ptr<array<14 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%2) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
