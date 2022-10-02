// RUN: cgeist %s --function=* -S | FileCheck %s

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

// CHECK:   func.func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %cst = arith.constant 3.140000e+00 : f64
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>> to memref<?x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>>
// CHECK-NEXT:     call @_ZN3SubC1Eid(%1, %c3_i32, %cst) : (memref<?x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>>, i32, f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN3SubC1Eid(%arg0: memref<?x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>>, %arg1: i32, %arg2: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> memref<?x1xi32>
// CHECK-NEXT:     call @_ZN4RootC1Ei(%1, %arg1) : (memref<?x1xi32>, i32) -> ()
// CHECK-NEXT:     %2 = llvm.getelementptr %0[0, 1] : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> !llvm.ptr<struct<(f32)>>
// CHECK-NEXT:     %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<struct<(f32)>>) -> memref<?x1xf32>
// CHECK-NEXT:     call @_ZN5FRootC1Ev(%3) : (memref<?x1xf32>) -> ()
// CHECK-NEXT:     %4 = llvm.getelementptr %0[0, 2] : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %arg2, %4 : !llvm.ptr<f64>
// CHECK-NEXT:     %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<12 x i8>>
// CHECK-NEXT:     %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr<array<12 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%6) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN4RootC1Ei(%arg0: memref<?x1xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     affine.store %arg1, %arg0[0, 0] : memref<?x1xi32>
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str1 : !llvm.ptr<array<13 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<13 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%1) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN5FRootC1Ev(%arg0: memref<?x1xf32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %cst = arith.constant 2.180000e+00 : f32
// CHECK-NEXT:     affine.store %cst, %arg0[0, 0] : memref<?x1xf32>
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str2 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<14 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%1) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
