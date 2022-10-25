// RUN: cgeist %s --function=* -S | FileCheck %s

extern void print(char*);

class Root {
public:
    Root(int y) {
        print("Calling root");
    }
};

class FRoot {
public:
    FRoot() {
        print("Calling froot");
    }
};

class Sub : public Root, public FRoot {
public:
    Sub(int i, double y) : Root(i) {
        print("Calling Sub");
    }
};

void make() {
    Sub s(3, 3.14);
}

// CHECK:   func.func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 3.140000e+00 : f64
// CHECK-DAG:     %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x!llvm.struct<(struct<(i8)>, struct<(i8)>)>> to memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>
// CHECK-NEXT:     call @_ZN3SubC1Eid(%[[V1]], %[[c3_i32]], %[[cst]]) : (memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>, i32, f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN3SubC1Eid(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>, %[[arg1:.+]]: i32, %[[arg2:.+]]: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>) -> !llvm.ptr<struct<(struct<(i8)>, struct<(i8)>)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<struct<(struct<(i8)>, struct<(i8)>)>>) -> memref<?x!llvm.struct<(i8)>>
// CHECK-NEXT:     call @_ZN4RootC1Ei(%[[V1]], %[[arg1]]) : (memref<?x!llvm.struct<(i8)>>, i32) -> ()
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V0]][0, 1] : (!llvm.ptr<struct<(struct<(i8)>, struct<(i8)>)>>) -> !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:     %[[V3:.+]] = "polygeist.pointer2memref"(%[[V2]]) : (!llvm.ptr<struct<(i8)>>) -> memref<?x!llvm.struct<(i8)>>
// CHECK-NEXT:     call @_ZN5FRootC1Ev(%[[V3]]) : (memref<?x!llvm.struct<(i8)>>) -> ()
// CHECK-NEXT:     %[[V4:.+]] = llvm.mlir.addressof @str0 : !llvm.ptr<array<12 x i8>>
// CHECK-NEXT:     %[[V5:.+]] = "polygeist.pointer2memref"(%[[V4]]) : (!llvm.ptr<array<12 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%[[V5]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN4RootC1Ei(%[[arg0:.+]]: memref<?x!llvm.struct<(i8)>>, %[[arg1:.+]]: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.addressof @str1 : !llvm.ptr<array<13 x i8>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<array<13 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%[[V1]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN5FRootC1Ev(%[[arg0:.+]]: memref<?x!llvm.struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.addressof @str2 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<array<14 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%[[V1]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
