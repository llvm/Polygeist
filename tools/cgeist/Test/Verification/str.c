// RUN: cgeist %s --function=meta -S | FileCheck %s

int foo(const char*);

int meta() {
    return foo("bar") + foo(__PRETTY_FUNCTION__);
}

// CHECK:   llvm.mlir.global internal constant @str1("int meta()\00")
// CHECK:   llvm.mlir.global internal constant @str0("bar\00")
// CHECK:   func @meta() -> i32 attributes {llvm.linkage =
// #llvm.linkage<external>} { CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.addressof @str0 :
// !llvm.ptr<array<4 x i8>> CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]])
// : (!llvm.ptr<array<4 x i8>>) -> memref<?xi8> CHECK-NEXT:     %[[V2:.+]] = call
// @foo(%[[V1]]) : (memref<?xi8>) -> i32 CHECK-NEXT:     %[[V3:.+]] = llvm.mlir.addressof
// @str1 : !llvm.ptr<array<11 x i8>> CHECK-NEXT:     %[[V4:.+]] =
// "polygeist.pointer2memref"(%[[V3]]) : (!llvm.ptr<array<11 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     %[[V5:.+]] = call @foo(%[[V4]]) : (memref<?xi8>) -> i32
// CHECK-NEXT:     %[[V6:.+]] = arith.addi %[[V2]], %[[V5]] : i32
// CHECK-NEXT:     return %[[V6]] : i32
// CHECK-NEXT:   }
