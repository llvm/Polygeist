// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  llvm.func @scanf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str4("%d\00") {addr_space = 0 : i32}
  func.func @_Z8BFSGraphiPPc(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca[0] : memref<1xi32>
    affine.store %c0_i32, %alloca[0] : memref<1xi32>
    %4 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
    %8 = llvm.mlir.addressof @str4 : !llvm.ptr
    %9 = llvm.getelementptr %8[0, 0] {elem_type = !llvm.array<3 x i8>} : (!llvm.ptr) -> !llvm.ptr
    %10 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr
    %11 = llvm.call @scanf(%9, %10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %12 = affine.load %alloca[0] : memref<1xi32>
    %13 = affine.load %alloca[0] : memref<1xi32>
    return %13, %12 : i32, i32
  }
// CHECK:          %[[i4:.+]] = "polygeist.memref2pointer"(%[[alloca:.+]]) : (memref<1xi32>) -> !llvm.ptr
// CHECK-NEXT:     %[[i5:.+]] = llvm.call @scanf(%[[i3:.+]], %[[i4]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:     %[[i6:.+]] = affine.load %[[alloca]][0] : memref<1xi32>
// CHECK-NEXT:     return %[[i6]], %[[i6]] : i32, i32
}

