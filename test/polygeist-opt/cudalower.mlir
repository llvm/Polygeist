// RUN: polygeist-opt --parallel-lower --split-input-file %s | FileCheck %s

module attributes {llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", llvm.target_triple = "nvptx64-nvidia-cuda"}  {
  llvm.func @cudaMemcpy(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i32) -> i32
  func @_Z1aPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i64 = arith.constant 64 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr<i8>
    %1 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr<i8>
    %2 = llvm.call @cudaMemcpy(%0, %1, %c64_i64, %c1_i32) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i32) -> i32
    return %2 : i32
  }
}

// CHECK:   func @_Z1aPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%0, %1, %c64_i64, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }

