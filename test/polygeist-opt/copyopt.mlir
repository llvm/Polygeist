// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

module {
  func.func @cpy(%46: i64, %66: memref<?xi32>, %51: memref<?xi32>) {
    %c4_i64 = arith.constant 4 : i64
    %false = arith.constant false
      %47 = arith.muli %46, %c4_i64 : i64
      %48 = arith.trunci %47 : i64 to i32
      %67 = "polygeist.memref2pointer"(%66) : (memref<?xi32>) -> !llvm.ptr<i8>
      %68 = "polygeist.memref2pointer"(%51) : (memref<?xi32>) -> !llvm.ptr<i8>
      %69 = arith.extsi %48 : i32 to i64
      "llvm.intr.memcpy"(%67, %68, %69, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    return
  }
}

// CHECK:   func.func @cpy(%[[arg0:.+]]: i64, %[[arg1:.+]]: memref<?xi32>, %[[arg2:.+]]: memref<?xi32>) {
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[c4:.+]] = arith.constant 4 : index
// CHECK-NEXT:     %[[c4_i64:.+]] = arith.constant 4 : i64
// CHECK-NEXT:     %[[V0:.+]] = arith.muli %[[arg0]], %[[c4_i64]] : i64
// CHECK-NEXT:     %[[V1:.+]] = arith.trunci %[[V0]] : i64 to i32
// CHECK-NEXT:     %[[V2:.+]] = arith.index_cast %[[V1]] : i32 to index
// CHECK-NEXT:     %[[V3:.+]] = arith.divui %[[V2]], %[[c4]] : index
// CHECK-NEXT:     scf.for %[[arg3:.+]] = %[[c0]] to %[[V3]] step %[[c1]] {
// CHECK-NEXT:       %[[V4:.+]] = memref.load %[[arg2]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:       memref.store %[[V4]], %[[arg1]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
