// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s --allow-unregistered-dialect | FileCheck %s

module {
  llvm.mlir.global internal constant @str5("%d   \00") {addr_space = 0 : i32}
  llvm.func @scanf(!llvm.ptr, ...) -> i32
  func.func @overwrite(%arg: index, %arg2: index) -> (i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c2_i32 = arith.constant 2 : i32
    %alloca = memref.alloca(%arg) : memref<?xi32>
    %alloca2 = memref.alloca() : memref<i32>
    %ptr = "polygeist.memref2pointer"(%alloca2) : (memref<i32>) -> !llvm.ptr
    
    %6 = llvm.mlir.addressof @str5 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] {elem_type = !llvm.array<6 x i8>} : (!llvm.ptr) -> !llvm.ptr

    scf.for %arg4 = %c0 to %arg step %c1 {
      %12 = llvm.call @scanf(%7, %ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %ld = memref.load %alloca2[] : memref<i32>
      memref.store %ld, %alloca[%arg4] : memref<?xi32>
    }
    %a10 = memref.load %alloca[%arg2] : memref<?xi32>
    return %a10 : i32
  }
}

// CHECK:   func.func @overwrite(%arg0: index, %arg1: index) -> i32 {
// CHECK:     scf.for %arg2 = %c0 to %arg0 step %c1 {
// CHECK-NEXT:       %[[i4:.+]] = llvm.call @scanf(%[[i2:.+]], %[[i0:.+]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:       %[[i5:.+]] = memref.load %[[alloca_0:.+]][] : memref<i32>
// CHECK-NEXT:       memref.store %[[i5]], %[[alloca:.+]][%arg2] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[i3:.+]] = memref.load %[[alloca]][%arg1] : memref<?xi32>
// CHECK-NEXT:     return %[[i3]] : i32
// CHECK-NEXT:   }
