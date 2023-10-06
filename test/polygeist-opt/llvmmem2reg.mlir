// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// TODO: Fix mem2reg using opaque llvm pointers
// XFAIL: *

module {
  func.func @ll(%arg0: !llvm.ptr) -> !llvm.ptr {
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %arg0, %2 : !llvm.ptr, !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
    return %3 : !llvm.ptr
  }
}

// TODO Stopped working after opaque pointer update

// CHECK:   func.func @ll(%[[arg0:.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT:     %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:     return %[[arg0]] : !llvm.ptr
// CHECK-NEXT:   }

// -----

module {
  func.func @mixed(%mr : !llvm.ptr) {
    %2 = memref.alloc() : memref<2xf32>
    llvm.store %2, %mr : memref<2xf32>, !llvm.ptr
    return
  }
}

// CHECK-LABEL:   func.func @mixed(
// CHECK-SAME:                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloc() : memref<2xf32>
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_0]] : memref<2xf32>, !llvm.ptr
// CHECK:           return
// CHECK:         }
