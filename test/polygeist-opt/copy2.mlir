// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s | FileCheck %s

module  {
  func.func private @_ZN11ACUDAStreamC1EOS_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.getelementptr %arg0[%c0_i32, 0] {elem_type = !llvm.struct<(struct<(i32, i32)>)>} : (!llvm.ptr, i32) -> !llvm.ptr
    %1 = llvm.getelementptr %arg1[%c0_i32, 0] {elem_type = !llvm.struct<(struct<(i32, i32)>)>} : (!llvm.ptr, i32) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x2xi32>
    %3 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x2xi32>
    %a0 = memref.load %3[%c0, %c0] : memref<?x2xi32>
    memref.store %a0, %2[%c0, %c0] : memref<?x2xi32>
    %a1 = memref.load %3[%c0, %c1] : memref<?x2xi32>
    memref.store %a1, %2[%c0, %c1] : memref<?x2xi32>
    return
  }
}

// CHECK-LABEL:   func.func private @_ZN11ACUDAStreamC1EOS_(
// CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr)
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_0]] : i32, !llvm.ptr
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_1]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_5]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

