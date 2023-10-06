// RUN: cgeist %s %stdinclude --function=func -S | FileCheck %s

float hload(const void* data);

struct OperandInfo {
  char dtype = 'a';

  void* data;

  bool end;
};

extern "C" {
float func(struct OperandInfo* op) {
    return hload(op->data);
}
}

// CHECK-LABEL:   func.func @func(
// CHECK-SAME:                    %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> f32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, memref<?xi8>, i8)>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> memref<?xi8>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = call @_Z5hloadPKv(%[[VAL_3]]) : (memref<?xi8>) -> f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }
