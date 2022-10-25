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

// CHECK:   func.func @func(%[[arg0:.+]]: memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> f32 
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> !llvm.ptr<!llvm.struct<(i8, memref<?xi8>, i8)>>
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 1] : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>, i8)>>) -> !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.load %[[V1]] : !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:     %[[V3:.+]] = call @_Z5hloadPKv(%[[V2]]) : (memref<?xi8>) -> f32
// CHECK-NEXT:     return %[[V3]] : f32
// CHECK-NEXT:   }
