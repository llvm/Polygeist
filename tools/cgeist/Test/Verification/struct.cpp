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

// CHECK:   func.func @func(%arg0: memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> f32 
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i8, memref<?xi8>, i8)>>) -> !llvm.ptr<!llvm.struct<(i8, memref<?xi8>, i8)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>, i8)>>) -> !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:     %3 = call @_Z5hloadPKv(%2) : (memref<?xi8>) -> f32
// CHECK-NEXT:     return %3 : f32
// CHECK-NEXT:   }
