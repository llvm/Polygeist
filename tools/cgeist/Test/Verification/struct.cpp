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

// CHECK:   func @func(%arg0: memref<?x!llvm.struct<(i8, ptr<i8>, i8)>>) -> f32
// attributes {llvm.linkage = #llvm.linkage<external>} { CHECK-NEXT:     %0 =
// "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i8, ptr<i8>,
// i8)>>) -> !llvm.ptr<struct<(i8, ptr<i8>, i8)>> CHECK-NEXT:     %1 =
// llvm.getelementptr %0[0, 1] : (!llvm.ptr<struct<(i8, ptr<i8>, i8)>>) ->
// !llvm.ptr<ptr<i8>> CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %3 = call @_Z5hloadPKv(%2) : (!llvm.ptr<i8>) -> f32
// CHECK-NEXT:     return %3 : f32
// CHECK-NEXT:   }
