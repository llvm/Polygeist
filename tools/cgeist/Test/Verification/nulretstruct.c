// RUN: cgeist -S --function=* %s | FileCheck %s

struct C {
  int a;
  double* b;
};

struct C* make() {
    return (struct C*)0;
}

float* makeF() {
    return (float*)0;
}



// CHECK-LABEL:   func.func @make() -> memref<?x!llvm.struct<(i32, memref<?xf64>)>>  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, memref<?xf64>)>>
// CHECK:           return %[[VAL_1]] : memref<?x!llvm.struct<(i32, memref<?xf64>)>>
// CHECK:         }

// CHECK-LABEL:   func.func @makeF() -> memref<?xf32>  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:           return %[[VAL_1]] : memref<?xf32>
// CHECK:         }

