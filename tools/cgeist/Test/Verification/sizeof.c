// RUN: cgeist %s --function=* -S | FileCheck %s

void* malloc(unsigned long);

struct Meta {
    float* f;
    char x;
};

struct Meta* create() {
    return (struct Meta*)malloc(sizeof(struct Meta));
}


// CHECK-LABEL:   func.func @create() -> memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = "polygeist.typeSize"() <{source = !llvm.struct<(memref<?xf32>, i8)>}> : () -> index
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.divui %[[VAL_0]], %[[VAL_0]] : index
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.alloc(%[[VAL_1]]) : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK:           return %[[VAL_2]] : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK:         }
