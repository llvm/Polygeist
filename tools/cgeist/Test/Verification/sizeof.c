// RUN: cgeist %s --function=* -S | FileCheck %s

void* malloc(unsigned long);

struct Meta {
    float* f;
    char x;
};

struct Meta* create() {
    return (struct Meta*)malloc(sizeof(struct Meta));
}

// CHECK:   func @create() -> memref<?x!llvm.struct<(memref<?xf32>, i8)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.typeSize"() {source = !llvm.struct<(memref<?xf32>, i8)>} : () -> index
// CHECK-NEXT:     %[[V1:.+]] = arith.divui %[[V0]], %[[V0]] : index
// CHECK-NEXT:     %[[V2:.+]] = memref.alloc(%[[V1]]) : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:     return %[[V2]] : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:   }
