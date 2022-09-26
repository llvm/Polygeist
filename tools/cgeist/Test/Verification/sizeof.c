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
// CHECK-NEXT:     %0 = "polygeist.typeSize"() {source = !llvm.struct<(memref<?xf32>, i8)>} : () -> index
// CHECK-NEXT:     %1 = arith.divui %0, %0 : index
// CHECK-NEXT:     %2 = memref.alloc(%1) : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:     return %2 : memref<?x!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:   }
