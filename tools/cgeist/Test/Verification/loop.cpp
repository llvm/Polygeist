// RUN: cgeist %s --function=* -S | FileCheck %s

int MAX_DIMS;

struct A {
    int x;
    double y;
};

void div_(int* sizes) {
    A data[25];
    for (int i=0; i < MAX_DIMS; ++i) {
            data[i].x = sizes[i];
    }
}

// CHECK:   func @_Z4div_Pi(%[[arg0:.+]]: memref<?xi32>) attributes {llvm.linkage =
// #llvm.linkage<external>} { CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<25x!llvm.struct<(i32, f64)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.get_global @MAX_DIMS : memref<1xi32>
// CHECK-NEXT:     %[[V2:.+]] = affine.load %[[V1]][0] : memref<1xi32>
// CHECK-NEXT:     %[[V3:.+]] = "polygeist.memref2pointer"(%[[V0]]) :
// (memref<25x!llvm.struct<(i32, f64)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %[[V4:.+]] = arith.index_cast %[[V2]] : i32 to index
// CHECK-NEXT:     scf.for %[[arg1:.+]] = %[[c0]] to %[[V4]] step %[[c1]] {
// CHECK-NEXT:       %[[V5:.+]] = arith.index_cast %[[arg1]] : index to i64
// CHECK-NEXT:       %[[V6:.+]] = llvm.getelementptr %[[V3]][%[[V5]]] : (!llvm.ptr<struct<(i32,
// f64)>>, i64) -> !llvm.ptr<struct<(i32, f64)>> CHECK-NEXT:       %[[V7:.+]] =
// llvm.getelementptr %[[V6]][0, 0] : (!llvm.ptr<struct<(i32, f64)>>) ->
// !llvm.ptr<i32> CHECK-NEXT:       %[[V8:.+]] = memref.load %[[arg0]][%[[arg1]]] :
// memref<?xi32> CHECK-NEXT:       llvm.store %[[V8]], %[[V7]] : !llvm.ptr<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
