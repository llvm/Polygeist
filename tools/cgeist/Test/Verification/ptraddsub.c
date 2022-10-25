// RUN: cgeist %s --function=* -S | FileCheck %s

int sub() {
    int data[10];
    int* start = &data[0];
    int* end = &data[7];
    return end - start;
}

int* add (int* in) {
	return &in[7];
}

// CHECK:   func @sub() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[c4_i64:.+]] = arith.constant 4 : i64
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<10xi32>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[V0]]) : (memref<10xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][7] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-DAG:     %[[i3:.+]] = llvm.ptrtoint %[[V1]] : !llvm.ptr<i32> to i64
// CHECK-DAG:     %[[i4:.+]] = llvm.ptrtoint %[[V2]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:     %[[V5:.+]] = arith.subi %[[i4]], %[[i3]] : i64
// CHECK-NEXT:     %[[V6:.+]] = arith.divsi %[[V5]], %[[c4_i64]] : i64
// CHECK-NEXT:     %[[V7:.+]] = arith.trunci %[[V6]] : i64 to i32
// CHECK-NEXT:     return %[[V7]] : i32
// CHECK-NEXT:   }

// CHECK:   func @add(%[[arg0:.+]]: memref<?xi32>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[c7:.+]] = arith.constant 7 : index
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.subindex"(%[[arg0]], %[[c7]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:     return %[[V0]] : memref<?xi32>
// CHECK-NEXT:   }
