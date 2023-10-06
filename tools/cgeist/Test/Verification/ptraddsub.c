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


// CHECK-LABEL:   func.func @sub() -> i32  
// CHECK-DAG:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 4 : i64
// CHECK-DAG:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<10xi32>
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<10xi32>) -> !llvm.ptr
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][28] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-DAG:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK-DAG:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr to i64
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = arith.subi %[[VAL_4]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = arith.divsi %[[VAL_6]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = arith.trunci %[[VAL_7]] : i64 to i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @add(
// CHECK-SAME:                   %[[VAL_0:[A-Za-z0-9_]*]]: memref<?xi32>) -> memref<?xi32>  
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 7 : index
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.subindex"(%[[VAL_0]], %[[VAL_1]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK:           return %[[VAL_2]] : memref<?xi32>
// CHECK:         }

