// RUN: mlir-clang %s --function=* -S | FileCheck %s

void* calloc(unsigned long a, unsigned long b);

float* zmem(int n) {
    float* out = (float*)calloc(sizeof(float), n);
    return out;
}

// CHECK:   func @zmem(%arg0: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c4_i64 = arith.constant 4 : i64
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c0_i8 = arith.constant 0 : i8
// CHECK-DAG:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %0 = arith.extui %arg0 : i32 to i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = arith.muli %1, %c4 : index
// CHECK-NEXT:     %3 = arith.divui %2, %c4 : index
// CHECK-NEXT:     %4 = memref.alloc(%3) : memref<?xf32>
// CHECK-NEXT:     %5 = "polygeist.memref2pointer"(%4) : (memref<?xf32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %6 = arith.muli %0, %c4_i64 : i64
// CHECK-NEXT:     "llvm.intr.memset"(%5, %c0_i8, %6, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// TODO-NEXT:     scf.for %arg1 = %c0 to %3 step %c1 {
// TODO-NEXT:       memref.store %cst, %4[%arg1] : memref<?xf32>
// TODO-NEXT:     }
// CHECK-NEXT:     return %4 : memref<?xf32>
// CHECK-NEXT:   }
