// RUN: mlir-clang %s --function=* -S | FileCheck %s

void* calloc(unsigned long a, unsigned long b);

float* zmem(int n) {
    float* out = (float*)calloc(sizeof(float), n);
    return out;
}

// CHECK:   func @zmem(%arg0: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.extui %arg0 : i32 to i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = arith.muli %1, %c4 : index
// CHECK-NEXT:     %3 = arith.divui %2, %c4 : index
// CHECK-NEXT:     %4 = memref.alloc(%3) : memref<?xf32>
// CHECK-NEXT:     scf.for %arg1 = %c0 to %3 step %c1 {
// CHECK-NEXT:       memref.store %cst, %4[%arg1] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %4 : memref<?xf32>
// CHECK-NEXT:   }
