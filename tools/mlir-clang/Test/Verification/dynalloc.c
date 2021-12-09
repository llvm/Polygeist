// RUN: mlir-clang %s --function=create_matrix -S | FileCheck %s

void create_matrix(float *m, int size) {
  float coe[2 * size - 1];
  coe[size] = 0;
  m[size] = coe[size];
}

// CHECK:   func @create_matrix(%arg0: memref<?xf32>, %arg1: i32)
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.muli %0, %c2 : index
// CHECK-NEXT:     %2 = arith.subi %1, %c1 : index
// CHECK-NEXT:     %3 = memref.alloca(%2) : memref<?xf32>
// CHECK-NEXT:     %4 = arith.sitofp %c0_i32 : i32 to f32
// CHECK-NEXT:     affine.store %4, %3[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     %5 = affine.load %3[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     affine.store %5, %arg0[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
