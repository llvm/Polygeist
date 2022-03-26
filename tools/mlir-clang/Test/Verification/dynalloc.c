// RUN: mlir-clang %s --function=create_matrix -S | FileCheck %s

void create_matrix(float *m, int size) {
  float coe[2 * size + 1];
  coe[size] = 1.0;
  m[size] = coe[size] + coe[0];
}

// CHECK:   func @create_matrix(%arg0: memref<?xf32>, %arg1: i32)
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.muli %0, %c2 : index
// CHECK-NEXT:     %2 = arith.addi %1, %c1 : index
// CHECK-NEXT:     %3 = memref.alloca(%2) : memref<?xf32>
// CHECK-NEXT:     affine.store %cst, %3[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     %[[i5:.+]] = affine.load %3[0] : memref<?xf32>
// CHECK-NEXT:     %[[i6:.+]] = arith.addf %[[i5]], %cst : f32
// CHECK-NEXT:     affine.store %[[i6]], %arg0[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
