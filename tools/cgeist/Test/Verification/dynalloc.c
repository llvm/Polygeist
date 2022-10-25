// RUN: cgeist %s --function=create_matrix -S | FileCheck %s

void create_matrix(float *m, int size) {
  float coe[2 * size + 1];
  coe[size] = 1.0;
  m[size] = coe[size] + coe[0];
}

// CHECK:   func @create_matrix(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: i32)
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[cst:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %[[i3:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:     %[[i4:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V2:.+]] = arith.cmpi eq, %{{.*}}, %[[c0]] : index
// CHECK-NEXT:     scf.if %[[V2]] {
// CHECK-NEXT:       affine.store %[[cst]], %[[i3]][] : memref<f32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[i5:.+]] = affine.load %[[i3]][] : memref<f32>
// CHECK-NEXT:     %[[i6:.+]] = arith.addf %[[i5]], %[[cst]] : f32
// CHECK-NEXT:     affine.store %[[i6]], %[[arg0]][symbol(%[[i4]])] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
