// RUN: mlir-clang %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int m, double corr[28])
{
  for (int i = 0; i < m-1; i++) {
    corr[i] = 0.;
  }
}

// CHECK:   func @kernel_correlation(%arg0: i32, %arg1: memref<?xf64>)
// CHECK-NEXT:      %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:      %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      affine.for %arg2 = 0 to #map()[%0] {
// CHECK-NEXT:        affine.store %cst, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
