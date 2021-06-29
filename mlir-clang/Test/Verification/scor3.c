// RUN: mlir-clang %s --function=kernel_correlation --raise-scf-to-affine | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int m, double corr[28][28])
{
    int i, j, k;
    //i = 0;
    for (i = 0; i < 28; i++)
    {
      for (j = i+1; j < m; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
        }
    }
}

// CHECK:   func @kernel_correlation(%arg0: i32, %arg1: memref<?x28xf64>) {
// CHECK-DAG:     %cst = constant 0.000000e+00 : f64
// CHECK-DAG:     %c1_i32 = constant 1 : i32
// CHECK-DAG:     %c2_i32 = constant 2 : i32
// CHECK-DAG:     %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg2 = 0 to 28 {
// CHECK-NEXT:       %1 = index_cast %arg2 : index to i32
// CHECK-NEXT:       %2 = addi %1, %c1_i32 : i32
// CHECK-NEXT:       %3 = affine.for %arg3 = #map(%arg2) to %0 iter_args(%arg4 = %2) -> (i32) {
// CHECK-NEXT:         affine.store %cst, %arg1[%arg2, %arg2 + 1] : memref<?x28xf64>
// CHECK-NEXT:         %4 = addi %1, %c2_i32 : i32
// CHECK-NEXT:         affine.yield %4 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }