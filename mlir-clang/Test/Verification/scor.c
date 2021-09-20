// RUN: mlir-clang %s --function=kernel_correlation -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(double out[28], double stddev[28], _Bool cmp)
{
  int j;


#pragma scop

   for (j = 0; j < 28; j++)
    {
      stddev[j] = 0.0;
      stddev[j] = 3.14;
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
    
      out[j] = cmp ? 1.0 : stddev[j];
    }

#pragma endscop

}

// CHECK:   func @kernel_correlation(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i1)
// CHECK-DAG:     %[[cst:.+]] = constant 1.000000e+00 : f64
// CHECK-DAG:     %[[cst_0:.+]] = constant 3.140000e+00 : f64
// CHECK:     %0 = select %arg2, %[[cst]], %[[cst_0]] : f64
// CHECK-NEXT:     affine.for %arg3 = 0 to 28 {
// CHECK-NEXT:       affine.store %[[cst_0]], %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
