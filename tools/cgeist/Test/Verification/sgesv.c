// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s
// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int n, double alpha, double beta,
                    double A[28][28],
                    double B[28][28],
                    double tmp[28],
                    double x[28],
                    double y[28])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < n; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }

}

// CHECK:   func @kernel_correlation(%[[arg0:.+]]: i32, %[[arg1:.+]]: f64, %[[arg2:.+]]: f64, %[[arg3:.+]]: memref<?x28xf64>, %[[arg4:.+]]: memref<?x28xf64>, %[[arg5:.+]]: memref<?xf64>, %[[arg6:.+]]: memref<?xf64>, %[[arg7:.+]]: memref<?xf64>)
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg0]] : i32 to index
// CHECK-NEXT:     affine.for %[[arg8:.+]] = 0 to %[[V0]] {
// CHECK-NEXT:       affine.store %[[cst]], %[[arg5]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       affine.store %[[cst]], %[[arg7]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       affine.for %[[arg9:.+]] = 0 to %[[V0]] {
// CHECK-NEXT:         %[[V6:.+]] = affine.load %[[arg3]][%[[arg8]], %[[arg9]]] : memref<?x28xf64>
// CHECK-NEXT:         %[[V7:.+]] = affine.load %[[arg6]][%[[arg9]]] : memref<?xf64>
// CHECK-NEXT:         %[[V8:.+]] = arith.mulf %[[V6]], %[[V7]] : f64
// CHECK-NEXT:         %[[V9:.+]] = affine.load %[[arg5]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:         %[[V10:.+]] = arith.addf %[[V8]], %[[V9]] : f64
// CHECK-NEXT:         affine.store %[[V10]], %[[arg5]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:         %[[V11:.+]] = affine.load %[[arg4]][%[[arg8]], %[[arg9]]] : memref<?x28xf64>
// CHECK-NEXT:         %[[V12:.+]] = affine.load %[[arg6]][%[[arg9]]] : memref<?xf64>
// CHECK-NEXT:         %[[V13:.+]] = arith.mulf %[[V11]], %[[V12]] : f64
// CHECK-NEXT:         %[[V14:.+]] = affine.load %[[arg7]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:         %[[V15:.+]] = arith.addf %[[V13]], %[[V14]] : f64
// CHECK-NEXT:         affine.store %[[V15]], %[[arg7]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[V1:.+]] = affine.load %[[arg5]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       %[[V2:.+]] = arith.mulf %[[arg1]], %[[V1]] : f64
// CHECK-NEXT:       %[[V3:.+]] = affine.load %[[arg7]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       %[[V4:.+]] = arith.mulf %[[arg2]], %[[V3]] : f64
// CHECK-NEXT:       %[[V5:.+]] = arith.addf %[[V2]], %[[V4]] : f64
// CHECK-NEXT:       affine.store %[[V5]], %[[arg7]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK: func @kernel_correlation(%{{.*}}: i32, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>)
