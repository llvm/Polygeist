// RUN: polymer-translate %s -export-scop | FileCheck %s

func @matmul() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}


// CHECK: <OpenScop>
//
// CHECK: # =============================================== Global
// CHECK: # Language
// CHECK: C
//
// CHECK: # Context
// CHECK: CONTEXT
// CHECK: 0 2 0 0 0 0
//
// CHECK: # Parameters are not provided
// CHECK: 0
//
//
// CHECK: # Number of statements
// CHECK: 1
//
// CHECK: # =============================================== Statement 1
// CHECK: # Number of relations describing the statement:
// CHECK: 6
//
// CHECK: # ----------------------------------------------  1.1 Domain
// CHECK: DOMAIN
// CHECK: 6 5 3 0 0 0
// CHECK: # e/i| i0   i1   i2 |  1  
// CHECK:    1    1    0    0    0    ## i0 >= 0
// CHECK:    1   -1    0    0   63    ## -i0+63 >= 0
// CHECK:    1    0    1    0    0    ## i1 >= 0
// CHECK:    1    0   -1    0   63    ## -i1+63 >= 0
// CHECK:    1    0    0    1    0    ## i2 >= 0
// CHECK:    1    0    0   -1   63    ## -i2+63 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 7 12 7 3 0 0
// CHECK: # e/i| c1   c2   c3   c4   c5   c6   c7 | i0   i1   i2 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    0    0    0    0    1    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    0    0    0    0    ## c3 == 0
// CHECK:    0    0    0    0   -1    0    0    0    0    1    0    0    ## c4 == i1
// CHECK:    0    0    0    0    0   -1    0    0    0    0    0    0    ## c5 == 0
// CHECK:    0    0    0    0    0    0   -1    0    0    0    1    0    ## c6 == i2
// CHECK:    0    0    0    0    0    0    0   -1    0    0    0    0    ## c7 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: WRITE
// CHECK: 3 8 3 3 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1   i2 |  1  
// CHECK:    0   -1    0    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    0    ## [2] == i1
//
// CHECK: READ
// CHECK: 3 8 3 3 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1   i2 |  1  
// CHECK:    0   -1    0    0    0    0    0    2    ## Arr == A2
// CHECK:    0    0   -1    0    1    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    0    1    0    ## [2] == i2
//
// CHECK: READ
// CHECK: 3 8 3 3 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1   i2 |  1  
// CHECK:    0   -1    0    0    0    0    0    3    ## Arr == A3
// CHECK:    0    0   -1    0    0    0    1    0    ## [1] == i2
// CHECK:    0    0    0   -1    0    1    0    0    ## [2] == i1
//
// CHECK: READ
// CHECK: 3 8 3 3 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1   i2 |  1  
// CHECK:    0   -1    0    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    0    ## [2] == i1
//
// CHECK: # ----------------------------------------------  1.4 Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 1
// CHECK: <body>
// CHECK: # Number of original iterators
// CHECK: 3
// CHECK: # List of original iterators
// CHECK: i0 i1 i2
// CHECK: # Statement body expression
// CHECK: S0(A1, 2, A2, 2, A3, 2, A1, 2, i0, i1, i2)
// CHECK: </body>

// Won't check the extension generation for this case.
// # =============================================== Extensions
// <arrays>
// # Number of arrays
// 3
// # Mapping array-identifiers/array-names
// 2 A2
// 3 A3
// 1 A1
// </arrays>
//
// <scatnames>
// c0 i0 c2 i1 c4 i2 c6
// </scatnames>
//
// </OpenScop>
