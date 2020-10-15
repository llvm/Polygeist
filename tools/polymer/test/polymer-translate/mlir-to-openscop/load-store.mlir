// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

func @load_store() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    affine.store %0, %A[%i] : memref<32xf32>
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
// CHECK: 4
//
// CHECK: # ----------------------------------------------  1.1 Domain
// CHECK: DOMAIN
// CHECK: 2 3 1 0 0 0
// CHECK: # e/i| i0 |  1  
// CHECK:    1    1    0    ## i0 >= 0
// CHECK:    1   -1   31    ## -i0+31 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 6 3 1 0 0
// CHECK: # e/i| c1   c2   c3 | i0 |  1  
// CHECK:    0   -1    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    1    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    ## c3 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: WRITE
// CHECK: 2 5 2 1 0 0
// CHECK: # e/i| Arr  [1]| i0 |  1  
// CHECK:    0   -1    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    1    0    ## [1] == i0
//
// CHECK: READ
// CHECK: 2 5 2 1 0 0
// CHECK: # e/i| Arr  [1]| i0 |  1  
// CHECK:    0   -1    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    1    0    ## [1] == i0
//
// CHECK: # ----------------------------------------------  1.4 Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 1
// CHECK: <body>
// CHECK: # Number of original iterators
// CHECK: 1
// CHECK: # List of original iterators
// CHECK: i0
// CHECK: # Statement body expression
// CHECK: S0(A1, 1, A1, 1, i0)
// CHECK: </body>
//
// CHECK: # =============================================== Extensions
// CHECK: <arrays>
// CHECK: # Number of arrays
// CHECK: 1
// CHECK: # Mapping array-identifiers/array-names
// CHECK: 1 A1
// CHECK: </arrays>
//
// CHECK: <scatnames>
// CHECK: c0 i0 c2
// CHECK: </scatnames>
//
// CHECK: </OpenScop>
