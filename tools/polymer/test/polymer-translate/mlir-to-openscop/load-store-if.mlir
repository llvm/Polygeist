// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

// Consider if operations in the domain.
// We will make this test valid when the diff D86421 is landed.

#set = affine_set<(d0, d1): (d0 - 16 >= 0, d1 - 16 >= 0, d1 - d0 >= 0)>

func @load_store_if(%A : memref<32x32xf32>, %B : memref<32x32xf32>) -> () {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.if #set(%i, %j) {
        %0 = affine.load %A[%i, %j] : memref<32x32xf32>
        affine.store %0, %A[%i, %j] : memref<32x32xf32>
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
// CHECK: 4
//
// CHECK: # ----------------------------------------------  1.1 Domain
// CHECK: DOMAIN
// CHECK: 7 4 2 0 0 0
// CHECK: # e/i| i0   i1 |  1  
// CHECK:    1    1    0    0    ## i0 >= 0
// CHECK:    1   -1    0   31    ## -i0+31 >= 0
// CHECK:    1    0    1    0    ## i1 >= 0
// CHECK:    1    0   -1   31    ## -i1+31 >= 0
// CHECK:    1    1    0  -16    ## i0-16 >= 0
// CHECK:    1    0    1  -16    ## i1-16 >= 0
// CHECK:    1   -1    1    0    ## -i0+i1 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 5 9 5 2 0 0
// CHECK: # e/i| c1   c2   c3   c4   c5 | i0   i1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    0    0    1    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    0    ## c3 == 0
// CHECK:    0    0    0    0   -1    0    0    1    0    ## c4 == i1
// CHECK:    0    0    0    0    0   -1    0    0    0    ## c5 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: WRITE
// CHECK: 3 7 3 2 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1 |  1  
// CHECK:    0   -1    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    ## [2] == i1
//
// CHECK: READ
// CHECK: 3 7 3 2 0 0
// CHECK: # e/i| Arr  [1]  [2]| i0   i1 |  1  
// CHECK:    0   -1    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    ## [2] == i1
//
// CHECK: # ----------------------------------------------  1.4 Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 1
// CHECK: <body>
// CHECK: # Number of original iterators
// CHECK: 2
// CHECK: # List of original iterators
// CHECK: i0 i1
// CHECK: # Statement body expression
// CHECK: S0(A1, 2, A1, 2, i0, i1)
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
// CHECK: c0 i0 c2 i1 c4
// CHECK: </scatnames>
//
// CHECK: </OpenScop>
