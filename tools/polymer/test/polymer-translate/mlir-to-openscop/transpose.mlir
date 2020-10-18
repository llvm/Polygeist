// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

func @transpose(%A : memref<?x?xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %A, %c0 : memref<?x?xf32>
  %M = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %0 = affine.load %A[%i, %j] : memref<?x?xf32>
      affine.store %0, %A[%j, %i] : memref<?x?xf32>
    }
  }

  return
}


// func @triangular_transpose(%A : memref<?x?xf32>) -> () {
//   %c0 = constant 0 : index
//   %c1 = constant 1 : index
//   %N = dim %A, %c0 : memref<?x?xf32>
//   %M = dim %A, %c1 : memref<?x?xf32>

//   affine.for %i = 0 to %N {
//     affine.for %j = 0 to %M {
//       affine.if affine_set<(d0, d1): (d1 - d0 >= 0)>(%j, %i) {
//         %1 = affine.load %A[%i, %j] : memref<?x?xf32>
//         affine.store %1, %A[%j, %i] : memref<?x?xf32>
//       }
//     }
//   }

//   return
// }

// CHECK: <OpenScop>
//
// CHECK: # =============================================== Global
// CHECK: # Language
// CHECK: C
//
// CHECK: # Context
// CHECK: CONTEXT
// CHECK: 2 4 0 0 0 2
// CHECK: # e/i| P0   P1 |  1  
// CHECK:    1    1    0   -1    ## P0-1 >= 0
// CHECK:    1    0    1   -1    ## P1-1 >= 0
//
// CHECK: # Parameters are provided
// CHECK: 1
// CHECK: <strings>
// CHECK: P0 P1
// CHECK: </strings>
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
// CHECK: 4 6 2 0 0 2
// CHECK: # e/i| i0   i1 | P0   P1 |  1  
// CHECK:    1    1    0    0    0    0    ## i0 >= 0
// CHECK:    1   -1    0    1    0   -1    ## -i0+P0-1 >= 0
// CHECK:    1    0    1    0    0    0    ## i1 >= 0
// CHECK:    1    0   -1    0    1   -1    ## -i1+P1-1 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 5 11 5 2 0 2
// CHECK: # e/i| c1   c2   c3   c4   c5 | i0   i1 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    0    0    1    0    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    0    0    0    ## c3 == 0
// CHECK:    0    0    0    0   -1    0    0    1    0    0    0    ## c4 == i1
// CHECK:    0    0    0    0    0   -1    0    0    0    0    0    ## c5 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: WRITE
// CHECK: 3 9 3 2 0 2
// CHECK: # e/i| Arr  [1]  [2]| i0   i1 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    0    0    ## [2] == i1
//
// CHECK: READ
// CHECK: 3 9 3 2 0 2
// CHECK: # e/i| Arr  [1]  [2]| i0   i1 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    0    1    0    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    0    0    ## [2] == i1
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
// CHECK: c0 i0 c2 i1
// CHECK: </scatnames>
//
// CHECK: </OpenScop>
