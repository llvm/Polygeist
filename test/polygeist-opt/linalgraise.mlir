// RUN: polygeist-opt --raise-affine-to-linalg --split-input-file %s | FileCheck %s

module {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    scf.if %12 {
      affine.for %arg4 = 0 to %17 {
        %ld = affine.load %18[%arg4] : memref<?xf32>
        affine.store %ld, %19[%arg4] : memref<?xf32>
      }
   }
    return
  }
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK:   func.func @main(%[[arg0:.+]]: i1, %[[arg1:.+]]: i32, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: memref<?xf32>) {
// CHECK-NEXT:     %[[c4:.+]] = arith.constant 4 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.muli %[[V0]], %[[c4]] : index
// CHECK-NEXT:     %[[V2:.+]] = arith.divui %[[V1]], %[[c4]] : index
// CHECK-NEXT:     scf.if %[[arg0]] {
// TODO note that presently we do not ensure that the memrefs are sliced to the right size as the space requires
// CHECK-NEXT:        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg2 : memref<?xf32>) outs(%alloca : memref<?xf32>) {
// CHECK-NEXT:        ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:          linalg.yield %in : f32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:     }
