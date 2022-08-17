// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

#set = affine_set<(d0) : (-d0 + 31 >= 0)>
module {
  func.func @_Z5randuPii(%714: memref<512xf64>, %703: memref<512xf64>) {
        affine.parallel (%arg8) = (0) to (512) {
          %716 = affine.load %714[%arg8] : memref<512xf64>
          affine.if #set(%arg8) {
            %717 = affine.load %703[%arg8 + 1] : memref<512xf64>
            %718 = arith.addf %716, %717 : f64
            affine.store %718, %703[%arg8] : memref<512xf64>
          }
        }
    return
  }
}

// CHECK:   func.func @_Z5randuPii(%arg0: memref<512xf64>, %arg1: memref<512xf64>) {
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to (32) {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg2] : memref<512xf64>
// CHECK-NEXT:       %1 = affine.load %arg1[%arg2 + 1] : memref<512xf64>
// CHECK-NEXT:       %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:       affine.store %2, %arg1[%arg2] : memref<512xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
