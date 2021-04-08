// RUN: mlir-opt %s -detect-reduction | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module  {
  func @main() {
    %0 = memref.alloc() : memref<40x40xf64>
    %1 = memref.alloc() : memref<40xf64>
    %2 = memref.alloc() : memref<40xf64>
    %3 = memref.alloc() : memref<40xf64>
    %4 = memref.alloc() : memref<1xf64>
    %5 = affine.load %1[0] : memref<40xf64>
    affine.store %5, %4[0] : memref<1xf64>
    %6 = affine.load %4[0] : memref<1xf64>
    affine.store %6, %3[0] : memref<40xf64>
    // CHECK: affine.for
    affine.for %arg0 = 1 to 40 {
      // CHECK-NEXT: affine.load
      %11 = affine.load %1[%arg0] : memref<40xf64>
      // CHECK-NEXT: affine.store
      affine.store %11, %4[0] : memref<1xf64>
      // CHECK-NEXT: %[[C:.*]] = affine.load {{.*}} : memref<1xf64>
      // CHECK-NEXT: affine.for {{.*}} = 0 to {{.*}} iter_args({{.*}} = %[[C]]) -> (f64) {
      affine.for %arg1 = 0 to #map(%arg0) {
        %13 = affine.load %4[0] : memref<1xf64>
        %14 = affine.load %0[%arg0, %arg1] : memref<40x40xf64>
        %15 = affine.load %3[%arg1] : memref<40xf64>
        %16 = mulf %14, %15 : f64
        %17 = subf %13, %16 : f64
        affine.store %17, %4[0] : memref<1xf64>
      }
      // CHECK: }
      // CHECK-NEXT: affine.store {{.*}}, {{.*}} : memref<1xf64>
      %12 = affine.load %4[0] : memref<1xf64>
      affine.store %12, %3[%arg0] : memref<40xf64>
    }
    %7 = affine.load %3[39] : memref<40xf64>
    affine.store %7, %4[0] : memref<1xf64>
    %8 = affine.load %4[0] : memref<1xf64>
    %9 = affine.load %0[39, 39] : memref<40x40xf64>
    %10 = divf %8, %9 : f64
    affine.store %10, %2[39] : memref<40xf64>
    return
  }
}
