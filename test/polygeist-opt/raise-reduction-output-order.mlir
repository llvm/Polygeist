// RUN: polygeist-opt %s --raise-affine-to-linalg-pipeline | FileCheck %s
// Every output contributes to dependence classification, regardless of order.
// CHECK-LABEL: func.func @carry_first
// CHECK: iterator_types = ["reduction"]
// CHECK-LABEL: func.func @carry_last
// CHECK: iterator_types = ["reduction"]
// CHECK-LABEL: func.func @independent
// CHECK: iterator_types = ["parallel"]
module {
  func.func @carry_first(%x: memref<8xf32>, %y: memref<8xf32>, %state: memref<f32>) {
    affine.for %i = 0 to 8 {
      %a = affine.load %x[%i] : memref<8xf32>
      %b = affine.load %state[] : memref<f32>
      %c = arith.subf %a, %b : f32
      affine.store %c, %state[] : memref<f32>
      affine.store %c, %y[%i] : memref<8xf32>
    }
    return
  }
  func.func @carry_last(%x: memref<8xf32>, %y: memref<8xf32>, %state: memref<f32>) {
    affine.for %i = 0 to 8 {
      %a = affine.load %x[%i] : memref<8xf32>
      %b = affine.load %state[] : memref<f32>
      %c = arith.subf %a, %b : f32
      affine.store %c, %y[%i] : memref<8xf32>
      affine.store %c, %state[] : memref<f32>
    }
    return
  }
  func.func @independent(%x: memref<8xf32>, %y: memref<8xf32>, %z: memref<8xf32>) {
    affine.for %i = 0 to 8 {
      %a = affine.load %x[%i] : memref<8xf32>
      affine.store %a, %y[%i] : memref<8xf32>
      affine.store %a, %z[%i] : memref<8xf32>
    }
    return
  }
}
