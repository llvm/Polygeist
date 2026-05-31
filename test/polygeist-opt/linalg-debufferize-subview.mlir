// RUN: polygeist-opt --linalg-debufferize %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

module {
  func.func @subview_after_cross_root(%a: memref<4xf32>, %b: memref<4xf32>,
                                      %out: memref<4xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %acc = memref.alloca() : memref<f32>
    affine.store %cst, %acc[] : memref<f32>
    linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    } ins(%a, %b : memref<4xf32>, memref<4xf32>)
      outs(%out : memref<4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %old: f32):
      %sum = arith.addf %in0, %in1 : f32
      linalg.yield %sum : f32
    }
    %tail = memref.subview %out[1] [3] [1]
      : memref<4xf32> to memref<3xf32, strided<[1], offset: 1>>
    linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["reduction"]
    } ins(%tail : memref<3xf32, strided<[1], offset: 1>>)
      outs(%acc : memref<f32>) {
    ^bb0(%in: f32, %old: f32):
      %sum = arith.addf %old, %in : f32
      linalg.yield %sum : f32
    }
    %res = affine.load %acc[] : memref<f32>
    return %res : f32
  }
}

// CHECK-LABEL: func.func @subview_after_cross_root
// CHECK: bufferization.to_tensor %arg2 : memref<4xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<4xf32>)
// CHECK: tensor.extract_slice %{{.*}}[1] [3] [1] : tensor<4xf32> to tensor<3xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<3xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<f32>)
// CHECK-NOT: memref.subview
