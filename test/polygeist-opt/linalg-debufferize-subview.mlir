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

  // The reduction accumulator is allocated afresh inside the outer loop.
  // Debufferization must tensorize it locally rather than trying to pass its
  // tensor value as an affine.for init operand, where it would not dominate.
  func.func @loop_local_reduction_scratch(%a: memref<4x8xf32>,
                                           %out: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    affine.for %i = 0 to 4 {
      %scratch = memref.alloca() : memref<f32>
      affine.store %zero, %scratch[] : memref<f32>
      %row = memref.subview %a[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf32> to memref<8xf32, strided<[1], offset: ?>>
      linalg.generic {
        indexing_maps = [#map0, #map1],
        iterator_types = ["reduction"]
      } ins(%row : memref<8xf32, strided<[1], offset: ?>>)
        outs(%scratch : memref<f32>) {
      ^bb0(%in: f32, %old: f32):
        %sum = arith.addf %old, %in : f32
        linalg.yield %sum : f32
      }
      %value = affine.load %scratch[] : memref<f32>
      affine.store %value, %out[%i] : memref<4xf32>
    }
    return
  }

  // The temporary connects three distinct roots.  Converting roots one at a
  // time used to leave only the final copy and silently discard the fill and
  // reduction that produce the temporary.
  func.func @cross_root_temporary_chain(%a: memref<4x8xf32>,
                                         %out: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    %tmp = memref.alloca() : memref<4xf32>
    linalg.generic {
      indexing_maps = [#map0], iterator_types = ["parallel"]
    } outs(%tmp : memref<4xf32>) {
    ^bb0(%old: f32):
      linalg.yield %zero : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a : memref<4x8xf32>) outs(%tmp : memref<4xf32>) {
    ^bb0(%in: f32, %old: f32):
      %sum = arith.addf %old, %in : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      indexing_maps = [#map0, #map0], iterator_types = ["parallel"]
    } ins(%tmp : memref<4xf32>) outs(%out : memref<4xf32>) {
    ^bb0(%in: f32, %old: f32):
      linalg.yield %in : f32
    }
    return
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

// CHECK-LABEL: func.func @loop_local_reduction_scratch
// CHECK: affine.for
// CHECK-SAME: iter_args(%{{.*}} = %{{.*}})
// CHECK: bufferization.to_tensor %{{.*}} : memref<f32>
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : tensor<f32>)

// CHECK-LABEL: func.func @cross_root_temporary_chain
// CHECK-COUNT-3: linalg.generic
// CHECK-NOT: memref.alloca
