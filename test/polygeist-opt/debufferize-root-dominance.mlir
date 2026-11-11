// RUN: polygeist-opt %s --linalg-debufferize=use-multi-root=true | FileCheck %s
// An argument root occurs after a later-defined view root in argument order.
// Its snapshot must still precede the earlier use.
// CHECK-LABEL: func.func @later_view_root
// CHECK: bufferization.to_tensor %arg1
// CHECK: linalg.generic
// CHECK: memref.reinterpret_cast %arg0
func.func @later_view_root(%arg0: memref<?xf64>, %arg1: memref<4xf64>) {
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>], iterator_types = ["parallel"]} outs(%arg1 : memref<4xf64>) {
  ^bb0(%x: f64):
    linalg.yield %zero : f64
  }
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xf64> to memref<2x2xf64>
  linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} outs(%view : memref<2x2xf64>) {
  ^bb0(%x: f64):
    linalg.yield %zero : f64
  }
  return
}
