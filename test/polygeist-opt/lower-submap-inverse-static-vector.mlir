// RUN: polygeist-opt --lower-polygeist-submap %s | FileCheck %s
module {
  func.func @full_vector(%base: tensor<4096xf32>, %updated: tensor<?xf32>)
      -> tensor<4096xf32> {
    %size = arith.constant 4096 : index
    %result = polygeist.submapInverse(%base, %updated, %size)
      {map = affine_map<(i)->(i)>} :
      (tensor<4096xf32>, tensor<?xf32>, index) -> tensor<4096xf32>
    return %result : tensor<4096xf32>
  }
  func.func @read_vector(%base: tensor<4096xf32>) -> tensor<?xf32> {
    %size = arith.constant 4096 : index
    %view = polygeist.submap(%base, %size)
      {map = affine_map<(i)->(i)>} :
      (tensor<4096xf32>, index) -> tensor<?xf32>
    return %view : tensor<?xf32>
  }
}
// CHECK-LABEL: func.func @full_vector
// CHECK: %[[UPDATED:.*]] = tensor.cast %arg1 : tensor<?xf32> to tensor<4096xf32>
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: polygeist.submapInverse
// CHECK: return %[[UPDATED]]
// CHECK-LABEL: func.func @read_vector
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[VIEW:.*]] = tensor.cast %arg0 : tensor<4096xf32> to tensor<?xf32>
// CHECK: return %[[VIEW]]
