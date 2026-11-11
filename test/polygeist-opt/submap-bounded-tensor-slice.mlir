// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s
// A logical subview must retain its own domain when the base is larger.
// CHECK-LABEL: func.func @translated_prefix
// CHECK: tensor.extract_slice {{.*}}[0, 5, 0] [4, 5, 5] [1, 1, 1]
// CHECK-NOT: linalg.generic
func.func @translated_prefix(%base: tensor<4x10x5xf64>) -> tensor<4x5x5xf64> {
  %four = arith.constant 4 : index
  %five = arith.constant 5 : index
  %view = polygeist.submap(%base, %four, %five, %five) {map=affine_map<(i,j,k)->(i,j+5,k)>} : (tensor<4x10x5xf64>,index,index,index)->tensor<4x5x5xf64>
  return %view : tensor<4x5x5xf64>
}
