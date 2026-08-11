// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cutensorUnary_cos_f32(
      %x: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }
  kernel.defn @cutensorUnary_acos_f32(
      %x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @cos2d(%x: tensor<?x?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %r = kernel.launch @cutensorUnary_cos_f32(%x, %out)
        : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  func.func @acos1d(%x: tensor<?xf32>, %out: tensor<?xf32>)
      -> tensor<?xf32> {
    %r = kernel.launch @cutensorUnary_acos_f32(%x, %out)
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @cos2d
// CHECK: %[[OP:.*]] = arith.constant 8 : i32
// CHECK: %[[D0:.*]] = memref.dim
// CHECK: %[[D1:.*]] = memref.dim
// CHECK: %[[N:.*]] = arith.muli
// CHECK: call @polygeist_cutensor_unary_f32(%[[OP]], %[[N]],
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @acos1d
// CHECK: %[[OP_ACOS:.*]] = arith.constant 1 : i32
// CHECK: call @polygeist_cutensor_unary_f32(%[[OP_ACOS]],
// CHECK-NOT: kernel.launch
