// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasBroadcastAxis0_f32(
      %x: tensor<?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }
  kernel.defn @cublasBroadcastAxis1_f32(
      %x: tensor<?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }
  func.func @axis0(%x: tensor<?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %r = kernel.launch @cublasBroadcastAxis0_f32(%x, %out)
        : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
  func.func @axis1(%x: tensor<?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %r = kernel.launch @cublasBroadcastAxis1_f32(%x, %out)
        : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @axis0
// CHECK: %[[AXIS0:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cublas_broadcast_1d_to_2d_f32(%[[AXIS0]],
// CHECK-LABEL: func.func @axis1
// CHECK: %[[AXIS1:.*]] = arith.constant 1 : i32
// CHECK: call @polygeist_cublas_broadcast_1d_to_2d_f32(%[[AXIS1]],
// CHECK-NOT: kernel.launch
