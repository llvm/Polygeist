// RUN: polygeist-opt --split-input-file --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnPointwiseAffineRelu_f32(
      %x: tensor<?xf32>, %bias: tensor<?xf32>, %out: tensor<?xf32>,
      %alpha: f32) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @affine_relu(%x: tensor<?xf32>, %bias: tensor<?xf32>,
                         %out: tensor<?xf32>, %alpha: f32)
      -> tensor<?xf32> {
    %r = kernel.launch @cudnnPointwiseAffineRelu_f32(
        %x, %bias, %out, %alpha)
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, f32)
          -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @affine_relu
// CHECK: %[[N:.*]] = arith.index_cast {{.*}} : index to i32
// CHECK: call @polygeist_cudnn_pointwise_affine_relu_f32(
// CHECK-SAME: %[[N]], %arg3,
// CHECK-SAME: (i32, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cudnnPointwiseGraph_f32(
      %in0: tensor<?xf32>, %in1: tensor<?xf32>,
      %in2: tensor<?xf32>, %in3: tensor<?xf32>,
      %out: tensor<?xf32>,
      %s0: f32, %s1: f32, %s2: f32, %s3: f32,
      %s4: f32, %s5: f32, %s6: f32, %s7: f32) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @generic_graph(
      %x: tensor<?xf32>, %y: tensor<?xf32>, %out: tensor<?xf32>,
      %s0: f32, %s1: f32, %s2: f32, %s3: f32) -> tensor<?xf32> {
    %r = kernel.launch @cudnnPointwiseGraph_f32(
        %x, %y, %x, %x, %out, %s0, %s1, %s2, %s3,
        %s0, %s1, %s2, %s3)
        {pointwise_graph = array<i64: 1334580892173348865, 0, 0, 0, 0, 0, 0, 0>,
         pointwise_num_nodes = 4 : i64}
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           tensor<?xf32>, f32, f32, f32, f32,
           f32, f32, f32, f32) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @generic_graph
// CHECK: %[[LO:.*]] = arith.constant 1334580892173348865 : i64
// CHECK: %[[NODES:.*]] = arith.constant 4 : i32
// CHECK: call @polygeist_cudnn_pointwise_graph_f32(
// CHECK-SAME: i32, i64, i64, i64, i64, i64, i64, i64, i64,
// CHECK-SAME: i64, i64, i64, i64, i32,
// CHECK-SAME: f32, f32, f32, f32, f32, f32, f32, f32,
// CHECK-SAME: i32, i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
// CHECK-NOT: kernel.launch
