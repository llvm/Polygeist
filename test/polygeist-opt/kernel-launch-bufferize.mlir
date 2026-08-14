// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse %s | FileCheck %s --check-prefix=BUFFERIZE
// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse --lower-kernel-launch-to-cublas %s | FileCheck %s --check-prefix=LOWER

module {
  kernel.defn @cudnnPointwiseGraph_f32(
      %in0: tensor<?xf32>, %in1: tensor<?xf32>,
      %in2: tensor<?xf32>, %in3: tensor<?xf32>,
      %out: tensor<?xf32>,
      %s0: f32, %s1: f32, %s2: f32, %s3: f32,
      %s4: f32, %s5: f32, %s6: f32, %s7: f32) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @gradient_boundary_and_interior(
      %x: memref<?xf32>, %out: memref<?xf32>, %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c126 = arith.constant 126 : index
    %c127 = arith.constant 127 : index
    %zero = arith.constant 0.0 : f32
    %xt = bufferization.to_tensor %x restrict writable : memref<?xf32>
    %outt = bufferization.to_tensor %out restrict writable : memref<?xf32>
    %left = tensor.insert %scale into %outt[%c0] : tensor<?xf32>
    %xp = tensor.extract_slice %xt[%c2] [%c126] [1]
        : tensor<?xf32> to tensor<?xf32>
    %xm = tensor.extract_slice %xt[%c0] [%c126] [1]
        : tensor<?xf32> to tensor<?xf32>
    %interior = tensor.extract_slice %left[%c1] [%c126] [1]
        : tensor<?xf32> to tensor<?xf32>
    %graph = kernel.launch @cudnnPointwiseGraph_f32(
        %xp, %xm, %xp, %xp, %interior,
        %scale, %zero, %zero, %zero, %zero, %zero, %zero, %zero)
        {pointwise_graph = array<i64: 1107570689, 0, 0, 0, 0, 0, 0, 0>,
         pointwise_num_nodes = 2 : i64}
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           tensor<?xf32>, f32, f32, f32, f32,
           f32, f32, f32, f32) -> tensor<?xf32>
    %whole = tensor.insert_slice %graph into %left[%c1] [%c126] [1]
        : tensor<?xf32> into tensor<?xf32>
    %right = tensor.insert %scale into %whole[%c127] : tensor<?xf32>
    %result = bufferization.to_memref %right : memref<?xf32>
    memref.copy %result, %out : memref<?xf32> to memref<?xf32>
    return
  }
}

// BUFFERIZE-LABEL: func.func @gradient_boundary_and_interior
// BUFFERIZE: memref.store %arg2, %arg1[%c0]
// BUFFERIZE: %[[OUT_VIEW:.*]] = memref.subview %arg1[1] [126] [1]
// BUFFERIZE: %[[OUT_CAST:.*]] = memref.cast %[[OUT_VIEW]]
// BUFFERIZE: kernel.launch @cudnnPointwiseGraph_f32(
// BUFFERIZE-SAME: %[[OUT_CAST]], %arg2
// BUFFERIZE-SAME: polygeist.bufferized
// BUFFERIZE-SAME: polygeist.result_destinations = array<i64: 4>
// BUFFERIZE: memref.copy %[[OUT_VIEW]], %[[OUT_VIEW]]
// BUFFERIZE: memref.store %arg2, %arg1[%c127]

// LOWER-LABEL: func.func @gradient_boundary_and_interior
// LOWER: memref.store %arg2, %arg1[%c0]
// LOWER: call @polygeist_cudnn_pointwise_graph_f32(
// LOWER-NOT: memref.copy
// LOWER: memref.store %arg2, %arg1[%c127]
// LOWER-NOT: kernel.launch
