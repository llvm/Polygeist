// RUN: polygeist-opt --split-input-file --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnAdaptivePool_f32_flat2(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %input: memref<?xf32>, %output: memref<?xf32>) {
    kernel.yield
  }

  func.func @average_forward(
      %input: memref<?xf32>, %output: memref<?xf32>) {
    %op = arith.constant 0 : i32
    %rank = arith.constant 2 : i32
    %n = arith.constant 1 : i32
    %c = arith.constant 2 : i32
    %i0 = arith.constant 6 : i32
    %i1 = arith.constant 7 : i32
    %i2 = arith.constant 1 : i32
    %o0 = arith.constant 3 : i32
    %o1 = arith.constant 3 : i32
    %o2 = arith.constant 1 : i32
    kernel.launch @cudnnAdaptivePool_f32_flat2(
        %op, %rank, %n, %c, %i0, %i1, %i2, %o0, %o1, %o2,
        %input, %output)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
           memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @average_forward
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: call @polygeist_cudnn_adaptive_pool_f32(
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cudnnAveragePool_f32_flat2(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %input: memref<?xf32>, %output: memref<?xf32>) {
    kernel.yield
  }
  func.func @fixed_average(%input: memref<?xf32>, %output: memref<?xf32>,
      %op: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32) {
    kernel.launch @cudnnAveragePool_f32_flat2(
        %op, %rank, %n, %c, %i0, %i1, %i2, %o0, %o1, %o2,
        %input, %output)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
           memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @fixed_average
// CHECK: call @polygeist_cudnn_adaptive_pool_f32(
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cudnnBatchNormBackward_f32_dx(
      %n: i32, %c: i32, %s: i32,
      %grad: memref<?x?x?x?xf32>, %x: memref<?x?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %dx: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  func.func @batchnorm_dx(%n: i32, %c: i32, %s: i32,
      %grad: memref<?x?x?x?xf32>, %x: memref<?x?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %dx: memref<?x?x?x?xf32>) {
    kernel.launch @cudnnBatchNormBackward_f32_dx(
        %n, %c, %s, %grad, %x, %mean, %invstd, %dx)
        : (i32, i32, i32, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>,
           memref<?xf32>, memref<?xf32>, memref<?x?x?x?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @batchnorm_dx
// CHECK: %[[FALSE:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cudnn_batchnorm_backward_f32(
// CHECK-SAME: i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cudnnAdaptivePool_f32_flat3_bwd(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %grad: memref<?xf32>, %index: memref<?xi32>,
      %output: memref<?xf32>) {
    kernel.yield
  }

  func.func @max_backward(
      %grad: memref<?xf32>, %index: memref<?xi32>,
      %output: memref<?xf32>,
      %op: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32) {
    kernel.launch @cudnnAdaptivePool_f32_flat3_bwd(
        %op, %rank, %n, %c, %i0, %i1, %i2, %o0, %o1, %o2,
        %grad, %index, %output)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
           memref<?xf32>, memref<?xi32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @max_backward
// CHECK: call @polygeist_cudnn_adaptive_pool_f32(
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
// CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
