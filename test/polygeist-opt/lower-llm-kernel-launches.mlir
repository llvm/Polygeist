// RUN: polygeist-opt --lower-kernel-launch-to-cublas --split-input-file %s | FileCheck %s

module {
  kernel.defn @rmsnorm_f32(%x: memref<?xf32>, %weight: memref<?xf32>,
                           %out: memref<?xf32>) {
    kernel.yield
  }

  func.func @rms(%x: memref<?xf32>, %weight: memref<?xf32>,
                 %out: memref<?xf32>) {
    kernel.launch @rmsnorm_f32(%x, %weight, %out)
        : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @rms
// CHECK: call @polygeist_rmsnorm_f32
// CHECK-SAME: (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cublasDgemm(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>,
                           %c: tensor<?x?xf64>, %beta: f64, %alpha: f64)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  func.func @dgemm_memref_destination(
      %a: memref<?x?xf64>, %b: memref<?x?xf64>, %c: memref<?x?xf64>) {
    %at = bufferization.to_tensor %a : memref<?x?xf64>
    %bt = bufferization.to_tensor %b : memref<?x?xf64>
    %ct = bufferization.to_tensor %c : memref<?x?xf64>
    %c4 = arith.constant 4 : index
    %as = tensor.extract_slice %at[0, 0] [%c4, %c4] [1, 1]
      : tensor<?x?xf64> to tensor<?x?xf64>
    %bs = tensor.extract_slice %bt[0, 0] [%c4, %c4] [1, 1]
      : tensor<?x?xf64> to tensor<?x?xf64>
    %cs = tensor.extract_slice %ct[0, 0] [%c4, %c4] [1, 1]
      : tensor<?x?xf64> to tensor<?x?xf64>
    %beta = arith.constant 0.0 : f64
    %alpha = arith.constant 1.0 : f64
    %r = kernel.launch @cublasDgemm(%as, %bs, %cs, %beta, %alpha)
      : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, f64, f64)
        -> tensor<?x?xf64>
    %out = tensor.insert_slice %r into %ct[0, 0] [%c4, %c4] [1, 1]
      : tensor<?x?xf64> into tensor<?x?xf64>
    return
  }
}

// CHECK-LABEL: func.func @dgemm_memref_destination
// CHECK-NOT: bufferization.to_memref
// CHECK-COUNT-3: memref.extract_aligned_pointer_as_index %subview
// CHECK: call @polygeist_cublas_dgemm
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: memref.copy
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @rmsnorm_f32_tensor(%x: tensor<?xf32>,
                                  %weight: tensor<?xf32>,
                                  %out: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @rms_tensor(%x: tensor<?xf32>, %weight: tensor<?xf32>,
                        %out: tensor<?xf32>) -> tensor<?xf32> {
    %0 = kernel.launch @rmsnorm_f32_tensor(%x, %weight, %out)
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @rms_tensor
// CHECK: call @polygeist_rmsnorm_f32
// CHECK-SAME: (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cudnnSoftmaxForward(%x: memref<?xf32>) {
    kernel.yield
  }

  func.func @softmax(%x: memref<?xf32>) {
    kernel.launch @cudnnSoftmaxForward(%x) : (memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @softmax
// CHECK: call @polygeist_cudnn_softmax_forward_f32
// CHECK-SAME: (i32, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cublasSgemv(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                            %y: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }

  func.func @sgemv(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                   %y: tensor<?xf32>) -> tensor<?xf32> {
    %0 = kernel.launch @cublasSgemv(%A, %x, %y)
        : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @sgemv
// CHECK: call @polygeist_cublas_sgemv
// CHECK-SAME: (i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, f32, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cublasSgemv(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                            %y: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }

  // Model the C ABI shape used by the ATen extraction: tensors and their
  // slices are views of the original memref arguments.  Lowering must pass
  // those buffers directly to cuBLAS instead of materializing tensor copies.
  func.func @sgemv_memref_views(%A: memref<?x?xf32>, %x: memref<?xf32>,
                                %y: memref<?xf32>) {
    %At = bufferization.to_tensor %A : memref<?x?xf32>
    %xt = bufferization.to_tensor %x : memref<?xf32>
    %yt = bufferization.to_tensor %y : memref<?xf32>
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %As = tensor.extract_slice %At[0, 0] [%c4, %c8] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
    %xs = tensor.extract_slice %xt[0] [%c8] [1]
      : tensor<?xf32> to tensor<?xf32>
    %ys = tensor.extract_slice %yt[0] [%c4] [1]
      : tensor<?xf32> to tensor<?xf32>
    %r = kernel.launch @cublasSgemv(%As, %xs, %ys)
      : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %out = tensor.insert_slice %r into %yt[0] [%c4] [1]
      : tensor<?xf32> into tensor<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @sgemv_memref_views
// CHECK-NOT: bufferization.to_memref
// CHECK: memref.extract_aligned_pointer_as_index %arg0
// CHECK: memref.extract_aligned_pointer_as_index %arg1
// CHECK: memref.extract_aligned_pointer_as_index %arg2
// CHECK: call @polygeist_cublas_sgemv
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: kernel.launch

// -----

module {
  kernel.defn @cublasSgemv_T(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                              %y: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }

  func.func @sgemv_t(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                     %y: tensor<?xf32>) -> tensor<?xf32> {
    %0 = kernel.launch @cublasSgemv_T(%A, %x, %y)
        : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @sgemv_t
// CHECK: call @polygeist_cublas_sgemv_T
// CHECK-SAME: (i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, f32, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
