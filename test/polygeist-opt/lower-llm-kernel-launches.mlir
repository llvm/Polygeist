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
