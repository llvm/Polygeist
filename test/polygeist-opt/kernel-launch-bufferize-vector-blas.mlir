// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse --lower-kernel-launch-to-cublas --canonicalize %s | FileCheck %s

module {
  func.func private @clobber(memref<?xf64>)

  // CHECK-LABEL: func.func @must_keep_snapshot
  // CHECK: memref.alloc
  // CHECK: call @polygeist_cuda_copy_f64
  // CHECK: call @clobber
  // CHECK: call @polygeist_cuda_copy_f64
  func.func @must_keep_snapshot(%y: memref<?xf64>, %n: index) {
    %snapshot = memref.alloc(%n) : memref<?xf64>
    memref.copy %y, %snapshot : memref<?xf64> to memref<?xf64>
    call @clobber(%y) : (memref<?xf64>) -> ()
    memref.copy %snapshot, %y : memref<?xf64> to memref<?xf64>
    return
  }
  kernel.defn @memset_zero_1D(%y: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }
  kernel.defn @cublasDgemv(%a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }
  kernel.defn @cublasDgemv_T(%a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }
  kernel.defn @cublasDaxpby(%x: tensor<?xf64>, %y: tensor<?xf64>, %alpha: f64, %beta: f64) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }

  // A matrix view has a physical leading dimension of 8, regardless of its
  // logical column count. Vector offsets must also survive ABI lowering.
  // CHECK-LABEL: func.func @vector_chain
  // CHECK-DAG: %[[LDA:.*]] = arith.constant 8 : i32
  // CHECK-DAG: arith.constant 24 : i64
  // CHECK-DAG: arith.constant 16 : i64
  // CHECK-NOT: memref.alloc
  // CHECK: call @polygeist_cublas_memset_zero_1d
  // CHECK: call @polygeist_cublas_dgemv({{.*}}, %[[LDA]],
  // CHECK: call @polygeist_cublas_daxpby
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: call @polygeist_cuda_copy
  // CHECK: return
  func.func @vector_chain(%a: memref<?x?xf64, strided<[8, 1], offset: 3>>,
      %x: memref<?xf64, strided<[1], offset: 2>>, %y: memref<?xf64>,
      %z: memref<?xf64>, %alpha: f64, %beta: f64) {
    %at = bufferization.to_tensor %a restrict : memref<?x?xf64, strided<[8, 1], offset: 3>>
    %xt = bufferization.to_tensor %x restrict : memref<?xf64, strided<[1], offset: 2>>
    %yt = bufferization.to_tensor %y restrict writable : memref<?xf64>
    %zt = bufferization.to_tensor %z restrict : memref<?xf64>
    %zero = kernel.launch @memset_zero_1D(%yt) : (tensor<?xf64>) -> tensor<?xf64>
    %mv = kernel.launch @cublasDgemv(%at, %xt, %zero) : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %result = kernel.launch @cublasDaxpby(%zt, %mv, %alpha, %beta) {polygeist.result_destinations = array<i64: 1>} : (tensor<?xf64>, tensor<?xf64>, f64, f64) -> tensor<?xf64>
    %mr = bufferization.to_memref %result : memref<?xf64>
    memref.copy %mr, %y : memref<?xf64> to memref<?xf64>
    %c0 = arith.constant 0 : index
    %n = memref.dim %y, %c0 : memref<?xf64>
    %snapshot = memref.alloc(%n) : memref<?xf64>
    memref.copy %y, %snapshot : memref<?xf64> to memref<?xf64>
    memref.copy %snapshot, %y : memref<?xf64> to memref<?xf64>
    return
  }

  // CHECK-LABEL: func.func @transpose
  // CHECK: call @polygeist_cublas_dgemv_T
  // CHECK-NOT: memref.alloc
  // CHECK: return
  func.func @transpose(%a: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>) {
    %at = bufferization.to_tensor %a restrict : memref<?x?xf64>
    %xt = bufferization.to_tensor %x restrict : memref<?xf64>
    %yt = bufferization.to_tensor %y restrict writable : memref<?xf64>
    %result = kernel.launch @cublasDgemv_T(%at, %xt, %yt) : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %mr = bufferization.to_memref %result : memref<?xf64>
    memref.copy %mr, %y : memref<?xf64> to memref<?xf64>
    return
  }
}
