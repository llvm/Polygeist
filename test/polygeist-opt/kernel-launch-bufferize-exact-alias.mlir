// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse %s | FileCheck %s
module {
  kernel.defn @elementwise(%x: tensor<16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
    kernel.yield %out : tensor<16xf32>
  }
  func.func @exact_alias(%buffer: memref<16xf32>) {
    %t = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
    %r = kernel.launch @elementwise(%t, %t) {polygeist.elementwise_exact_alias} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %m = bufferization.to_memref %r : memref<16xf32>
    memref.copy %m, %buffer : memref<16xf32> to memref<16xf32>
    return
  }
  // CHECK-LABEL: func.func @exact_alias
  // CHECK-NOT: memref.alloc
  // CHECK: kernel.launch @elementwise(%arg0, %arg0)
  // CHECK-NOT: memref.alloc
  // CHECK: return

  func.func @unproven_alias(%buffer: memref<16xf32>) {
    %t = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
    %r = kernel.launch @elementwise(%t, %t) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %m = bufferization.to_memref %r : memref<16xf32>
    memref.copy %m, %buffer : memref<16xf32> to memref<16xf32>
    return
  }
  // CHECK-LABEL: func.func @unproven_alias
  // CHECK: %[[COPY:.*]] = memref.alloc()
  // CHECK: memref.copy %arg0, %[[COPY]]
  // CHECK: kernel.launch @elementwise(%arg0, %[[COPY]])

  func.func @live_old_value(%buffer: memref<16xf32>, %old: memref<16xf32>) {
    %t = bufferization.to_tensor %buffer restrict writable : memref<16xf32>
    %r = kernel.launch @elementwise(%t, %t) {polygeist.elementwise_exact_alias} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %before = bufferization.to_memref %t : memref<16xf32>
    memref.copy %before, %old : memref<16xf32> to memref<16xf32>
    %after = bufferization.to_memref %r : memref<16xf32>
    memref.copy %after, %buffer : memref<16xf32> to memref<16xf32>
    return
  }
  // CHECK-LABEL: func.func @live_old_value
  // CHECK: %[[COPY:.*]] = memref.alloc()
  // CHECK: memref.copy %arg0, %[[COPY]]
  // CHECK: kernel.launch @elementwise(%arg0, %[[COPY]])
  // CHECK: memref.copy %arg0, %arg1
}
