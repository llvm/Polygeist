// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse %s | FileCheck %s
module {
  kernel.defn @read_value(%input: tensor<8xf32>) -> f32 {
    %zero = arith.constant 0.0 : f32
    kernel.yield %zero : f32
  }
  kernel.defn @mixed(%input: tensor<8xf32>, %output: tensor<8xf32>) -> (f32, tensor<8xf32>, i32) {
    %zero = arith.constant 0.0 : f32
    %integer = arith.constant 7 : i32
    kernel.yield %zero, %output, %integer : f32, tensor<8xf32>, i32
  }
  func.func @scalar(%input: memref<8xf32>) -> f32 {
    %t = bufferization.to_tensor %input restrict writable : memref<8xf32>
    %r = kernel.launch @read_value(%t) : (tensor<8xf32>) -> f32
    return %r : f32
  }
  // CHECK-LABEL: func.func @scalar
  // CHECK: %[[VALUE:.*]] = kernel.launch @read_value(%arg0) {polygeist.bufferized, polygeist.result_destinations = array<i64: -1>} : (memref<8xf32>) -> f32
  // CHECK: return %[[VALUE]] : f32
  func.func @mixed_results(%input: memref<8xf32>, %output: memref<8xf32>) -> (f32, i32) {
    %a = bufferization.to_tensor %input restrict writable : memref<8xf32>
    %b = bufferization.to_tensor %output restrict writable : memref<8xf32>
    %r:3 = kernel.launch @mixed(%a, %b) : (tensor<8xf32>, tensor<8xf32>) -> (f32, tensor<8xf32>, i32)
    %m = bufferization.to_memref %r#1 : memref<8xf32>
    memref.copy %m, %output : memref<8xf32> to memref<8xf32>
    return %r#0, %r#2 : f32, i32
  }
  // CHECK-LABEL: func.func @mixed_results
  // CHECK: %[[MIXED:.*]]:2 = kernel.launch @mixed(%arg0, %arg1) {polygeist.bufferized, polygeist.result_destinations = array<i64: -1, 1, -1>} : (memref<8xf32>, memref<8xf32>) -> (f32, i32)
  // CHECK: return %[[MIXED]]#0, %[[MIXED]]#1 : f32, i32
}
