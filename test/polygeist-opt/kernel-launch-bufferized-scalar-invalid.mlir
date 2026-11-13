// RUN: polygeist-opt --split-input-file --verify-diagnostics %s
module {
  kernel.defn @value(%x: tensor<8xf32>) -> f32 {
    %zero = arith.constant 0.0 : f32
    kernel.yield %zero : f32
  }
  func.func @wrong_destination(%x: memref<8xf32>) -> f32 {
    // expected-error @+1 {{bufferized launch scalar result mismatch}}
    %r = kernel.launch @value(%x) {polygeist.bufferized, polygeist.result_destinations = array<i64: 0>} : (memref<8xf32>) -> f32
    return %r : f32
  }
}
// -----
module {
  kernel.defn @value(%x: tensor<8xf32>) -> f32 {
    %zero = arith.constant 0.0 : f32
    kernel.yield %zero : f32
  }
  func.func @wrong_scalar_type(%x: memref<8xf32>) -> f64 {
    // expected-error @+1 {{bufferized launch scalar result mismatch}}
    %r = kernel.launch @value(%x) {polygeist.bufferized, polygeist.result_destinations = array<i64: -1>} : (memref<8xf32>) -> f64
    return %r : f64
  }
}
// -----
module {
  kernel.defn @tensor(%x: tensor<8xf32>) -> tensor<8xf32> {
    kernel.yield %x : tensor<8xf32>
  }
  func.func @missing_tensor_destination(%x: memref<8xf32>) {
    // expected-error @+1 {{bufferized launch destination operand index is out of range}}
    kernel.launch @tensor(%x) {polygeist.bufferized, polygeist.result_destinations = array<i64: -1>} : (memref<8xf32>) -> ()
    return
  }
}
// -----
module {
  kernel.defn @tensor(%x: tensor<8xf32>, %y: tensor<8xf64>) -> tensor<8xf32> {
    kernel.yield %x : tensor<8xf32>
  }
  func.func @wrong_destination_type(%x: memref<8xf32>, %y: memref<8xf64>) {
    // expected-error @+1 {{bufferized launch destination type mismatch}}
    kernel.launch @tensor(%x, %y) {polygeist.bufferized, polygeist.result_destinations = array<i64: 1>} : (memref<8xf32>, memref<8xf64>) -> ()
    return
  }
}
// -----
module {
  kernel.defn @tensor(%x: tensor<8xf32>) -> tensor<8xf32> {
    kernel.yield %x : tensor<8xf32>
  }
  func.func @extra_scalar(%x: memref<8xf32>) -> f32 {
    // expected-error @+1 {{incorrect number of bufferized scalar results}}
    %r = kernel.launch @tensor(%x) {polygeist.bufferized, polygeist.result_destinations = array<i64: 0>} : (memref<8xf32>) -> f32
    return %r : f32
  }
}
