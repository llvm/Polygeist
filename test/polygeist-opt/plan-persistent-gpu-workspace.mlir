// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s

module {
  func.func @target(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %zero = arith.constant 0.0 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %scratch0 = tensor.empty() : tensor<4x8xf32>
    %scratch1 = bufferization.alloc_tensor() : tensor<16xf64>
    %scratch2 = memref.alloc() : memref<32xi32>
    memref.store %c0_i32, %scratch2[%c0] : memref<32xi32>
    %filled = linalg.fill ins(%zero : f32) outs(%scratch0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %filled : tensor<4x8xf32>
  }

  func.func @untouched() -> tensor<2xf32> {
    %scratch = tensor.empty() : tensor<2xf32>
    return %scratch : tensor<2xf32>
  }
}

// CHECK-DAG: memref.global "private" @__polygeist_workspace_target_0 : memref<4x8xf32>
// CHECK-DAG: memref.global "private" @__polygeist_workspace_target_1 : memref<16xf64>
// CHECK-DAG: memref.global "private" @__polygeist_workspace_target_2 : memref<32xi32>
// CHECK: func.func @target
// CHECK-SAME: attributes {polygeist.persistent_workspace}
// CHECK-DAG: memref.get_global @__polygeist_workspace_target_0 : memref<4x8xf32>
// CHECK-DAG: bufferization.to_tensor {{.*}} restrict writable : memref<4x8xf32>
// CHECK-DAG: memref.get_global @__polygeist_workspace_target_1 : memref<16xf64>
// CHECK-DAG: bufferization.to_tensor {{.*}} restrict writable : memref<16xf64>
// CHECK-DAG: memref.get_global @__polygeist_workspace_target_2 : memref<32xi32>
// CHECK: func.func @untouched
// CHECK: tensor.empty() : tensor<2xf32>
