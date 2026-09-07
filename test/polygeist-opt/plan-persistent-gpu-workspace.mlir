// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=dynamic_target dynamic-vector-bound=64 dynamic-leading-dim-bound=32' %s | FileCheck %s --check-prefix=DYNAMIC

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

  func.func @dynamic_target(%n: index) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %vector = memref.alloc(%n) : memref<?xf32>
    %matrix = memref.alloc(%n) : memref<?x8xf32>
    memref.store %zero, %vector[%c0] : memref<?xf32>
    memref.store %zero, %matrix[%c0, %c0] : memref<?x8xf32>
    return
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

// DYNAMIC-DAG: memref.global "private" @__polygeist_workspace_dynamic_target_0 : memref<64xf32>
// DYNAMIC-DAG: memref.global "private" @__polygeist_workspace_dynamic_target_1 : memref<32x8xf32>
// DYNAMIC-LABEL: func.func @dynamic_target
// DYNAMIC: memref.subview {{.*}}[0] [%{{.*}}] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// DYNAMIC: memref.subview {{.*}}[0, 0] [%{{.*}}, 8] [1, 1] : memref<32x8xf32> to memref<?x8xf32, strided<[8, 1]>>
// DYNAMIC-NOT: memref.alloc
