// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=target' '--plan-persistent-gpu-workspace=function=target' %s | FileCheck %s
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=dynamic_target dynamic-vector-bound=64 dynamic-leading-dim-bound=32' %s | FileCheck %s --check-prefix=DYNAMIC
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=loop_target' %s | FileCheck %s --check-prefix=LOOP
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=loop_target' --empty-tensor-to-alloc-tensor --lower-affine '--one-shot-bufferize=bufferize-function-boundaries' %s | FileCheck %s --check-prefix=BUFFERIZED
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=late_loop_target' %s | FileCheck %s --check-prefix=LATE
// RUN: polygeist-opt '--plan-persistent-gpu-workspace=function=functional_tensor_target' %s | FileCheck %s --check-prefix=FUNCTIONAL

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

  func.func @loop_target(%n: index, %output: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    %output_tensor = bufferization.to_tensor %output restrict writable : memref<4xf32>
    %scratch = tensor.empty() : tensor<4xf32>
    %result:2 = affine.for %i = 0 to %n
        iter_args(%carried_scratch = %scratch,
                  %carried_output = %output_tensor)
        -> (tensor<4xf32>, tensor<4xf32>) {
      %next_scratch = linalg.fill ins(%zero : f32)
          outs(%carried_scratch : tensor<4xf32>) -> tensor<4xf32>
      %next_output = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%next_scratch : tensor<4xf32>)
          outs(%carried_output : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4xf32>
      affine.yield %next_scratch, %next_output
          : tensor<4xf32>, tensor<4xf32>
    }
    return
  }

  func.func @functional_tensor_target(%n: index, %input: tensor<4xf32>) {
    %zero = arith.constant 0.0 : f32
    %result = affine.for %i = 0 to %n
        iter_args(%carried = %input) -> tensor<4xf32> {
      %next = linalg.fill ins(%zero : f32)
          outs(%carried : tensor<4xf32>) -> tensor<4xf32>
      affine.yield %next : tensor<4xf32>
    }
    return
  }

  // Models a loop seen by the second planner invocation: an earlier pass has
  // already removed every scratch allocation, but ABI lowering produced a
  // fresh tensor result that must be copied back to the storage-backed carry.
  func.func @late_loop_target(%n: index, %output: memref<4xf32>,
                              %fresh_buffer: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    %output_tensor = bufferization.to_tensor %output restrict writable : memref<4xf32>
    %fresh = bufferization.to_tensor %fresh_buffer restrict writable : memref<4xf32>
    %result = affine.for %i = 0 to %n
        iter_args(%carried = %output_tensor) -> tensor<4xf32> {
      %next = linalg.fill ins(%zero : f32)
          outs(%fresh : tensor<4xf32>) -> tensor<4xf32>
      affine.yield %next : tensor<4xf32>
    }
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

// LOOP-DAG: memref.global "private" @__polygeist_workspace_loop_target_0 : memref<4xf32>
// LOOP-LABEL: func.func @loop_target
// LOOP: affine.for
// LOOP-SAME: iter_args(%[[SCRATCH:.*]] = %{{.*}}, %[[OUTPUT:.*]] = %{{.*}})
// LOOP: %[[NEXT_SCRATCH:.*]] = linalg.fill
// LOOP: %[[NEXT_OUTPUT:.*]] = linalg.generic
// LOOP: %[[STORED_SCRATCH:.*]] = bufferization.materialize_in_destination %[[NEXT_SCRATCH]] in %[[SCRATCH]] : tensor<4xf32>
// LOOP: %[[STORED_OUTPUT:.*]] = bufferization.materialize_in_destination %[[NEXT_OUTPUT]] in %[[OUTPUT]] : tensor<4xf32>
// LOOP: affine.yield %[[STORED_SCRATCH]], %[[STORED_OUTPUT]]

// BUFFERIZED-LABEL: func.func @loop_target
// BUFFERIZED: scf.for
// BUFFERIZED-SAME: iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (memref<4xf32>, memref<4xf32>)
// BUFFERIZED-NOT: tensor<
// BUFFERIZED-NOT: bufferization.materialize_in_destination

// LATE-LABEL: func.func @late_loop_target
// LATE: affine.for
// LATE-SAME: iter_args(%[[CARRIED:.*]] = %{{.*}})
// LATE: %[[NEXT:.*]] = linalg.fill
// LATE: %[[STORED:.*]] = bufferization.materialize_in_destination %[[NEXT]] in %[[CARRIED]] : tensor<4xf32>
// LATE: affine.yield %[[STORED]]

// FUNCTIONAL-LABEL: func.func @functional_tensor_target
// FUNCTIONAL: affine.yield %{{.*}} : tensor<4xf32>
// FUNCTIONAL-NOT: bufferization.materialize_in_destination
