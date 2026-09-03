// RUN: polygeist-opt --wrap-kernel-launch-pipeline %s | FileCheck %s
// RUN: polygeist-opt --wrap-kernel-launch-pipeline --wrap-kernel-launch-pipeline %s | FileCheck %s
// RUN: polygeist-opt '--wrap-kernel-launch-pipeline=cuda-graphs=true' %s | FileCheck %s --check-prefix=GRAPH
// RUN: polygeist-opt '--wrap-kernel-launch-pipeline=cuda-graphs=true' '--wrap-kernel-launch-pipeline=cuda-graphs=true' %s | FileCheck %s --check-prefix=GRAPH
// RUN: polygeist-opt '--wrap-kernel-launch-pipeline=cuda-graphs=true capture-host-mapped-cutensornet=true' %s | FileCheck %s --check-prefix=HOST-GRAPH
// RUN: polygeist-opt '--wrap-kernel-launch-pipeline=cuda-graphs=true capture-host-mapped-cutensornet=true maximal-device-sequence=true' %s | FileCheck %s --check-prefix=MAX-GRAPH

module attributes {gpu.container_module} {
  gpu.module @generated_kernels {
    gpu.func @scale_kernel() kernel {
      gpu.return
    }
  }

  func.func private @polygeist_cublas_dgemm(i32)
  func.func private @polygeist_cutensornet_contraction2_f64(i32)
  func.func private @polygeist_cutensornet_contraction2_f64_device(i32)
      attributes {polygeist.cuda_graph_safe}
  func.func private @generated_scale_device(i32)
      attributes {polygeist.cuda_graph_safe}
  func.func private @some_host_helper(i32)

  func.func @matched_dispatch(%arg0: i32) {
    func.call @polygeist_cublas_dgemm(%arg0) : (i32) -> ()
    return
  }

  func.func @host_only(%arg0: i32) {
    func.call @some_host_helper(%arg0) : (i32) -> ()
    return
  }

  func.func @device_resident_dispatch(%arg0: i32) {
    func.call @polygeist_cutensornet_contraction2_f64(%arg0)
        {polygeist.cuda_graph_safe} : (i32) -> ()
    return
  }

  func.func @mixed_dispatch(%arg0: i32, %input: tensor<4xf64>,
                            %output: tensor<4xf64>) -> tensor<4xf64> {
    func.call @polygeist_cutensornet_contraction2_f64(%arg0) : (i32) -> ()
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : tensor<4xf64>) outs(%output : tensor<4xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %in, %out : f64
      linalg.yield %sum : f64
    } -> tensor<4xf64>
    func.call @polygeist_cutensornet_contraction2_f64(%arg0) : (i32) -> ()
    return %0 : tensor<4xf64>
  }

  // A compiler-generated wrapper and a device-library call form one capture
  // region through their declaration-level graph-safety contract. The
  // generated wrapper is intentionally not named polygeist_*.
  func.func @mixed_generated_and_library(%arg0: i32) {
    func.call @polygeist_cutensornet_contraction2_f64_device(%arg0)
        : (i32) -> ()
    func.call @generated_scale_device(%arg0) : (i32) -> ()
    return
  }

  func.func @mixed_mlir_gpu_and_library(%arg0: i32) {
    %c1 = arith.constant 1 : index
    func.call @polygeist_cutensornet_contraction2_f64_device(%arg0)
        : (i32) -> ()
    gpu.launch_func @generated_kernels::@scale_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        {polygeist.cuda_graph_safe}
    func.call @polygeist_cutensornet_contraction2_f64_device(%arg0)
        : (i32) -> ()
    return
  }

  func.func @device_sequence_with_metadata(%arg0: i32) {
    %c0 = arith.constant 0 : index
    %metadata = memref.alloca() : memref<1xi32>
    func.call @polygeist_cutensornet_contraction2_f64_device(%arg0)
        : (i32) -> ()
    memref.store %arg0, %metadata[%c0] : memref<1xi32>
    func.call @polygeist_cutensornet_contraction2_f64_device(%arg0)
        : (i32) -> ()
    return
  }
}

// CHECK-LABEL: func.func @matched_dispatch
// CHECK-NEXT: call @polygeist_cublas_pipeline_begin() : () -> ()
// CHECK-NEXT: call @polygeist_cublas_dgemm
// CHECK-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// CHECK-NEXT: return

// CHECK-LABEL: func.func @host_only
// CHECK-NEXT: call @some_host_helper
// CHECK-NEXT: return

// CHECK-LABEL: func.func @mixed_dispatch
// CHECK: call @polygeist_cublas_pipeline_begin() : () -> ()
// CHECK-NEXT: call @polygeist_cutensornet_contraction2_f64
// CHECK-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// CHECK-NEXT: %[[GENERIC:.*]] = linalg.generic
// CHECK: call @polygeist_cublas_pipeline_begin() : () -> ()
// CHECK-NEXT: call @polygeist_cutensornet_contraction2_f64
// CHECK-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// CHECK-NEXT: return %[[GENERIC]]

// CHECK-LABEL: func.func @mixed_generated_and_library
// CHECK-NEXT: call @polygeist_cublas_pipeline_begin() : () -> ()
// CHECK-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// CHECK-NEXT: call @generated_scale_device
// CHECK-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// CHECK-NEXT: return

// CHECK-LABEL: func.func @mixed_mlir_gpu_and_library
// CHECK: call @polygeist_cublas_pipeline_begin() : () -> ()
// CHECK-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// CHECK-NEXT: gpu.launch_func @generated_kernels::@scale_kernel
// CHECK-SAME: polygeist.cuda_graph_safe
// CHECK-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// CHECK-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// CHECK-NEXT: return
// CHECK-DAG: func.func private @polygeist_cublas_pipeline_begin()
// CHECK-DAG: func.func private @polygeist_cublas_pipeline_end()

// GRAPH-LABEL: func.func @device_resident_dispatch
// GRAPH: %[[ID:.*]] = arith.constant 0 : i64
// GRAPH-NEXT: %[[DO:.*]] = call @polygeist_cuda_graph_begin(%[[ID]]) : (i64) -> i32
// GRAPH-NEXT: %[[ZERO:.*]] = arith.constant 0 : i32
// GRAPH-NEXT: %[[COND:.*]] = arith.cmpi ne, %[[DO]], %[[ZERO]] : i32
// GRAPH-NEXT: scf.if %[[COND]] {
// GRAPH: call @polygeist_cublas_pipeline_begin() : () -> ()
// GRAPH-NEXT: call @polygeist_cutensornet_contraction2_f64(%arg0)
// GRAPH-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// GRAPH-NEXT: call @polygeist_cuda_graph_end(%[[ID]]) : (i64) -> ()
// GRAPH-NEXT: }
// GRAPH-NEXT: return
// GRAPH-LABEL: func.func @mixed_generated_and_library
// GRAPH: %[[MIXED_ID:.*]] = arith.constant 1 : i64
// GRAPH-NEXT: %[[MIXED_DO:.*]] = call @polygeist_cuda_graph_begin(%[[MIXED_ID]]) : (i64) -> i32
// GRAPH: scf.if
// GRAPH: call @polygeist_cublas_pipeline_begin() : () -> ()
// GRAPH-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// GRAPH-NEXT: call @generated_scale_device
// GRAPH-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// GRAPH-NEXT: call @polygeist_cuda_graph_end(%[[MIXED_ID]]) : (i64) -> ()
// GRAPH-NEXT: }
// GRAPH-NEXT: return

// GRAPH-LABEL: func.func @mixed_mlir_gpu_and_library
// GRAPH: %[[GPU_ID:.*]] = arith.constant 2 : i64
// GRAPH-NEXT: %[[GPU_DO:.*]] = call @polygeist_cuda_graph_begin(%[[GPU_ID]]) : (i64) -> i32
// GRAPH: scf.if
// GRAPH: call @polygeist_cublas_pipeline_begin() : () -> ()
// GRAPH-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// GRAPH-NEXT: gpu.launch_func @generated_kernels::@scale_kernel
// GRAPH-SAME: polygeist.cuda_graph_safe
// GRAPH-NEXT: call @polygeist_cutensornet_contraction2_f64_device
// GRAPH-NEXT: call @polygeist_cublas_pipeline_end() : () -> ()
// GRAPH-NEXT: call @polygeist_cuda_graph_end(%[[GPU_ID]]) : (i64) -> ()
// GRAPH-NEXT: }
// GRAPH-NEXT: return
// GRAPH-DAG: func.func private @polygeist_cuda_graph_begin(i64) -> i32
// GRAPH-DAG: func.func private @polygeist_cuda_graph_end(i64)

// HOST-GRAPH-LABEL: func.func @matched_dispatch
// HOST-GRAPH-NOT: call @polygeist_cuda_graph_begin
// HOST-GRAPH: call @polygeist_cublas_dgemm
// HOST-GRAPH-NOT: call @polygeist_cuda_graph_begin
// HOST-GRAPH: return

// MAX-GRAPH-LABEL: func.func @device_sequence_with_metadata
// MAX-GRAPH: call @polygeist_cuda_graph_begin
// MAX-GRAPH: scf.if
// MAX-GRAPH: call @polygeist_cutensornet_contraction2_f64_device
// MAX-GRAPH: memref.store
// MAX-GRAPH: call @polygeist_cutensornet_contraction2_f64_device
// MAX-GRAPH: call @polygeist_cuda_graph_end
// MAX-GRAPH: polygeist.cuda_graph_scope
// MAX-GRAPH-NOT: polygeist.cuda_graph_scope
// MAX-GRAPH: return

// HOST-GRAPH-LABEL: func.func @mixed_dispatch
// HOST-GRAPH: call @polygeist_cuda_graph_begin
// HOST-GRAPH: scf.if
// HOST-GRAPH: call @polygeist_cutensornet_contraction2_f64
// HOST-GRAPH: polygeist.cuda_graph_scope
