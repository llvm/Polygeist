// RUN: polygeist-opt --wrap-kernel-launch-pipeline %s | FileCheck %s
// RUN: polygeist-opt --wrap-kernel-launch-pipeline --wrap-kernel-launch-pipeline %s | FileCheck %s

module {
  func.func private @polygeist_cublas_dgemm(i32)
  func.func private @polygeist_cutensornet_contraction2_f64(i32)
  func.func private @some_host_helper(i32)

  func.func @matched_dispatch(%arg0: i32) {
    func.call @polygeist_cublas_dgemm(%arg0) : (i32) -> ()
    return
  }

  func.func @host_only(%arg0: i32) {
    func.call @some_host_helper(%arg0) : (i32) -> ()
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

// CHECK-DAG: func.func private @polygeist_cublas_pipeline_begin()
// CHECK-DAG: func.func private @polygeist_cublas_pipeline_end()
