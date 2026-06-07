// RUN: polygeist-opt --wrap-kernel-launch-pipeline %s | FileCheck %s
// RUN: polygeist-opt --wrap-kernel-launch-pipeline --wrap-kernel-launch-pipeline %s | FileCheck %s

module {
  func.func private @polygeist_cublas_dgemm(i32)
  func.func private @some_host_helper(i32)

  func.func @matched_dispatch(%arg0: i32) {
    func.call @polygeist_cublas_dgemm(%arg0) : (i32) -> ()
    return
  }

  func.func @host_only(%arg0: i32) {
    func.call @some_host_helper(%arg0) : (i32) -> ()
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

// CHECK-DAG: func.func private @polygeist_cublas_pipeline_begin()
// CHECK-DAG: func.func private @polygeist_cublas_pipeline_end()
