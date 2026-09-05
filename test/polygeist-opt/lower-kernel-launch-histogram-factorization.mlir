// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cubHistogramEvenI32ShiftZero_memref(
      %samples: memref<?xi32>, %histogram: memref<?xi32>, %shift: i32) {
    kernel.yield
  }
  kernel.defn @cublasDtrsvLowerRowMajor_memref(
      %A: memref<?x?xf64>, %b: memref<?xf64>, %x: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cusolverDnDpotrfLowerRowMajor_memref(
      %A: memref<?x?xf64>) { kernel.yield }

  func.func @histogram(%samples: memref<?xi32>, %histogram: memref<?xi32>) {
    %shift = arith.constant 2 : i32
    kernel.launch @cubHistogramEvenI32ShiftZero_memref(
        %samples, %histogram, %shift)
        : (memref<?xi32>, memref<?xi32>, i32) -> ()
    return
  }

  func.func @trisolv(%A: memref<?x?xf64>, %b: memref<?xf64>,
                     %x: memref<?xf64>) {
    kernel.launch @cublasDtrsvLowerRowMajor_memref(%A, %b, %x)
        : (memref<?x?xf64>, memref<?xf64>, memref<?xf64>) -> ()
    return
  }

  func.func @cholesky(%A: memref<?x?xf64>) {
    kernel.launch @cusolverDnDpotrfLowerRowMajor_memref(%A)
        : (memref<?x?xf64>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @histogram
// CHECK: call @polygeist_cub_histogram_even_i32_shift_zero
// CHECK-SAME: (i32, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @trisolv
// CHECK: call @polygeist_cublas_dtrsv_lower_row_major
// CHECK-SAME: (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @cholesky
// CHECK: call @polygeist_cusolver_dpotrf_lower_row_major
// CHECK-SAME: (i32, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
