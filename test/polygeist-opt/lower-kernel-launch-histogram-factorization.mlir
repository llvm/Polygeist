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
  kernel.defn @cublasDgramschmidtMGSRowMajor_memref(
      %A: memref<?x?xf64>, %R: memref<?x?xf64>,
      %Q: memref<?x?xf64>) { kernel.yield }
  kernel.defn @cublasDcovarianceRowMajor_memref(
      %count: f64, %data: memref<?x?xf64>, %cov: memref<?x?xf64>,
      %mean: memref<?xf64>) { kernel.yield }
  kernel.defn @cublasDcorrelationRowMajor_memref(
      %count: f64, %data: memref<?x?xf64>, %corr: memref<?x?xf64>,
      %mean: memref<?xf64>, %stddev: memref<?xf64>) { kernel.yield }

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

  func.func @gramschmidt(%A: memref<?x?xf64>, %R: memref<?x?xf64>,
                         %Q: memref<?x?xf64>) {
    kernel.launch @cublasDgramschmidtMGSRowMajor_memref(%A, %R, %Q)
        : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()
    return
  }

  func.func @covariance(%count: f64, %data: memref<?x?xf64>,
                        %cov: memref<?x?xf64>, %mean: memref<?xf64>) {
    kernel.launch @cublasDcovarianceRowMajor_memref(
        %count, %data, %cov, %mean)
        : (f64, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>) -> ()
    return
  }

  func.func @correlation(%count: f64, %data: memref<?x?xf64>,
                         %corr: memref<?x?xf64>, %mean: memref<?xf64>,
                         %stddev: memref<?xf64>) {
    kernel.launch @cublasDcorrelationRowMajor_memref(
        %count, %data, %corr, %mean, %stddev)
        : (f64, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>,
           memref<?xf64>) -> ()
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
// CHECK-LABEL: func.func @gramschmidt
// CHECK: call @polygeist_cublas_dgramschmidt_mgs_row_major
// CHECK-SAME: (i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32) -> ()
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @covariance
// CHECK: call @polygeist_cublas_dcovariance_row_major
// CHECK-SAME: (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @correlation
// CHECK: call @polygeist_cublas_dcorrelation_row_major
// CHECK-SAME: (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
