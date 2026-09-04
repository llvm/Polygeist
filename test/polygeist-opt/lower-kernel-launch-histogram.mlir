// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @customHistogramSaturatingU8_memref(
      %values: memref<?xi32>, %bins: memref<?xi8>,
      %count: i32, %num_bins: i32) {
    kernel.yield
  }
  kernel.defn @customTPACFHistogram_f32_memref(
      %data1: memref<?x3xf32>, %n1: i32,
      %data2: memref<?x3xf32>, %n2: i32, %self: i32,
      %bins: memref<?xi64>, %nbins: i32, %bounds: memref<?xf32>) {
    kernel.yield
  }
  func.func @histogram(%values: memref<?xi32>, %bins: memref<?xi8>,
                       %count: i32, %num_bins: i32) {
    kernel.launch @customHistogramSaturatingU8_memref(
        %values, %bins, %count, %num_bins)
        : (memref<?xi32>, memref<?xi8>, i32, i32) -> ()
    return
  }
  func.func @tpacf(%d1: memref<?x3xf32>, %n1: i32,
                   %d2: memref<?x3xf32>, %n2: i32, %self: i32,
                   %bins: memref<?xi64>, %nbins: i32,
                   %bounds: memref<?xf32>) {
    kernel.launch @customTPACFHistogram_f32_memref(
        %d1, %n1, %d2, %n2, %self, %bins, %nbins, %bounds)
        : (memref<?x3xf32>, i32, memref<?x3xf32>, i32, i32,
           memref<?xi64>, i32, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @histogram
// CHECK: call @polygeist_histogram_saturating_u8
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @tpacf
// CHECK: call @polygeist_tpacf_histogram_f32
// CHECK-NOT: kernel.launch
