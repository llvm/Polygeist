// RUN: not polygeist-opt %s --lower-kernel-launch-to-pva 2>&1 | FileCheck %s

module {
  kernel.defn @pvaBoxFilter_3x3_i8() { kernel.yield }
  kernel.defn @pvaBoxFilter_3x3_i16() { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_i8() { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_i16() { kernel.yield }
  kernel.defn @pvaBilateralFilter_3x3_i8() { kernel.yield }
  kernel.defn @pvaBilateralFilter_3x3_i16() { kernel.yield }
  kernel.defn @pvaHistogramEqualization_i8() { kernel.yield }

  func.func @legacy_launches_are_rejected() {
    kernel.launch @pvaBoxFilter_3x3_i8() : () -> ()
    kernel.launch @pvaBoxFilter_3x3_i16() : () -> ()
    kernel.launch @pvaGaussianFilter_3x3_i8() : () -> ()
    kernel.launch @pvaGaussianFilter_3x3_i16() : () -> ()
    kernel.launch @pvaBilateralFilter_3x3_i8() : () -> ()
    kernel.launch @pvaBilateralFilter_3x3_i16() : () -> ()
    kernel.launch @pvaHistogramEqualization_i8() : () -> ()
    return
  }
}

// CHECK-DAG: retired unsafe PVA ABI @pvaBoxFilter_3x3_i8
// CHECK-DAG: retired unsafe PVA ABI @pvaBoxFilter_3x3_i16
// CHECK-DAG: retired unsafe PVA ABI @pvaGaussianFilter_3x3_i8
// CHECK-DAG: retired unsafe PVA ABI @pvaGaussianFilter_3x3_i16
// CHECK-DAG: retired unsafe PVA ABI @pvaBilateralFilter_3x3_i8
// CHECK-DAG: retired unsafe PVA ABI @pvaBilateralFilter_3x3_i16
// CHECK-DAG: retired unsafe PVA ABI @pvaHistogramEqualization_i8
// CHECK-NOT: polygeist_pva_boxfilter_3x3_i8
// CHECK-NOT: polygeist_pva_gaussian_3x3_i8
// CHECK-NOT: polygeist_pva_bilateral_3x3_i8
// CHECK-NOT: polygeist_pva_histeq_i8
