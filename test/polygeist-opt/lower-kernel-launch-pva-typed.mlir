// RUN: polygeist-opt %s --lower-kernel-launch-to-pva | FileCheck %s

module {
  kernel.defn @pvaBoxFilter_3x3_u8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaBoxFilter_3x3_s8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaBoxFilter_3x3_u16(%h: i32, %w: i32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaBoxFilter_3x3_s16(%h: i32, %w: i32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_u8(%h: i32, %w: i32,
      %sx: f32, %sy: f32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_s8(%h: i32, %w: i32,
      %sx: f32, %sy: f32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_u16(%h: i32, %w: i32,
      %sx: f32, %sy: f32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaGaussianFilter_3x3_s16(%h: i32, %w: i32,
      %sx: f32, %sy: f32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaMorphologyDilate_3x3_u8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaMorphologyDilate_3x3_s8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaMorphologyDilate_3x3_u16(%h: i32, %w: i32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaMorphologyDilate_3x3_s16(%h: i32, %w: i32,
      %a: memref<?xi16>, %b: memref<?xi16>) { kernel.yield }
  kernel.defn @pvaBilateralFilter_3x3_u8(%h: i32, %w: i32,
      %sr: f32, %ss: f32, %a: memref<?xi8>, %b: memref<?xi8>) {
    kernel.yield
  }
  kernel.defn @pvaImageHistogram_256_u8_u32(%h: i32, %w: i32,
      %a: memref<?xi8>, %hist: memref<256xi32>) { kernel.yield }
  kernel.defn @pvaImageHistogram_256_u8_s32(%h: i32, %w: i32,
      %a: memref<?xi8>, %hist: memref<256xi32>) { kernel.yield }
  kernel.defn @pvaImageHistogram_256_u16_u32(%h: i32, %w: i32,
      %a: memref<?xi16>, %hist: memref<256xi32>) { kernel.yield }
  kernel.defn @pvaImageHistogram_256_u16_s32(%h: i32, %w: i32,
      %a: memref<?xi16>, %hist: memref<256xi32>) { kernel.yield }
  kernel.defn @pvaHistogramEqualization_u8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }

  func.func @typed_pva_abis(%h: i32, %w: i32, %sr: f32, %ss: f32,
      %a8: memref<?xi8>, %b8: memref<?xi8>,
      %a16: memref<?xi16>, %b16: memref<?xi16>,
      %hist: memref<256xi32>) {
    kernel.launch @pvaBoxFilter_3x3_u8(%h, %w, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaBoxFilter_3x3_s8(%h, %w, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaBoxFilter_3x3_u16(%h, %w, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaBoxFilter_3x3_s16(%h, %w, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaGaussianFilter_3x3_u8(%h, %w, %sr, %ss, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, f32, f32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaGaussianFilter_3x3_s8(%h, %w, %sr, %ss, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, f32, f32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaGaussianFilter_3x3_u16(%h, %w, %sr, %ss, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, f32, f32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaGaussianFilter_3x3_s16(%h, %w, %sr, %ss, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, f32, f32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaMorphologyDilate_3x3_u8(%h, %w, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaMorphologyDilate_3x3_s8(%h, %w, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaMorphologyDilate_3x3_u16(%h, %w, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaMorphologyDilate_3x3_s16(%h, %w, %a16, %b16) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<?xi16>) -> ()
    kernel.launch @pvaBilateralFilter_3x3_u8(%h, %w, %sr, %ss, %a8, %b8) {polygeist.numerical_contract = "approximate", polygeist.max_abs_error_budget = 1 : i64} : (i32, i32, f32, f32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaImageHistogram_256_u8_u32(%h, %w, %a8, %hist) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<256xi32>) -> ()
    kernel.launch @pvaImageHistogram_256_u8_s32(%h, %w, %a8, %hist) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<256xi32>) -> ()
    kernel.launch @pvaImageHistogram_256_u16_u32(%h, %w, %a16, %hist) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<256xi32>) -> ()
    kernel.launch @pvaImageHistogram_256_u16_s32(%h, %w, %a16, %hist) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi16>, memref<256xi32>) -> ()
    kernel.launch @pvaHistogramEqualization_u8(%h, %w, %a8, %b8) {polygeist.numerical_contract = "exact"} : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    return
  }
}

// CHECK-NOT: kernel.launch
// CHECK-DAG: call @polygeist_pva_boxfilter_3x3_u8
// CHECK-DAG: call @polygeist_pva_boxfilter_3x3_s8
// CHECK-DAG: call @polygeist_pva_boxfilter_3x3_u16
// CHECK-DAG: call @polygeist_pva_boxfilter_3x3_s16
// CHECK-DAG: call @polygeist_pva_gaussian_3x3_u8
// CHECK-DAG: call @polygeist_pva_gaussian_3x3_s8
// CHECK-DAG: call @polygeist_pva_gaussian_3x3_u16
// CHECK-DAG: call @polygeist_pva_gaussian_3x3_s16
// CHECK-DAG: call @polygeist_pva_morphology_dilate_3x3_u8
// CHECK-DAG: call @polygeist_pva_morphology_dilate_3x3_s8
// CHECK-DAG: call @polygeist_pva_morphology_dilate_3x3_u16
// CHECK-DAG: call @polygeist_pva_morphology_dilate_3x3_s16
// CHECK-DAG: call @polygeist_pva_bilateral_3x3_u8
// CHECK-DAG: call @polygeist_pva_histogram_256_u8_u32
// CHECK-DAG: call @polygeist_pva_histogram_256_u8_s32
// CHECK-DAG: call @polygeist_pva_histogram_256_u16_u32
// CHECK-DAG: call @polygeist_pva_histogram_256_u16_s32
// CHECK-DAG: call @polygeist_pva_histeq_u8
// CHECK-DAG: polygeist.numerical_contract = "exact"
// CHECK-DAG: polygeist.max_abs_error_budget = 1 : i64
// CHECK-DAG: polygeist.numerical_contract = "approximate"
