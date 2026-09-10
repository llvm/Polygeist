#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map16 = affine_map<(d0) -> (d0)>
#map17 = affine_map<(d0) -> ()>
#map18 = affine_map<(d0, d1) -> (d1)>
#map19 = affine_map<(d0, d1) -> (d0)>
#map20 = affine_map<(d0, d1) -> (d1, d0)>
#map21 = affine_map<(d0) -> (d0, d0)>
#map22 = affine_map<(d0) -> (d0 + 1)>
#map23 = affine_map<(d0, d1, d2) -> (d2, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  kernel.defn @cubHistogramEvenI32ShiftZero_memref(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) {
    kernel.yield
  }
  kernel.defn @cublasDtrsvLowerRowMajor_memref(%arg0: memref<?x?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cublasStrsvLowerRowMajor_memref(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cublasDsymmLeftLowerRowMajor_memref(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: f64, %arg4: f64) {
    kernel.yield
  }
  kernel.defn @cublasSsymmLeftLowerRowMajor_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: f32, %arg4: f32) {
    kernel.yield
  }
  kernel.defn @cublasDtrmmLeftLowerTransUnitRowMajor_memref(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: f64) {
    kernel.yield
  }
  kernel.defn @cublasStrmmLeftLowerTransUnitRowMajor_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: f32) {
    kernel.yield
  }
  kernel.defn @cusolverDnDpotrfLowerRowMajor_memref(%arg0: memref<?x?xf64>) {
    kernel.yield
  }
  kernel.defn @cublasDgramschmidtMGSRowMajor_memref(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>) {
    kernel.yield
  }
  kernel.defn @cublasDcovarianceRowMajor_memref(%arg0: f64, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cublasScovarianceRowMajor_memref(%arg0: f32, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cublasDcorrelationRowMajor_memref(%arg0: f64, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cublasScorrelationRowMajor_memref(%arg0: f32, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMV_CSR_f32_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMV_CSR_f64_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMM_CSR_f32_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMM_COO_f32_memref(%arg0: index, %arg1: index, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMM_BSR_f32_memref(%arg0: index, %arg1: index, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseSDDMM_CSR_f32_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: f32, %arg7: f32, %arg8: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cusparseXcoo2csr_i32_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cusparseXcsr2coo_i32_memref(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cusparseSpMV_JDS_f32_memref(%arg0: index, %arg1: index, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: memref<?xf32>, %arg6: memref<?xi32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @custenStencil2DXY_f64_memref(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?xf64>, %arg3: i32) {
    kernel.yield
  }
  kernel.defn @custenStencil2DXY_f64_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?xf64>, %arg3: i32) -> tensor<?x?xf64> {
    kernel.yield %arg1 : tensor<?x?xf64>
  }
  kernel.defn @cublasGemmEx_i8_i32_tensor(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
    kernel.yield %arg2 : tensor<?x?xi32>
  }
  kernel.defn @cublasSnrm2_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cublasJointMaxAbsProduct_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnFeatureMaskScale_f32_tensor(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: f32, %arg3: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg3 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolutionTranspose2D_f32_memref(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnConvolutionTranspose3D_f32_memref(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnConvolutionBackwardFilter3D_f32_memref(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnDepthwiseConvolution2D_f32_memref(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cutensorKroneckerProduct2D_f32_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnBinaryCrossEntropyMean_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnConvolutionTBC_f32_memref(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnConvolutionTBCBackward_f32_memref(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnTransformBiasRescaleQKV_f32_memref(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: f32, %arg3: memref<?x?x?x?xf32>, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAddrElementwise_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnLogSigmoid_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedLogicalAnd_i32_memref(%arg0: memref<?x64xi32>, %arg1: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedLogicalSelect_i32_memref(%arg0: memref<?x64xi32>, %arg1: memref<?x64xi32>, %arg2: i32, %arg3: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cublasSdot_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cublasDdot_memref(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedArgMax_f32_i32_memref(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedArgMin_f32_i32_memref(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cubQuantColOffsets_i8_i32_memref(%arg0: memref<?x48xi8>, %arg1: i32, %arg2: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cubAdjacentDifference_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cublasSgemvTZero_memref(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedSum_f32_memref(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedNanSum_f32_memref(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedSum_f64_memref(%arg0: memref<?x?xf64>, %arg1: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedMin_f32_memref(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cubSegmentedMax_f32_memref(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cutensornetNetwork_f32_n3_aten(%arg0: memref<?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnSinc_f32_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnReduceSum_f32(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    kernel.yield %arg1 : tensor<f32>
  }
  kernel.defn @cudnnReduceSum_f64(%arg0: tensor<?xf64>, %arg1: tensor<f64>) -> tensor<f64> {
    kernel.yield %arg1 : tensor<f64>
  }
  kernel.defn @cudnnReduceProduct_f32(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    kernel.yield %arg1 : tensor<f32>
  }
  kernel.defn @cudnnReduceMin_f32(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    kernel.yield %arg1 : tensor<f32>
  }
  kernel.defn @cudnnReduceMax_f32(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    kernel.yield %arg1 : tensor<f32>
  }
  kernel.defn @cudnnReduceMinMax_f32(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    kernel.yield %arg1, %arg2 : tensor<f32>, tensor<f32>
  }
  kernel.defn @cudnnReduceTrace_f32(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    kernel.yield %arg1 : tensor<f32>
  }
  kernel.defn @cubSegmentedLogicalAnd_i32(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg1 : tensor<?xi32>
  }
  kernel.defn @cubSegmentedLogicalOr_i32(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg1 : tensor<?xi32>
  }
  kernel.defn @cubSegmentedBitXor_i32(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg1 : tensor<?xi32>
  }
  kernel.defn @cubSegmentedPrefixSum_f32(%arg0: tensor<?x?xf32>, %arg1: tensor<?xi32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg2 : tensor<?xf32>
  }
  kernel.defn @cubSegmentedPrefixLogicalAnd_i32(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg2 : tensor<?xi32>
  }
  kernel.defn @cutensorPermute_f32_r2_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg1 : tensor<?x?xf32>
  }
  kernel.defn @cutensorPermute_f32_r3_tensor(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?xf32>
  }
  kernel.defn @cutensorPermute_f32_r4_tensor(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cutensorPermute_f32_r5_tensor(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?x?xf32>
  }
  kernel.defn @cutensorPermute_f32_r6_tensor(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?x?x?xf32>
  }
  kernel.defn @cutensorUnary_abs_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_acos_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_acosh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_asin_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_asinh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_atan_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_atanh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_ceil_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_cos_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_cosh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_exp_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_floor_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_log_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_mish_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_neg_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_reciprocal_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_relu_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_sigmoid_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_silu_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_sin_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_sinh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_sqrt_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_tan_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cutensorUnary_tanh_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cudnnAddTensor_batched(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %out, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnBatchNormalizationForwardInference(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>, %arg5: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%arg5 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %in_3: f32, %out: f32):
      %1 = arith.subf %in, %in_1 : f32
      %2 = arith.mulf %1, %in_2 : f32
      %3 = arith.mulf %in_0, %2 : f32
      %4 = arith.addf %3, %in_3 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolutionFwd_batched(%arg0: tensor<?x?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolutionFwd_batched_expanded(%arg0: tensor<?x?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x?x?x?xf32> {
    kernel.yield %arg0 : tensor<?x?x?x?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolution2DWindow_f32(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnAvgPoolWindow_f32(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnAvgPoolWindow_f32_expanded(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?x?xf32>, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) -> tensor<?x?x?x?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?x?x?x?xf32>
  }
  kernel.defn @cudnnAdaptivePool_f32_flat2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?xf32>, %arg11: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnBilinearUpsample2x_f32_r4(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x3x4x4xf32>, %arg11: memref<?x3x8x8xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_flat3_fwd(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?xf32>, %arg11: memref<?xf32>, %arg12: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_flat3_bwd(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?xf32>, %arg11: memref<?xi32>, %arg12: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?xf32>, %arg11: memref<?x?xf32>, %arg12: memref<?x?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r4_fwd(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xf32>, %arg12: memref<?x?x?x?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r4_bwd(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xi32>, %arg12: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r5(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?x?x?x?xf32>, %arg11: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAveragePool_f32_flat2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?xf32>, %arg11: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAveragePool_f32_r4(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAveragePool_f32_r5(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<?x?x?x?x?xf32>, %arg11: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnBatchNormBackward_f32_full(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?x?x?xf32>, %arg9: memref<?xf32>, %arg10: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnBatchNormBackward_f32_dx(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?x?xf32>, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnMaxPoolFwd_batched(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0 : tensor<?x?x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.cmpf ogt, %in, %out : f32
      %2 = arith.select %1, %in, %out : f32
      linalg.yield %2 : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cublasDgemm(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: f64, %arg4: f64) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %2 = arith.mulf %out, %arg3 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    %1 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%0 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = arith.mulf %in, %in_0 : f64
      %3 = arith.mulf %arg4, %2 : f64
      %4 = arith.addf %out, %3 : f64
      linalg.yield %4 : f64
    } -> tensor<?x?xf64>
    kernel.yield %1 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm_simple(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm_subtract(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm_strided_batched_subtract(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%arg2 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?xf64>
    kernel.yield %0 : tensor<?x?x?xf64>
  }
  kernel.defn @cublasDgemv_strided_batched_subtract(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map14, #map8, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm_zero(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %arg2 : tensor<?x?xf64>
  }
  kernel.defn @cublasSgemm_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nn_alpha_beta(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_alpha_beta(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_alpha_beta(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_alpha_beta(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nn_alpha(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_alpha(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_alpha(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_alpha(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nn_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_strided_batched_nn_zero(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %arg2 : tensor<?x?x?xf32>
  }
  kernel.defn @cublasSgemm_broadcast3d_simple(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map14, #map14, #map14], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?xf32>
  }
  kernel.defn @cublasSgemv_broadcast2d_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_broadcast3d_colmajor_nt_alpha_beta(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?x?xf32> {
    kernel.yield %arg0 : tensor<?x?x?xf32>
  }
  kernel.defn @cublasSgemm_flat_colmajor_nt_alpha_beta(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: f32, %arg10: f32) {
    kernel.yield
  }
  kernel.defn @cublasSgemm_broadcast3d_memref(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.generic {indexing_maps = [#map14, #map14, #map14], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    kernel.yield
  }
  kernel.defn @cublasSgemm_strided_batched_broadcast_rhs(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map11, #map15, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?xf32>
  }
  kernel.defn @cudnnConvolutionFwd_im2col_gemm(%arg0: memref<?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    kernel.yield
  }
  kernel.defn @rmsnorm_f32(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @rmsnorm_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg2 : tensor<?xf32>
  }
  kernel.defn @cubCountNonzero1D_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<i32> {
    kernel.yield %arg1 : tensor<i32>
  }
  kernel.defn @cubSegmentedCountNonzero2D_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg1 : tensor<?xi32>
  }
  kernel.defn @cubEqualAll1D_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<i32>) -> tensor<i32> {
    kernel.yield %arg2 : tensor<i32>
  }
  kernel.defn @cubSegmentedLogicalSelect_i32_tensor(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: i1, %arg3: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %arg3 : tensor<?xi32>
  }
  kernel.defn @cublasDdot(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16, #map17], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<?xf64>, tensor<?xf64>) outs(%arg2 : tensor<f64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<f64>
    kernel.yield %0 : tensor<f64>
  }
  kernel.defn @cublasSdot(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16, #map17], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
    kernel.yield %0 : tensor<f32>
  }
  kernel.defn @whisperExpShiftSum_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<f32>, %arg3: f32) -> (tensor<?xf32>, tensor<f32>) {
    %0:2 = linalg.generic {indexing_maps = [#map16, #map16, #map17], iterator_types = ["reduction"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg2 : tensor<?xf32>, tensor<f32>) {
    ^bb0(%in: f32, %out: f32, %out_0: f32):
      %1 = arith.subf %in, %arg3 : f32
      %2 = math.exp %1 : f32
      %3 = arith.addf %out_0, %2 : f32
      linalg.yield %2, %3 : f32, f32
    } -> (tensor<?xf32>, tensor<f32>)
    kernel.yield %0#0, %0#1 : tensor<?xf32>, tensor<f32>
  }
  kernel.defn @cudnnSoftmaxForward(%arg0: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnSoftmaxForward_tensor(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg0 : tensor<?xf32>
  }
  kernel.defn @cudnnSoftmaxForwardOut_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cudaCopy1D_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cudaCopy2D_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @cublasBroadcastAxis0_f32(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg1 : tensor<?x?xf32>
  }
  kernel.defn @cublasBroadcastAxis1_f32(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg1 : tensor<?x?xf32>
  }
  kernel.defn @cudaCopy3D_f32_tensor(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map14, #map14], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?xf32>
  }
  kernel.defn @cudaCopy6D_f32_tensor(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?x?x?xf32>
    kernel.yield %0 : tensor<?x?x?x?x?x?xf32>
  }
  kernel.defn @cudaAdd_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16, #map16], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cudaMaskSelect_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: i32) -> tensor<?xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %0 = linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.cmpi sgt, %2, %arg2 : i32
      %4 = arith.extui %3 : i1 to i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.subf %cst, %5 : f32
      %7 = arith.mulf %6, %in : f32
      %8 = arith.mulf %5, %cst_0 : f32
      %9 = arith.addf %7, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cudaSwiGLU_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = linalg.generic {indexing_maps = [#map16, #map16, #map16], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.negf %in : f32
      %2 = math.exp %1 : f32
      %3 = arith.addf %2, %cst : f32
      %4 = arith.divf %in, %3 : f32
      %5 = arith.mulf %4, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cudaRopeMulMulSub_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map7, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) outs(%arg4 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.mulf %in_1, %in_2 : f32
      %3 = arith.subf %1, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @cudaRopeMulMulAdd_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map7, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) outs(%arg4 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.mulf %in_1, %in_2 : f32
      %3 = arith.addf %1, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @cublasDgemm_alpha_only(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: f64) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.mulf %arg3, %1 : f64
      %3 = arith.addf %out, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgeam_scale2D(%arg0: tensor<?x?xf64>, %arg1: f64) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%arg0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %1 = arith.mulf %out, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemv(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasDgemv_T(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map20, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasDgemv_subtract(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasDgemv_subtract_T(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map20, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasSgemv(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cublasSgemv_T(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map20, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cublasSgemv_T_zero(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %arg2 : tensor<?xf32>
  }
  kernel.defn @cublasSgemv_broadcast2d(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @cublasDgemv_alpha(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>, %arg3: f64) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.mulf %arg3, %1 : f64
      %3 = arith.addf %out, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasSgemv_alpha(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: f32) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.mulf %arg3, %1 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @cublasSgemv_alpha_memref(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<?xf32>, %arg3: f32) {
    linalg.generic {indexing_maps = [#map7, #map18, #map19], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%arg2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.mulf %arg3, %0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    kernel.yield
  }
  kernel.defn @cublasSgemv_alpha_T_memref(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<?xf32>, %arg3: f32) {
    kernel.yield
  }
  kernel.defn @cublasDger_rank2(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>, %arg3: tensor<?xf64>, %arg4: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map19, #map18, #map19, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%arg4 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.mulf %in_1, %in_2 : f64
      %3 = arith.addf %out, %1 : f64
      %4 = arith.addf %3, %2 : f64
      linalg.yield %4 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasSger_rank2(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map19, #map18, #map19, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%arg4 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.mulf %in_1, %in_2 : f32
      %3 = arith.addf %out, %1 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @cublasSger_rank2_memref(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<?xf32, strided<[1], offset: ?>>, %arg3: memref<?xf32, strided<[1], offset: ?>>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    linalg.generic {indexing_maps = [#map19, #map18, #map19, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.mulf %in_1, %in_2 : f32
      %2 = arith.addf %out, %0 : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    kernel.yield
  }
  kernel.defn @cublasSaxpby_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32) {
    kernel.yield
  }
  kernel.defn @cudaCopy1D_f32_memref(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>) {
    linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf32, strided<[1], offset: ?>>) outs(%arg1 : memref<?xf32, strided<[1], offset: ?>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    kernel.yield
  }
  kernel.defn @cublasDgemm_outer_product(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map19, #map18, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cublasDaxpby(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: f64, %arg3: f64) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf64>) outs(%arg1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1 = arith.mulf %arg2, %in : f64
      %2 = arith.mulf %arg3, %out : f64
      %3 = arith.addf %1, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasSaxpby(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: f32, %arg3: f32) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cublasSscal(%arg0: tensor<?xf32>, %arg1: f32) -> tensor<?xf32> {
    kernel.yield %arg0 : tensor<?xf32>
  }
  kernel.defn @cublasDaxpy_unit(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf64>) outs(%arg1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1 = arith.addf %out, %in : f64
      linalg.yield %1 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @memset_zero_1D(%arg0: tensor<?xf64>) -> tensor<?xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = linalg.generic {indexing_maps = [#map16], iterator_types = ["parallel"]} outs(%arg0 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @memset_zero_1D_f32(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.generic {indexing_maps = [#map16], iterator_types = ["parallel"]} outs(%arg0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    kernel.yield %0 : tensor<?xf32>
  }
  kernel.defn @memset_zero_2D(%arg0: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%arg0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @memset_zero_2D_f32(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    kernel.yield %0 : tensor<?x?xf32>
  }
  kernel.defn @memset_const_1D(%arg0: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = linalg.generic {indexing_maps = [#map21], iterator_types = ["parallel"]} outs(%arg0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @elemwise_div_scalar(%arg0: tensor<?xf64>, %arg1: f64) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map16], iterator_types = ["parallel"]} outs(%arg0 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %1 = arith.divf %out, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @reduce_sum_axis(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf64>) outs(%arg1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1 = arith.addf %out, %in : f64
      linalg.yield %1 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasDsyrk(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: f64, %arg3: f64) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map20], iterator_types = ["parallel", "parallel"]} outs(%arg1 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = affine.apply #map22(%2)
      %5 = arith.cmpi slt, %3, %4 : index
      %6 = arith.mulf %out, %arg2 : f64
      %7 = arith.select %5, %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?x?xf64>
    %1 = linalg.generic {indexing_maps = [#map9, #map10, #map23], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg0 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%0 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = linalg.index 0 : index
      %3 = linalg.index 2 : index
      %4 = arith.mulf %arg3, %in : f64
      %5 = arith.mulf %4, %in_0 : f64
      %6 = arith.addf %out, %5 : f64
      %7 = affine.apply #map22(%2)
      %8 = arith.cmpi slt, %3, %7 : index
      %9 = arith.select %8, %6, %out : f64
      linalg.yield %9 : f64
    } -> tensor<?x?xf64>
    kernel.yield %1 : tensor<?x?xf64>
  }
  kernel.defn @cublasDsyr2k(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: f64, %arg4: f64) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map20], iterator_types = ["parallel", "parallel"]} outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = affine.apply #map22(%2)
      %5 = arith.cmpi slt, %3, %4 : index
      %6 = arith.mulf %out, %arg3 : f64
      %7 = arith.select %5, %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?x?xf64>
    %1 = linalg.generic {indexing_maps = [#map10, #map9, #map10, #map9, #map23], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1, %arg1, %arg0 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%0 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %2 = linalg.index 0 : index
      %3 = linalg.index 2 : index
      %4 = arith.mulf %in, %arg4 : f64
      %5 = arith.mulf %4, %in_0 : f64
      %6 = arith.mulf %in_1, %arg4 : f64
      %7 = arith.mulf %6, %in_2 : f64
      %8 = arith.addf %5, %7 : f64
      %9 = arith.addf %out, %8 : f64
      %10 = affine.apply #map22(%2)
      %11 = arith.cmpi slt, %3, %10 : index
      %12 = arith.select %11, %9, %out : f64
      linalg.yield %12 : f64
    } -> tensor<?x?xf64>
    kernel.yield %1 : tensor<?x?xf64>
  }
  kernel.defn @cublasSsyrk(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: f32, %arg3: f32) -> tensor<?x?xf32> {
    kernel.yield %arg1 : tensor<?x?xf32>
  }
  kernel.defn @cublasSsyr2k(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: f32, %arg4: f32) -> tensor<?x?xf32> {
    kernel.yield %arg2 : tensor<?x?xf32>
  }
  kernel.defn @jacobi_1d_3pt(%arg0: memref<?xf64, strided<[1]>>, %arg1: memref<?xf64, strided<[1], offset: 1>>, %arg2: memref<?xf64, strided<[1], offset: 2>>, %arg3: memref<?xf64, strided<[1], offset: 1>>) {
    %cst = arith.constant 0.33333333333333331 : f64
    linalg.generic {indexing_maps = [#map16, #map16, #map16, #map16], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1], offset: 1>>, memref<?xf64, strided<[1], offset: 2>>) outs(%arg3 : memref<?xf64, strided<[1], offset: 1>>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %0 = arith.addf %in, %in_0 : f64
      %1 = arith.addf %0, %in_1 : f64
      %2 = arith.mulf %1, %cst : f64
      linalg.yield %2 : f64
    }
    kernel.yield
  }
  kernel.defn @jacobi_2d_5pt(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %cst = arith.constant 2.000000e-01 : f64
    linalg.generic {indexing_maps = [#map20, #map20, #map20, #map20, #map20, #map20], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4 : memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%arg5 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %0 = arith.addf %in, %in_0 : f64
      %1 = arith.addf %0, %in_1 : f64
      %2 = arith.addf %1, %in_2 : f64
      %3 = arith.addf %2, %in_3 : f64
      %4 = arith.mulf %3, %cst : f64
      linalg.yield %4 : f64
    }
    kernel.yield
  }
  kernel.defn @heat_3d_7pt(%arg0: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg1: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg2: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg3: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg4: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg5: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg6: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, %arg7: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    %cst = arith.constant 1.250000e-01 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    linalg.generic {indexing_maps = [#map14, #map14, #map14, #map14, #map14, #map14, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) outs(%arg7 : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %0 = arith.mulf %in_1, %cst_0 : f64
      %1 = arith.subf %in, %0 : f64
      %2 = arith.addf %1, %in_2 : f64
      %3 = arith.mulf %2, %cst : f64
      %4 = arith.subf %in_3, %0 : f64
      %5 = arith.addf %4, %in_4 : f64
      %6 = arith.mulf %5, %cst : f64
      %7 = arith.subf %in_5, %0 : f64
      %8 = arith.addf %7, %in_6 : f64
      %9 = arith.mulf %8, %cst : f64
      %10 = arith.addf %3, %6 : f64
      %11 = arith.addf %10, %9 : f64
      %12 = arith.addf %11, %in_1 : f64
      linalg.yield %12 : f64
    }
    kernel.yield
  }
  kernel.defn @fdtd_update_2in(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %cst = arith.constant 5.000000e-01 : f64
    linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%arg2 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %0 = arith.subf %in, %in_0 : f64
      %1 = arith.mulf %0, %cst : f64
      %2 = arith.subf %out, %1 : f64
      linalg.yield %2 : f64
    }
    kernel.yield
  }
  kernel.defn @fdtd_E_update(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %cst = arith.constant 0.69999999999999984 : f64
    linalg.generic {indexing_maps = [#map7, #map7, #map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%arg4 : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %0 = arith.subf %in, %in_0 : f64
      %1 = arith.addf %0, %in_1 : f64
      %2 = arith.subf %1, %in_2 : f64
      %3 = arith.mulf %2, %cst : f64
      %4 = arith.subf %out, %3 : f64
      linalg.yield %4 : f64
    }
    kernel.yield
  }
  kernel.defn @broadcast_scalar_to_vec(%arg0: memref<f64, strided<[], offset: ?>>, %arg1: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {indexing_maps = [#map17, #map16], iterator_types = ["parallel"]} ins(%arg0 : memref<f64, strided<[], offset: ?>>) outs(%arg1 : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    }
    kernel.yield
  }
  kernel.defn @cublasDcopy(%arg0: memref<?xf64, strided<[1]>>, %arg1: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf64, strided<[1]>>) outs(%arg1 : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    }
    kernel.yield
  }
  kernel.defn @centered_sum_squares(%arg0: tensor<?x?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map7, #map18, #map18], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.subf %in, %in_0 : f64
      %2 = arith.mulf %1, %1 : f64
      %3 = arith.addf %out, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @jacobi_1d_3pt_tensor(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>, %arg3: tensor<?xf64>) -> tensor<?xf64> {
    %cst = arith.constant 0.33333333333333331 : f64
    %0 = linalg.generic {indexing_maps = [#map16, #map16, #map16, #map16], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%arg3 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %1 = arith.addf %in, %in_0 : f64
      %2 = arith.addf %1, %in_1 : f64
      %3 = arith.mulf %2, %cst : f64
      linalg.yield %3 : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @jacobi_2d_5pt_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: tensor<?x?xf64>, %arg4: tensor<?x?xf64>, %arg5: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 2.000000e-01 : f64
    %0 = linalg.generic {indexing_maps = [#map7, #map7, #map7, #map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg5 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %1 = arith.addf %in, %in_0 : f64
      %2 = arith.addf %1, %in_1 : f64
      %3 = arith.addf %2, %in_2 : f64
      %4 = arith.addf %3, %in_3 : f64
      %5 = arith.mulf %4, %cst : f64
      linalg.yield %5 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @heat_3d_7pt_tensor(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>, %arg3: tensor<?x?x?xf64>, %arg4: tensor<?x?x?xf64>, %arg5: tensor<?x?x?xf64>, %arg6: tensor<?x?x?xf64>, %arg7: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %cst = arith.constant 1.250000e-01 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %0 = linalg.generic {indexing_maps = [#map14, #map14, #map14, #map14, #map14, #map14, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%arg7 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %1 = arith.mulf %in_1, %cst_0 : f64
      %2 = arith.subf %in, %1 : f64
      %3 = arith.addf %2, %in_2 : f64
      %4 = arith.mulf %3, %cst : f64
      %5 = arith.subf %in_3, %1 : f64
      %6 = arith.addf %5, %in_4 : f64
      %7 = arith.mulf %6, %cst : f64
      %8 = arith.subf %in_5, %1 : f64
      %9 = arith.addf %8, %in_6 : f64
      %10 = arith.mulf %9, %cst : f64
      %11 = arith.addf %4, %7 : f64
      %12 = arith.addf %11, %10 : f64
      %13 = arith.addf %12, %in_1 : f64
      linalg.yield %13 : f64
    } -> tensor<?x?x?xf64>
    kernel.yield %0 : tensor<?x?x?xf64>
  }
  kernel.defn @fdtd_update_2in_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 5.000000e-01 : f64
    %0 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.subf %in, %in_0 : f64
      %2 = arith.mulf %1, %cst : f64
      %3 = arith.subf %out, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @broadcast_scalar_to_vec_tensor(%arg0: tensor<f64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map17, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<f64>) outs(%arg1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @cublasDcopy_tensor(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic {indexing_maps = [#map16, #map16], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf64>) outs(%arg1 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?xf64>
    kernel.yield %0 : tensor<?xf64>
  }
  kernel.defn @fdtd_E_update_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: tensor<?x?xf64>, %arg4: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 0.69999999999999984 : f64
    %0 = linalg.generic {indexing_maps = [#map7, #map7, #map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg4 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %1 = arith.subf %in, %in_0 : f64
      %2 = arith.addf %1, %in_1 : f64
      %3 = arith.subf %2, %in_2 : f64
      %4 = arith.mulf %3, %cst : f64
      %5 = arith.subf %out, %4 : f64
      linalg.yield %5 : f64
    } -> tensor<?x?xf64>
    kernel.yield %0 : tensor<?x?xf64>
  }
  kernel.defn @cudnnConvolution2D_9tap(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_9tap_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>, %arg3: tensor<?x?xf64>, %arg4: tensor<?x?xf64>, %arg5: tensor<?x?xf64>, %arg6: tensor<?x?xf64>, %arg7: tensor<?x?xf64>, %arg8: tensor<?x?xf64>, %arg9: tensor<?x?xf64>, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64) -> tensor<?x?xf64> {
    kernel.yield %arg9 : tensor<?x?xf64>
  }
  kernel.defn @cudnnConvolution2D_9tap_f32(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg10: f32, %arg11: f32, %arg12: f32, %arg13: f32, %arg14: f32, %arg15: f32, %arg16: f32, %arg17: f32, %arg18: f32) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_25tap(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg10: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg11: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg12: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg13: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg14: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg15: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg16: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg17: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg18: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg19: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg20: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg21: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg22: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg23: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg24: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg25: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg26: f64, %arg27: f64, %arg28: f64, %arg29: f64, %arg30: f64, %arg31: f64, %arg32: f64, %arg33: f64, %arg34: f64, %arg35: f64, %arg36: f64, %arg37: f64, %arg38: f64, %arg39: f64, %arg40: f64, %arg41: f64, %arg42: f64, %arg43: f64, %arg44: f64, %arg45: f64, %arg46: f64, %arg47: f64, %arg48: f64, %arg49: f64, %arg50: f64) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_25tap_f32(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg10: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg11: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg12: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg13: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg14: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg15: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg16: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg17: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg18: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg19: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg20: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg21: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg22: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg23: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg24: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg25: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg26: f32, %arg27: f32, %arg28: f32, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: f32, %arg33: f32, %arg34: f32, %arg35: f32, %arg36: f32, %arg37: f32, %arg38: f32, %arg39: f32, %arg40: f32, %arg41: f32, %arg42: f32, %arg43: f32, %arg44: f32, %arg45: f32, %arg46: f32, %arg47: f32, %arg48: f32, %arg49: f32, %arg50: f32) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_ntap(%arg0: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg2: memref<?xf64>, %arg3: i32) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_ntap_f32(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg2: memref<?xf32>, %arg3: i32) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_ntap_tensor(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?xf64>, %arg3: i32) -> tensor<?x?xf64> {
    kernel.yield %arg1 : tensor<?x?xf64>
  }
  kernel.defn @cudnnConvolution2D_ntap_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>, %arg3: i32) -> tensor<?x?xf32> {
    kernel.yield %arg1 : tensor<?x?xf32>
  }
  kernel.defn @cudnnConvolution3D_ntap_tensor(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>, %arg3: i32) -> tensor<?x?x?xf64> {
    kernel.yield %arg1 : tensor<?x?x?xf64>
  }
  kernel.defn @cudnnConvolution3D_ntap_f32_tensor(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: i32) -> tensor<?x?x?xf32> {
    kernel.yield %arg1 : tensor<?x?x?xf32>
  }
  kernel.defn @cudnnStencil3DSymmetric_f64_memref(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32) {
    kernel.yield
  }
  kernel.defn @cudnnStencil3D7pt_f32_flat_tensor(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: f32, %arg3: f32, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) -> tensor<?xf32> {
    kernel.yield %arg1 : tensor<?xf32>
  }
  kernel.defn @cudnnConvolution3D_f32(%arg0: tensor<?x?x?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg2 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolution3D_f32_bias(%arg0: tensor<?x?x?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %arg3 : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolution1D_f32_bias(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %arg3 : tensor<?x?x?xf32>
  }
  kernel.defn @cudnnConvolution1D_f32_bias_expanded(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
    kernel.yield %arg3 : tensor<?x?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolution2D_f32_dilated(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %arg2 : tensor<?x?x?xf32>
  }
  kernel.defn @cufftZ2Z_1D_tensor(%arg0: tensor<?x2xf64>, %arg1: tensor<?x2xf64>, %arg2: i32) -> tensor<?x2xf64> {
    kernel.yield %arg1 : tensor<?x2xf64>
  }
  kernel.defn @cufftC2C_1D_tensor(%arg0: tensor<?x2xf32>, %arg1: tensor<?x2xf32>, %arg2: i32) -> tensor<?x2xf32> {
    kernel.yield %arg1 : tensor<?x2xf32>
  }
  kernel.defn @cutensornetTensorProduct3D_f32_tensor(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?x?xf32>, %arg3: tensor<?x?x?x?x?x?xf32>, %arg4: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    kernel.yield %arg4 : tensor<?x?x?x?x?x?xf32>
  }
  kernel.defn @cutensornetTensorProduct3D_f64_tensor(%arg0: tensor<?x?x?x?x?x?xf64>, %arg1: tensor<?x?x?x?x?x?xf64>, %arg2: tensor<?x?x?x?x?x?xf64>, %arg3: tensor<?x?x?x?x?x?xf64>, %arg4: tensor<?x?x?x?x?x?xf64>) -> tensor<?x?x?x?x?x?xf64> {
    kernel.yield %arg4 : tensor<?x?x?x?x?x?xf64>
  }
  kernel.defn @cutensornetContraction2_f64(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>, %arg2: tensor<*xf64>) -> tensor<*xf64> {
    kernel.yield %arg2 : tensor<*xf64>
  }
  kernel.defn @cutensornetContraction2_f64_r4r5r4(%arg0: tensor<?x?x?x?xf64>, %arg1: tensor<?x?x?x?x?xf64>, %arg2: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %arg2 : tensor<?x?x?x?xf64>
  }
  kernel.defn @cutensornetContraction2_f64_r5r4r4(%arg0: tensor<?x?x?x?x?xf64>, %arg1: tensor<?x?x?x?xf64>, %arg2: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %arg2 : tensor<?x?x?x?xf64>
  }
  kernel.defn @cutensornetContraction2_f64_r5r5r4(%arg0: tensor<?x?x?x?x?xf64>, %arg1: tensor<?x?x?x?x?xf64>, %arg2: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %arg2 : tensor<?x?x?x?xf64>
  }
  kernel.defn @cudnnPointwiseAffineRelu_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: f32) -> tensor<?xf32> {
    kernel.yield %arg2 : tensor<?xf32>
  }
  kernel.defn @cudnnPointwiseGraph_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32, %arg12: f32) -> tensor<?xf32> {
    kernel.yield %arg4 : tensor<?xf32>
  }
  kernel.defn @cubInclusiveSum1D_f32_tensor(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<?xf32>) -> (tensor<f32>, tensor<?xf32>) {
    kernel.yield %arg1, %arg2 : tensor<f32>, tensor<?xf32>
  }
  kernel.defn @cubSegmentedInclusiveProduct2D_f32_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) {
    kernel.yield %arg1, %arg2 : tensor<?x?xf32>, tensor<?xf32>
  }
  kernel.defn @cubExclusiveSum1D_i32_memref(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_9tap_f16(%arg0: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg10: f16, %arg11: f16, %arg12: f16, %arg13: f16, %arg14: f16, %arg15: f16, %arg16: f16, %arg17: f16, %arg18: f16) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_9tap_bf16(%arg0: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xbf16, strided<[?, 1], offset: ?>>, %arg10: bf16, %arg11: bf16, %arg12: bf16, %arg13: bf16, %arg14: bf16, %arg15: bf16, %arg16: bf16, %arg17: bf16, %arg18: bf16) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_9tap_i32(%arg0: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xi32, strided<[?, 1], offset: ?>>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) {
    kernel.yield
  }
  kernel.defn @cudnnConvolution2D_9tap_i16(%arg0: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg3: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg4: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg5: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg6: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg7: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg8: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg9: memref<?x?xi16, strided<[?, 1], offset: ?>>, %arg10: i16, %arg11: i16, %arg12: i16, %arg13: i16, %arg14: i16, %arg15: i16, %arg16: i16, %arg17: i16, %arg18: i16) {
    kernel.yield
  }
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg7 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg10 : memref<?x?xf64>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg0 : i32 to index
    %9 = linalg.generic {doc = "", indexing_maps = [#map7], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%8, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%5, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %9[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_1 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %15 = arith.mulf %arg4, %in : f64
      %16 = arith.mulf %15, %in_5 : f64
      %17 = arith.addf %out, %16 : f64
      linalg.yield %17 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %11 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %11, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map7], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%4 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %15 = arith.mulf %out, %arg5 : f64
      linalg.yield %15 : f64
    } -> tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %3[0, 0] [%7, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %12[0, 0] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%10, %extracted_slice_2 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_3 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %15 = arith.mulf %in, %in_5 : f64
      %16 = arith.addf %out, %15 : f64
      linalg.yield %16 : f64
    } -> tensor<?x?xf64>
    %inserted_slice_4 = tensor.insert_slice %13 into %12[0, 0] [%8, %6] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %14 = bufferization.to_memref %inserted_slice_4 : memref<?x?xf64>
    memref.copy %14, %arg10 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2lf \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("D\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c800 = arith.constant 800 : index
    %cst = arith.constant 1.100000e+03 : f64
    %cst_0 = arith.constant 1.200000e+03 : f64
    %cst_1 = arith.constant 9.000000e+02 : f64
    %cst_2 = arith.constant 8.000000e+02 : f64
    %c20 = arith.constant 20 : index
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant 1.500000e+00 : f64
    %cst_5 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1100_i32 = arith.constant 1100 : i32
    %c900_i32 = arith.constant 900 : i32
    %c800_i32 = arith.constant 800 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<800x900xf64>
    %alloc_6 = memref.alloc() : memref<800x1100xf64>
    %alloc_7 = memref.alloc() : memref<1100x900xf64>
    %alloc_8 = memref.alloc() : memref<900x1200xf64>
    %alloc_9 = memref.alloc() : memref<800x1200xf64>
    %0 = "polygeist.subindex"(%alloc_6, %c0) : (memref<800x1100xf64>, index) -> memref<?xf64>
    %1 = "polygeist.memref2pointer"(%0) : (memref<?xf64>) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x?xf64>
    %3 = "polygeist.subindex"(%alloc_7, %c0) : (memref<1100x900xf64>, index) -> memref<?xf64>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?xf64>) -> !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?x?xf64>
    %6 = "polygeist.subindex"(%alloc_8, %c0) : (memref<900x1200xf64>, index) -> memref<?xf64>
    %7 = "polygeist.memref2pointer"(%6) : (memref<?xf64>) -> !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x?xf64>
    %9 = "polygeist.subindex"(%alloc_9, %c0) : (memref<800x1200xf64>, index) -> memref<?xf64>
    %10 = "polygeist.memref2pointer"(%9) : (memref<?xf64>) -> !llvm.ptr
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?x?xf64>
    affine.for %arg2 = 0 to 800 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1100 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.muli %38, %39 : i32
        %41 = arith.addi %40, %c1_i32 : i32
        %42 = arith.remsi %41, %c800_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst_2 : f64
        affine.store %44, %2[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 1100 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 900 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c1_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.remsi %41, %c900_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst_1 : f64
        affine.store %44, %5[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 900 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c3_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.addi %41, %c1_i32 : i32
        %43 = arith.remsi %42, %c1200_i32 : i32
        %44 = arith.sitofp %43 : i32 to f64
        %45 = arith.divf %44, %cst_0 : f64
        affine.store %45, %8[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 800 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c2_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.remsi %41, %c1100_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst : f64
        affine.store %44, %11[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    %12 = "polygeist.subindex"(%alloc, %c0) : (memref<800x900xf64>, index) -> memref<?xf64>
    %13 = "polygeist.memref2pointer"(%12) : (memref<?xf64>) -> !llvm.ptr
    %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?x?xf64>
    affine.for %arg2 = 0 to 800 {
      affine.for %arg3 = 0 to 900 {
        affine.store %cst_3, %14[%arg2, %arg3] : memref<?x?xf64>
        affine.for %arg4 = 0 to 1100 {
          %38 = affine.load %2[%arg2, %arg4] : memref<?x?xf64>
          %39 = arith.mulf %38, %cst_4 : f64
          %40 = affine.load %5[%arg4, %arg3] : memref<?x?xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = affine.load %14[%arg2, %arg3] : memref<?x?xf64>
          %43 = arith.addf %42, %41 : f64
          affine.store %43, %14[%arg2, %arg3] : memref<?x?xf64>
        }
      }
    }
    affine.for %arg2 = 0 to 800 {
      affine.for %arg3 = 0 to 1200 {
        %38 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
        %39 = arith.mulf %38, %cst_5 : f64
        affine.store %39, %11[%arg2, %arg3] : memref<?x?xf64>
        affine.for %arg4 = 0 to 900 {
          %40 = affine.load %14[%arg2, %arg4] : memref<?x?xf64>
          %41 = affine.load %8[%arg4, %arg3] : memref<?x?xf64>
          %42 = arith.mulf %40, %41 : f64
          %43 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
          %44 = arith.addf %43, %42 : f64
          affine.store %44, %11[%arg2, %arg3] : memref<?x?xf64>
        }
      }
    }
    %15 = llvm.mlir.addressof @stderr : !llvm.ptr
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %17 = llvm.mlir.addressof @str0 : !llvm.ptr
    %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %20 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.mlir.addressof @str1 : !llvm.ptr
    %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %23 = llvm.mlir.addressof @str2 : !llvm.ptr
    %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %25 = llvm.call @fprintf(%20, %22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %26 = llvm.mlir.addressof @str4 : !llvm.ptr
    %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %28 = llvm.mlir.addressof @str3 : !llvm.ptr
    %29 = llvm.getelementptr %28[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    affine.for %arg2 = 0 to 800 {
      %38 = arith.muli %arg2, %c800 : index
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.addi %arg3, %38 : index
        %40 = arith.remsi %39, %c20 : index
        %41 = arith.cmpi slt, %40, %c0 : index
        %42 = arith.addi %40, %c20 : index
        %43 = arith.select %41, %42, %40 : index
        %44 = arith.cmpi eq, %43, %c0 : index
        scf.if %44 {
          %48 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
          %49 = llvm.call @fprintf(%48, %29) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        %45 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
        %46 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
        %47 = llvm.call @fprintf(%45, %27, %46) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
      }
    }
    %30 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %31 = llvm.mlir.addressof @str5 : !llvm.ptr
    %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %33 = llvm.call @fprintf(%30, %32, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %34 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %35 = llvm.mlir.addressof @str6 : !llvm.ptr
    %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %37 = llvm.call @fprintf(%34, %36) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    memref.dealloc %alloc : memref<800x900xf64>
    memref.dealloc %alloc_6 : memref<800x1100xf64>
    memref.dealloc %alloc_7 : memref<1100x900xf64>
    memref.dealloc %alloc_8 : memref<900x1200xf64>
    memref.dealloc %alloc_9 : memref<800x1200xf64>
    return %c0_i32 : i32
  }
}

