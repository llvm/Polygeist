// Phase-2 kernel library — canonical linalg implementations for each library
// symbol the kernel matcher emits. The --lower-kernel-launch pass loads this
// file (via kernel-library-path=) and substitutes each kernel.defn's body
// in place of its matching kernel.launch op.
//
// Conventions:
//   - All bodies operate on `f64` tensors. The PolyBench corpus is double-only.
//   - Operand order matches what kernel_match_rewrite.py emits:
//     all tensor inputs (in source order) + first generic's outs + scalars.
//   - Each defn's linalg.generic uses *self-contained* indexing_maps and
//     iterator_types; it operates on whatever shape the launch's operands
//     have at the call site, without referring to any caller context.
//
// To add a new library entry: pick a unique kernel.launch signature observed
// in `kernel_match_rewrite.py` output and author a kernel.defn with that
// signature whose body computes the canonical semantics for that library op.

module {
  // Device-wide integer histogram. The executable matcher selects this
  // overwrite form only when it also proves a zero-initialized destination.
  kernel.defn @cubHistogramEvenI32ShiftZero_memref(
      %samples: memref<?xi32>, %histogram: memref<?xi32>, %right_shift: i32) {
    kernel.yield
  }

  // Whole-algorithm dense linear algebra recovered from row-major C loops.
  kernel.defn @cublasDtrsvLowerRowMajor_memref(
      %A: memref<?x?xf64>, %b: memref<?xf64>, %x: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @cusolverDnDpotrfLowerRowMajor_memref(
      %A: memref<?x?xf64>) { kernel.yield }

  // NVIDIA cuSPARSE generic-API CSR SpMV. These are ABI contracts: the
  // matcher proves the row-pointer traversal and multiply-add reduction, and
  // the CUDA runtime invokes cusparseSpMV with alpha=1 and beta=0.
  kernel.defn @cusparseSpMV_CSR_f32_memref(
      %rows: index, %row_offsets: memref<?xi32>,
      %column_indices: memref<?xi32>, %values: memref<?xf32>,
      %x: memref<?xf32>, %y: memref<?xf32>) { kernel.yield }
  kernel.defn @cusparseSpMV_CSR_f64_memref(
      %rows: index, %row_offsets: memref<?xi32>,
      %column_indices: memref<?xi32>, %values: memref<?xf64>,
      %x: memref<?xf64>, %y: memref<?xf64>) { kernel.yield }

  // cuSten's compiled 2D XY non-periodic weighted-stencil API. The packed
  // KxK weights and valid-region layout are the same operands used by the
  // existing generalized convolution matcher; only f64 is exposed upstream.
  kernel.defn @custenStencil2DXY_f64_memref(
      %input: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %output: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %weights: memref<?xf64>, %K: i32) { kernel.yield }
  kernel.defn @custenStencil2DXY_f64_tensor(
      %input: tensor<?x?xf64>, %output: tensor<?x?xf64>,
      %weights: tensor<?xf64>, %K: i32) -> tensor<?x?xf64> {
    kernel.yield %output : tensor<?x?xf64>
  }

  kernel.defn @cublasGemmEx_i8_i32_tensor(
      %A: tensor<?x?xi8>, %B: tensor<?x?xi8>, %C: tensor<?x?xi32>)
      -> tensor<?x?xi32> { kernel.yield %C : tensor<?x?xi32> }
  kernel.defn @cublasSnrm2_f32_memref(
      %input: memref<?xf32>, %output: memref<?xf32>) { kernel.yield }
  kernel.defn @cublasJointMaxAbsProduct_f32_memref(
      %a: memref<?xf32>, %b: memref<?xf32>, %output: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnFeatureMaskScale_f32_tensor(
      %input: tensor<?x?x?x?xf32>, %mask: tensor<?x?xf32>, %scale: f32,
      %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %output : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolutionTranspose2D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>,
      %output: memref<?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionTranspose3D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>,
      %output: memref<?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionBackwardFilter3D_f32_memref(
      %input: memref<?x?x?x?xf32>, %gradient: memref<?x?x?x?xf32>,
      %filter: memref<?x?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnDepthwiseConvolution2D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %bias: memref<?xf32>, %output: memref<?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cutensorKroneckerProduct2D_f32_memref(
      %x: memref<?x?xf32>, %y: memref<?x?xf32>,
      %output: memref<?x?xf32>) { kernel.yield }
  kernel.defn @cudnnBinaryCrossEntropyMean_f32_memref(
      %input: memref<?xf32>, %target: memref<?xf32>,
      %output: memref<?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionTBC_f32_memref(
      %input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %output: memref<?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionTBCBackward_f32_memref(
      %gradient: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %output: memref<?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnTransformBiasRescaleQKV_f32_memref(
      %qkv: memref<?x?x?x?x?xf32>, %bias: memref<?x?x?xf32>, %scale: f32,
      %q: memref<?x?x?x?xf32>, %k: memref<?x?x?x?xf32>,
      %v: memref<?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnAddrElementwise_f32_memref(
      %self: memref<?xf32>, %x: memref<?xf32>, %y: memref<?xf32>,
      %beta: f32, %alpha: f32, %output: memref<?xf32>) { kernel.yield }
  kernel.defn @cudnnLogSigmoid_f32_memref(
      %x: memref<?xf32>, %output: memref<?xf32>,
      %buffer: memref<?xf32>) { kernel.yield }
  kernel.defn @cubSegmentedLogicalAnd_i32_memref(
      %x: memref<?x64xi32>, %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cubSegmentedLogicalSelect_i32_memref(
      %all_x: memref<?x64xi32>, %any_x: memref<?x64xi32>, %all: i32,
      %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cublasSdot_memref(
      %x: memref<?xf32>, %y: memref<?xf32>,
      %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cublasDdot_memref(
      %x: memref<?xf64>, %y: memref<?xf64>,
      %out: memref<?xf64>) { kernel.yield }
  kernel.defn @cubSegmentedArgMax_f32_i32_memref(
      %x: memref<?x64xf32>, %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cubSegmentedArgMin_f32_i32_memref(
      %x: memref<?x64xf32>, %out: memref<?xi32>) { kernel.yield }
  kernel.defn @cublasSgemvTZero_memref(
      %matrix: memref<?x?xf32>, %vector: memref<?xf32>,
      %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cubSegmentedSum_f32_memref(
      %input: memref<?x?xf32>, %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cubSegmentedSum_f64_memref(
      %input: memref<?x?xf64>, %out: memref<?xf64>) { kernel.yield }
  kernel.defn @cubSegmentedMin_f32_memref(
      %input: memref<?x?xf32>, %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cubSegmentedMax_f32_memref(
      %input: memref<?x?xf32>, %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cutensornetNetwork_f32_n3_aten(
      %lhs: memref<?x?xf32>, %weights: memref<?x?x?xf32>,
      %rhs: memref<?x?xf32>, %out: memref<?x?xf32>) { kernel.yield }
  kernel.defn @cudnnSinc_f32_memref(
      %x: memref<?xf32>, %out: memref<?xf32>) { kernel.yield }

  // ABI-only declarations for general contiguous reductions.  The reduction
  // combiner and element type are encoded by the symbol; the CUDA ABI pass
  // lowers these to cuDNN's tensor-reduction API.
  kernel.defn @cudnnReduceSum_f32(%x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> { kernel.yield %out : tensor<f32> }
  kernel.defn @cudnnReduceSum_f64(%x: tensor<?xf64>, %out: tensor<f64>) -> tensor<f64> { kernel.yield %out : tensor<f64> }
  kernel.defn @cudnnReduceProduct_f32(%x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> { kernel.yield %out : tensor<f32> }
  kernel.defn @cudnnReduceMin_f32(%x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> { kernel.yield %out : tensor<f32> }
  kernel.defn @cudnnReduceMax_f32(%x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> { kernel.yield %out : tensor<f32> }
  kernel.defn @cudnnReduceMinMax_f32(%x: tensor<?xf32>, %max: tensor<f32>, %min: tensor<f32>) -> (tensor<f32>, tensor<f32>) { kernel.yield %max, %min : tensor<f32>, tensor<f32> }
  kernel.defn @cudnnReduceTrace_f32(%x: tensor<?x?xf32>, %out: tensor<f32>) -> tensor<f32> { kernel.yield %out : tensor<f32> }
  kernel.defn @cubSegmentedLogicalAnd_i32(%x: tensor<?x?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> { kernel.yield %out : tensor<?xi32> }
  kernel.defn @cubSegmentedLogicalOr_i32(%x: tensor<?x?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> { kernel.yield %out : tensor<?xi32> }
  kernel.defn @cubSegmentedBitXor_i32(%x: tensor<?x?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> { kernel.yield %out : tensor<?xi32> }
  kernel.defn @cubSegmentedPrefixSum_f32(%x: tensor<?x?xf32>, %lengths: tensor<?xi32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cubSegmentedPrefixLogicalAnd_i32(%x: tensor<?x?xi32>, %lengths: tensor<?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> { kernel.yield %out : tensor<?xi32> }
  kernel.defn @cutensorPermute_f32_r2_tensor(%input: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> { kernel.yield %out : tensor<?x?xf32> }
  kernel.defn @cutensorPermute_f32_r3_tensor(%input: tensor<?x?x?xf32>, %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> { kernel.yield %out : tensor<?x?x?xf32> }
  kernel.defn @cutensorPermute_f32_r4_tensor(%input: tensor<?x?x?x?xf32>, %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> { kernel.yield %out : tensor<?x?x?x?xf32> }
  kernel.defn @cutensorPermute_f32_r5_tensor(%input: tensor<?x?x?x?x?xf32>, %out: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> { kernel.yield %out : tensor<?x?x?x?x?xf32> }
  kernel.defn @cutensorPermute_f32_r6_tensor(%input: tensor<?x?x?x?x?x?xf32>, %out: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> { kernel.yield %out : tensor<?x?x?x?x?x?xf32> }

  // ABI-only canonical declarations for the parameterized cuTENSOR unary
  // lowering. The runtime operation id is encoded in the symbol name by the
  // matcher and materialized by --lower-kernel-launch-to-cublas. These bodies
  // intentionally preserve the destination if the ABI pass is not selected.
  kernel.defn @cutensorUnary_abs_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_acos_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_acosh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_asin_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_asinh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_atan_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_atanh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_ceil_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_cos_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_cosh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_exp_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_floor_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_log_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_mish_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_neg_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_reciprocal_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_relu_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_sigmoid_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_silu_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_sin_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_sinh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_sqrt_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_tan_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }
  kernel.defn @cutensorUnary_tanh_f32(%x: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> { kernel.yield %out : tensor<?xf32> }

  // cuDNN tensor add: output += input (NCHW).
  kernel.defn @cudnnAddTensor_batched(
      %input: tensor<?x?x?x?xf32>,
      %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x?x?x?xf32>)
      outs(%output : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %out, %in : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %result : tensor<?x?x?x?xf32>
  }

  // cuDNN inference batch normalization with caller-provided reciprocal stddev.
  kernel.defn @cudnnBatchNormalizationForwardInference(
      %input: tensor<?x?x?x?xf32>, %weight: tensor<?xf32>,
      %mean: tensor<?xf32>, %inv_std: tensor<?xf32>,
      %bias: tensor<?xf32>, %output: tensor<?x?x?x?xf32>)
      -> tensor<?x?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %weight, %mean, %inv_std, %bias
          : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>,
            tensor<?xf32>, tensor<?xf32>)
      outs(%output : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %w: f32, %m: f32, %is: f32, %b: f32, %out: f32):
      %centered = arith.subf %in, %m : f32
      %scaled0 = arith.mulf %centered, %is : f32
      %scaled1 = arith.mulf %w, %scaled0 : f32
      %value = arith.addf %scaled1, %b : f32
      linalg.yield %value : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %result : tensor<?x?x?x?xf32>
  }

  // cuDNN valid NCHW convolution. The first operand is the rank-7 window
  // view produced by polygeist.submap: [N, OC, OH, OW, IC, KH, KW].
  kernel.defn @cudnnConvolutionFwd_batched(
      %windows: tensor<?x?x?x?x?x?x?xf32>,
      %filter: tensor<?x?x?x?xf32>,
      %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) ->
                   (d0, d1, d2, d3, d4, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]
    } ins(%windows, %filter
          : tensor<?x?x?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%output : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %f: f32, %out: f32):
      %product = arith.mulf %in, %f : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %result : tensor<?x?x?x?xf32>
  }

  // Uniform-weight channel-preserving fixed-window convolution.  This is the
  // generic library form for box stencils and regular adaptive-average-pool
  // specializations.  The runtime constructs a [C,1,KH,KW] filter and selects
  // grouped/depthwise cuDNN convolution with groups=C.
  kernel.defn @cudnnConvolution2DWindow_f32(
      %input: tensor<?x?x?x?xf32>,
      %output: tensor<?x?x?x?xf32>,
      %weight: f32,
      %kh: i32, %kw: i32, %sh: i32, %sw: i32,
      %dh: i32, %dw: i32, %ph: i32, %pw: i32)
      -> tensor<?x?x?x?xf32> {
    kernel.yield %output : tensor<?x?x?x?xf32>
  }

  // Tensor-form average pooling. Same signature as the window convolution
  // above, but the matcher emits this symbol only when the weight is exactly
  // 1/(KH*KW) over a non-overlapping valid window (i.e. a box average), so the
  // ABI lowering runs cuDNN average pooling instead of a depthwise convolution.
  kernel.defn @cudnnAvgPoolWindow_f32(
      %input: tensor<?x?x?x?xf32>,
      %output: tensor<?x?x?x?xf32>,
      %weight: f32,
      %kh: i32, %kw: i32, %sh: i32, %sw: i32,
      %dh: i32, %dw: i32, %ph: i32, %pw: i32)
      -> tensor<?x?x?x?xf32> {
    kernel.yield %output : tensor<?x?x?x?xf32>
  }

  // Rank-parameterized ATen adaptive average/max pooling. Operation is
  // 0=avg-fwd, 1=avg-bwd, 2=max-fwd, 3=max-bwd. Spatial dimensions unused by
  // rank-1/rank-2 forms are one. ptr2 is unused for average pooling.
  kernel.defn @cudnnAdaptivePool_f32_flat2(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?xf32>, %ptr1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_flat3_fwd(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?xf32>, %ptr1: memref<?xf32>, %ptr2: memref<?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_flat3_bwd(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?xf32>, %ptr1: memref<?xi32>, %ptr2: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r2(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?xf32>, %ptr1: memref<?x?xf32>,
      %ptr2: memref<?x?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r4_fwd(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?x?x?xf32>, %ptr1: memref<?x?x?x?xf32>,
      %ptr2: memref<?x?x?x?xi32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r4_bwd(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?x?x?xf32>, %ptr1: memref<?x?x?x?xi32>,
      %ptr2: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAdaptivePool_f32_r5(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?x?x?x?xf32>, %ptr1: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }

  // Fixed-window average pooling uses the same runtime ABI as adaptive
  // pooling, with operation tags 4 (forward) and 5 (backward).  Keeping
  // distinct semantic symbols prevents fixed odd-size windows from being
  // confused with adaptive partitions.
  kernel.defn @cudnnAveragePool_f32_flat2(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?xf32>, %ptr1: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAveragePool_f32_r4(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?x?x?xf32>, %ptr1: memref<?x?x?x?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnAveragePool_f32_r5(
      %operation: i32, %rank: i32, %n: i32, %c: i32,
      %i0: i32, %i1: i32, %i2: i32,
      %o0: i32, %o1: i32, %o2: i32,
      %ptr0: memref<?x?x?x?x?xf32>, %ptr1: memref<?x?x?x?x?xf32>) {
    kernel.yield
  }

  kernel.defn @cudnnBatchNormBackward_f32_full(
      %n: i32, %c: i32, %spatial: i32,
      %grad: memref<?x?x?xf32>, %x: memref<?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %weight: memref<?xf32>, %dx: memref<?x?x?xf32>,
      %dweight: memref<?xf32>, %dbias: memref<?xf32>) {
    kernel.yield
  }
  kernel.defn @cudnnBatchNormBackward_f32_dx(
      %n: i32, %c: i32, %spatial: i32,
      %grad: memref<?x?x?x?xf32>, %x: memref<?x?x?x?xf32>,
      %mean: memref<?xf32>, %invstd: memref<?xf32>,
      %dx: memref<?x?x?x?xf32>) {
    kernel.yield
  }

  // cuDNN max pool. The rank-6 input is [N, C, OH, OW, KH, KW].
  kernel.defn @cudnnMaxPoolFwd_batched(
      %windows: tensor<?x?x?x?x?x?xf32>,
      %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction"]
    } ins(%windows : tensor<?x?x?x?x?x?xf32>)
      outs(%output : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %take_input = arith.cmpf ogt, %in, %out : f32
      %maximum = arith.select %take_input, %in, %out : f32
      linalg.yield %maximum : f32
    } -> tensor<?x?x?x?xf32>
    kernel.yield %result : tensor<?x?x?x?xf32>
  }

  // GEMM: C = alpha*A*B + beta*C    (standard textbook gemm)
  // Operand order: A, B, C, beta, alpha.
  kernel.defn @cublasDgemm(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                            %C: tensor<?x?xf64>,
                            %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    // Step 1: C = beta * C
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %t = arith.mulf %out, %beta : f64
      linalg.yield %t : f64
    } -> tensor<?x?xf64>
    // Step 2: C = alpha * A * B + C
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMM-SIMPLE: C += A*B (alpha=1, beta=1, accumulate-into-C).
  kernel.defn @cublasDgemm_simple(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                                   %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMM-SUBTRACT: C -= A*B (alpha=-1, beta=1). This spelling occurs in
  // block-tridiagonal elimination after fixed-size scalar code is rerolled.
  kernel.defn @cublasDgemm_subtract(%A: tensor<?x?xf64>,
                                     %B: tensor<?x?xf64>,
                                     %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %s = arith.subf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }
  kernel.defn @cublasDgemm_zero(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                                %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %C : tensor<?x?xf64>
  }

  // The suffix records the physical row-major layout of A and B. Semantic
  // matching proves the indexing maps; ABI lowering supplies transpose flags.
  kernel.defn @cublasSgemm_nn(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                              %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                              %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                              %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                              %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  // ABI-only FP32 variants.  Their scalar semantics are supplied by the
  // dtype-polymorphic Dgemm templates above; these names carry the physical
  // transpose state and scalar mode to the common cuBLAS lowering.
  kernel.defn @cublasSgemm_nn_alpha_beta(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_alpha_beta(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_alpha_beta(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_alpha_beta(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nn_alpha(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_alpha(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_alpha(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_alpha(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
      %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nn_zero(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
      %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_nt_zero(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
      %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tn_zero(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
      %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_tt_zero(
      %A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
      %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }
  kernel.defn @cublasSgemm_strided_batched_nn_zero(
      %A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>,
      %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %C : tensor<?x?x?xf32>
  }

  // FP32 Darknet im2col+GEMM lowered shape. The linalg raiser represents the
  // scalar A[i,k] load as a broadcasted rank-3 input so the output submap can
  // still ignore the reduction dim when lowered back to the flat C buffer.
  kernel.defn @cublasSgemm_broadcast3d_simple(
      %A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>,
      %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%C : tensor<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %result : tensor<?x?x?xf32>
  }

  // Parboil's basic SGEMM is raised as (M,N,K) broadcast views over flat,
  // column-major buffers.  The semantic body is still cublasDgemm above;
  // this ABI name records the proven view layout for lowering.
  kernel.defn @cublasSgemm_broadcast3d_colmajor_nt_alpha_beta(
      %A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>,
      %C: tensor<?x?xf32>, %beta: f32, %alpha: f32)
      -> tensor<?x?x?xf32> {
    // ABI contract only: the launch result is the rank-3 updated C view.
    kernel.yield %A : tensor<?x?x?xf32>
  }

  // Source-faithful form of the same Parboil operation. The matcher has
  // already proved and extracted the enclosing M/N loops, K reduction, and
  // beta*C + alpha*dot epilogue; this definition records only its flat-buffer
  // ABI contract.
  kernel.defn @cublasSgemm_flat_colmajor_nt_alpha_beta(
      %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
      %M: index, %N: index, %K: index,
      %lda: index, %ldb: index, %ldc: index,
      %beta: f32, %alpha: f32) {
    kernel.yield
  }

  kernel.defn @cublasSgemm_broadcast3d_memref(
      %A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>,
      %C: memref<?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%C : memref<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    }
    kernel.yield
  }

  // Batched FP32 matrix multiplication with one right-hand matrix shared by
  // every batch: C[b,m,n] = sum_k A[b,m,k] * B[k,n].  The CUDA ABI lowers
  // this to cublasSgemmStridedBatched with strideB=0 and beta=0.
  kernel.defn @cublasSgemm_strided_batched_broadcast_rhs(
      %A: tensor<?x?x?xf32>, %B: tensor<?x?xf32>,
      %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %result : tensor<?x?x?xf32>
  }

  // Darknet-style explicit im2col + SGEMM as one library op. The matcher
  // recognizes the zero-fill, guarded im2col workspace materialization, and
  // following GEMM as a single composition; ABI lowering maps this directly
  // to cuDNN convolution with caller-supplied padding and stride.
  kernel.defn @cudnnConvolutionFwd_im2col_gemm(
      %input: memref<?xf32>, %weights: memref<?x?x?xf32>,
      %output: memref<?xf32>,
      %channels: i32, %height: i32, %width: i32, %out_channels: i32,
      %ksize: i32, %stride: i32, %pad: i32) {
    kernel.yield
  }

  // llama2.c RMSNorm matched as:
  //   ss = sum(x[i] * x[i])
  //   out[i] = weight[i] * x[i] * rsqrt(ss / N + 1e-5)
  // ABI lowering maps this to a runtime shim. The shim owns the optimized
  // implementation choice (cuDNN frontend/custom CUDA/CPU fallback).

  // Cyclic 1-D reindexing.  The matcher proves
  // out[i] = input[(i + rotate_offset) mod N]; ABI lowering maps it to two


  // Independent runtime-controlled reflection of a 2-D tensor's axes.

  kernel.defn @cubCountNonzero1D_f32_tensor(
      %input: tensor<?xf32>, %out: tensor<i32>) -> tensor<i32> {
    kernel.yield %out : tensor<i32>
  }

  kernel.defn @cubSegmentedCountNonzero2D_f32_tensor(
      %input: tensor<?x?xf32>, %out: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }

  kernel.defn @cubEqualAll1D_f32_tensor(
      %lhs: tensor<?xf32>, %rhs: tensor<?xf32>, %out: tensor<i32>)
      -> tensor<i32> {
    kernel.yield %out : tensor<i32>
  }

  kernel.defn @cubSegmentedLogicalSelect_i32_tensor(
      %all_input: tensor<?x?xi32>, %any_input: tensor<?x?xi32>, %all: i1,
      %out: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }

  kernel.defn @cublasDdot(
      %x: tensor<?xf64>, %y: tensor<?xf64>,
      %out: tensor<f64>) -> tensor<f64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%x, %y : tensor<?xf64>, tensor<?xf64>) outs(%out : tensor<f64>) {
    ^bb0(%xv: f64, %yv: f64, %ov: f64):
      %p = arith.mulf %xv, %yv : f64
      %s = arith.addf %ov, %p : f64
      linalg.yield %s : f64
    } -> tensor<f64>
    kernel.yield %result : tensor<f64>
  }

  kernel.defn @cublasSdot(
      %x: tensor<?xf32>, %y: tensor<?xf32>,
      %out: tensor<f32>) -> tensor<f32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%x, %y : tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<f32>) {
    ^bb0(%xv: f32, %yv: f32, %ov: f32):
      %p = arith.mulf %xv, %yv : f32
      %s = arith.addf %ov, %p : f32
      linalg.yield %s : f32
    } -> tensor<f32>
    kernel.yield %result : tensor<f32>
  }

  kernel.defn @whisperExpShiftSum_f32_tensor(
      %x: tensor<?xf32>, %out: tensor<?xf32>, %sum: tensor<f32>,
      %max: f32) -> (tensor<?xf32>, tensor<f32>) {
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%x : tensor<?xf32>) outs(%out, %sum : tensor<?xf32>, tensor<f32>) {
    ^bb0(%xv: f32, %ov: f32, %sumv: f32):
      %shifted = arith.subf %xv, %max : f32
      %e = math.exp %shifted : f32
      %acc = arith.addf %sumv, %e : f32
      linalg.yield %e, %acc : f32, f32
    } -> (tensor<?xf32>, tensor<f32>)
    kernel.yield %result#0, %result#1 : tensor<?xf32>, tensor<f32>
  }

  // llama2.c row softmax in-place:
  //   x = exp(x - max(x)) / sum(exp(x - max(x)))
  // ABI lowering maps this to cudnnSoftmaxForward for FP32.
  kernel.defn @cudnnSoftmaxForward(%x: memref<?xf32>) {
    kernel.yield
  }

  kernel.defn @cudnnSoftmaxForward_tensor(%x: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %x : tensor<?xf32>
  }

  kernel.defn @cudnnSoftmaxForwardOut_tensor(
      %scores: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  // Llama standalone elementwise / copy helpers. ABI lowering routes these
  // to CUDA-runtime/cuDNN/cuBLAS shims in the CUDA backend.
  kernel.defn @cudaCopy1D_f32_tensor(
      %src: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%sv: f32, %ov: f32):
      linalg.yield %sv : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cudaCopy2D_f32_tensor(
      %src: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%src : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%sv: f32, %ov: f32):
      linalg.yield %sv : f32
    } -> tensor<?x?xf32>
    kernel.yield %result : tensor<?x?xf32>
  }

  kernel.defn @cublasBroadcastAxis0_f32(
      %src: tensor<?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }

  kernel.defn @cublasBroadcastAxis1_f32(
      %src: tensor<?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }

  kernel.defn @cudaCopy3D_f32_tensor(
      %src: tensor<?x?x?xf32>,
      %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%src : tensor<?x?x?xf32>) outs(%out : tensor<?x?x?xf32>) {
    ^bb0(%sv: f32, %ov: f32):
      linalg.yield %sv : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %result : tensor<?x?x?xf32>
  }

  kernel.defn @cudaCopy6D_f32_tensor(
      %src: tensor<?x?x?x?x?x?xf32>,
      %out: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
      ],
      iterator_types = [
        "parallel", "parallel", "parallel",
        "parallel", "parallel", "parallel"
      ]
    } ins(%src : tensor<?x?x?x?x?x?xf32>)
      outs(%out : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%sv: f32, %ov: f32):
      linalg.yield %sv : f32
    } -> tensor<?x?x?x?x?x?xf32>
    kernel.yield %result : tensor<?x?x?x?x?x?xf32>
  }

  kernel.defn @cudaAdd_f32_tensor(
      %x: tensor<?xf32>, %y: tensor<?xf32>,
      %out: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%x, %y : tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%xv: f32, %yv: f32, %ov: f32):
      %sum = arith.addf %xv, %yv : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cudaMaskSelect_f32_tensor(
      %scores: tensor<?xf32>, %out: tensor<?xf32>, %pos: i32)
      -> tensor<?xf32> {
    %one = arith.constant 1.000000e+00 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%scores : tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%sv: f32, %ov: f32):
      %i = linalg.index 0 : index
      %ii = arith.index_cast %i : index to i32
      %pred = arith.cmpi sgt, %ii, %pos : i32
      %drop_i = arith.extui %pred : i1 to i32
      %drop = arith.sitofp %drop_i : i32 to f32
      %keep = arith.subf %one, %drop : f32
      %kept = arith.mulf %keep, %sv : f32
      %masked = arith.mulf %drop, %neg_inf : f32
      %r = arith.addf %kept, %masked : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cudaSwiGLU_f32_tensor(
      %gate: tensor<?xf32>, %up: tensor<?xf32>,
      %out: tensor<?xf32>) -> tensor<?xf32> {
    %one = arith.constant 1.000000e+00 : f32
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%gate, %up : tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%g: f32, %u: f32, %ov: f32):
      %ng = arith.negf %g : f32
      %e = math.exp %ng : f32
      %den = arith.addf %e, %one : f32
      %silu = arith.divf %g, %den : f32
      %r = arith.mulf %silu, %u : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cudaRopeMulMulSub_f32_tensor(
      %a: tensor<?x?xf32>, %b: tensor<?xf32>,
      %c: tensor<?x?xf32>, %d: tensor<?xf32>,
      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b, %c, %d : tensor<?x?xf32>, tensor<?xf32>,
          tensor<?x?xf32>, tensor<?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %cv: f32, %dv: f32, %ov: f32):
      %p0 = arith.mulf %av, %bv : f32
      %p1 = arith.mulf %cv, %dv : f32
      %r = arith.subf %p0, %p1 : f32
      linalg.yield %r : f32
    } -> tensor<?x?xf32>
    kernel.yield %result : tensor<?x?xf32>
  }

  kernel.defn @cudaRopeMulMulAdd_f32_tensor(
      %a: tensor<?x?xf32>, %b: tensor<?xf32>,
      %c: tensor<?x?xf32>, %d: tensor<?xf32>,
      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b, %c, %d : tensor<?x?xf32>, tensor<?xf32>,
          tensor<?x?xf32>, tensor<?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %cv: f32, %dv: f32, %ov: f32):
      %p0 = arith.mulf %av, %bv : f32
      %p1 = arith.mulf %cv, %dv : f32
      %r = arith.addf %p0, %p1 : f32
      linalg.yield %r : f32
    } -> tensor<?x?xf32>
    kernel.yield %result : tensor<?x?xf32>
  }

  // GEMM-ALPHA-ONLY: C += alpha*A*B (beta=1, accumulate-into-C, custom alpha).
  kernel.defn @cublasDgemm_alpha_only(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                                       %C: tensor<?x?xf64>,
                                       %alpha: f64) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEAM-SCALE-2D: C = alpha * C (elementwise scaling, 2D).
  kernel.defn @cublasDgeam_scale2D(%C: tensor<?x?xf64>, %alpha: f64)
                                  -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %t = arith.mulf %out, %alpha : f64
      linalg.yield %t : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMV (2D matrix x 1D vector): y += A * x.
  // Operand order seen in atax, mvt, gesummv, 3mm.
  kernel.defn @cublasDgemv(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                            %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  kernel.defn @cublasDgemv_T(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                              %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // GEMV-SUBTRACT variants: y -= A*x (alpha=-1, beta=1).
  kernel.defn @cublasDgemv_subtract(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                                     %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %s = arith.subf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  kernel.defn @cublasDgemv_subtract_T(%A: tensor<?x?xf64>,
                                       %x: tensor<?xf64>,
                                       %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %s = arith.subf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  kernel.defn @cublasSgemv(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                            %y: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf32>, tensor<?xf32>)
      outs(%y : tensor<?xf32>) {
    ^bb0(%a: f32, %xv: f32, %out: f32):
      %p = arith.mulf %a, %xv : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cublasSgemv_T(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                              %y: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf32>, tensor<?xf32>)
      outs(%y : tensor<?xf32>) {
    ^bb0(%a: f32, %xv: f32, %out: f32):
      %p = arith.mulf %a, %xv : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  // GEMV-ALPHA: y += alpha * A * x (gemver pattern).
  kernel.defn @cublasDgemv_alpha(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                                  %y: tensor<?xf64>,
                                  %alpha: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // GER-RANK2: A += u1*v1^T + u2*v2^T.
  // gemver-style fused rank-2 update.
  kernel.defn @cublasDger_rank2(%u1: tensor<?xf64>, %v1: tensor<?xf64>,
                                 %u2: tensor<?xf64>, %v2: tensor<?xf64>,
                                 %A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%u1, %v1, %u2, %v2
          : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
      outs(%A : tensor<?x?xf64>) {
    ^bb0(%u1v: f64, %v1v: f64, %u2v: f64, %v2v: f64, %out: f64):
      %p1 = arith.mulf %u1v, %v1v : f64
      %p2 = arith.mulf %u2v, %v2v : f64
      %s1 = arith.addf %out, %p1 : f64
      %s2 = arith.addf %s1, %p2 : f64
      linalg.yield %s2 : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // Overwriting outer product: C[i,j] = u[i] * v[j].  Unlike the BLAS GER
  // update primitive this operation does not consume the old contents of C.
  kernel.defn @cublasDgemm_outer_product(
      %u: tensor<?xf64>, %v: tensor<?xf64>,
      %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%u, %v : tensor<?xf64>, tensor<?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%uv: f64, %vv: f64, %out: f64):
      %p = arith.mulf %uv, %vv : f64
      linalg.yield %p : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // AXPBY: y = a*x + b*y (gesummv pattern).
  kernel.defn @cublasDaxpby(%x: tensor<?xf64>, %y: tensor<?xf64>,
                             %alpha: f64, %beta: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%x : tensor<?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%xv: f64, %out: f64):
      %ax = arith.mulf %alpha, %xv : f64
      %by = arith.mulf %beta, %out : f64
      %s = arith.addf %ax, %by : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  kernel.defn @cublasSaxpby(%x: tensor<?xf32>, %y: tensor<?xf32>,
                            %a: f32, %b: f32) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }

  kernel.defn @cublasSscal(%x: tensor<?xf32>, %a: f32) -> tensor<?xf32> {
    kernel.yield %x : tensor<?xf32>
  }

  // AXPY (alpha=1): y += x.
  kernel.defn @cublasDaxpy_unit(%x: tensor<?xf64>, %y: tensor<?xf64>)
                                -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%x : tensor<?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%xv: f64, %out: f64):
      %s = arith.addf %out, %xv : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // MEMSET-ZERO-1D: y[i] = 0 for all i.
  kernel.defn @memset_zero_1D(%y: tensor<?xf64>) -> tensor<?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  kernel.defn @memset_zero_1D_f32(%y: tensor<?xf32>) -> tensor<?xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  // MEMSET-ZERO-2D: A[i,j] = 0 for all i,j.
  kernel.defn @memset_zero_2D(%A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%A : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  kernel.defn @memset_zero_2D_f32(%A: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%A : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?xf32>
    kernel.yield %result : tensor<?x?xf32>
  }

  // MEMSET-CONST-1D: fill the diagonal of a 2D tensor with 1.0.
  // The matcher names this "1D" because the iter space is 1D (single d0) —
  // the tensor is 2D but accessed at (d0, d0). Used in correlation's
  // diagonal initialization. NOTE: the constant value is HARD-CODED to 1.0
  // because the matcher's Cap binding for the literal isn't currently
  // propagated through render_launch. A different caller wanting a
  // different fill value would need a separate library entry.
  kernel.defn @memset_const_1D(%A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %one = arith.constant 1.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0, d0)>],
      iterator_types = ["parallel"]
    } outs(%A : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %one : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // ELEMWISE-DIV-SCALAR: y[i] = y[i] / s.
  kernel.defn @elemwise_div_scalar(%y: tensor<?xf64>, %s: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf64>) {
    ^bb0(%out: f64):
      %t = arith.divf %out, %s : f64
      linalg.yield %t : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // REDUCE-SUM-AXIS: out[j] = sum over the *other* axis of a 2D tensor.
  // The 1D output's length matches the parallel axis of the 2D input.
  // Indexing maps mirror what correlation's raise step produces.
  kernel.defn @reduce_sum_axis(%X: tensor<?x?xf64>, %y: tensor<?xf64>)
                                -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%X : tensor<?x?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %s = arith.addf %out, %in : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // SYRK: C[j<=i] = beta*C[j<=i] + alpha*A*A^T  (symmetric rank-k update).
  //
  // Two-step canonical body matching what RaiseToLinalg emits for PolyBench
  // syrk: masked beta-scale of C on the lower triangle, then masked
  // alpha-A*A^T-accumulate. The mask is recomputed from linalg.index +
  // affine.apply inside each linalg.generic so the defn body is
  // self-contained — no external mask SSA is threaded as an operand.
  //
  // Operand order (matches matcher emit): two A-views (the matcher passes
  // both ins of the gemm-shape linalg, which is the same A twice), C, beta,
  // alpha.
  kernel.defn @cublasDsyrk(%A: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
                            %C: tensor<?x?xf64>,
                            %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %scaled_val = arith.mulf %out, %beta : f64
      %r = arith.select %cond, %scaled_val, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d0)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %A2 : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a: f64, %a_t: f64, %out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 2 : index
      %scaled_a = arith.mulf %alpha, %a : f64
      %p = arith.mulf %scaled_a, %a_t : f64
      %s = arith.addf %out, %p : f64
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %r = arith.select %cond, %s, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // SYR2K: C[j<=i] = beta*C[j<=i] + alpha*(A*B^T + B*A^T)  (rank-2k update).
  //
  // Five tensor operands: (A1, B1, B2, A2, C) — the matcher's body splits
  // the rank-2 update across four ins to the second linalg.generic. Maps
  // and iter ordering replicate exactly what RaiseToLinalg emits.
  kernel.defn @cublasDsyr2k(%A1: tensor<?x?xf64>, %B1: tensor<?x?xf64>,
                             %B2: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
                             %C: tensor<?x?xf64>,
                             %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %scaled_val = arith.mulf %out, %beta : f64
      %r = arith.select %cond, %scaled_val, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d0)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A1, %B1, %B2, %A2
          : tensor<?x?xf64>, tensor<?x?xf64>,
            tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a1: f64, %b1: f64, %b2: f64, %a2: f64, %out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 2 : index
      %t1 = arith.mulf %a1, %alpha : f64
      %t2 = arith.mulf %t1, %b1 : f64
      %t3 = arith.mulf %b2, %alpha : f64
      %t4 = arith.mulf %t3, %a2 : f64
      %t5 = arith.addf %t2, %t4 : f64
      %t6 = arith.addf %out, %t5 : f64
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %r = arith.select %cond, %t6, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // ========================================================================
  // Stencils (Bucket 2). These bodies operate on memref-form linalg.generic
  // because the surrounding time-stepping loop holds a memref iter, so
  // --linalg-debufferize never lifts them to tensor form. The defns mirror
  // the strided memref types that RaiseToLinalg emits for PolyBench stencils.
  // Constants are hard-coded to PolyBench's values (1/3, 1/5, 1/8, etc.) —
  // a Cap-bound literal would be passed as a runtime operand for general
  // callers; we don't do that yet (matcher's Cap-binds-to-Lit means the
  // launch operand list drops the literal).
  // ========================================================================

  // JACOBI 1D 3-point: out[i] = (a[i] + b[i+1] + c[i+2]) / 3
  // The "shift" is baked into the subview offsets (the linalg body sees
  // identity-accessed memrefs at different base offsets).
  kernel.defn @jacobi_1d_3pt(
      %a: memref<?xf64, strided<[1]>>,
      %b: memref<?xf64, strided<[1], offset: 1>>,
      %c: memref<?xf64, strided<[1], offset: 2>>,
      %out: memref<?xf64, strided<[1], offset: 1>>) {
    %cst = arith.constant 0.33333333333333331 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b, %c
          : memref<?xf64, strided<[1]>>,
            memref<?xf64, strided<[1], offset: 1>>,
            memref<?xf64, strided<[1], offset: 2>>)
      outs(%out : memref<?xf64, strided<[1], offset: 1>>) {
    ^bb0(%av: f64, %bv: f64, %cv: f64, %outv: f64):
      %s1 = arith.addf %av, %bv : f64
      %s2 = arith.addf %s1, %cv : f64
      %r  = arith.mulf %s2, %cst : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // JACOBI 2D 5-point: out[i,j] = (c + n + s + w + e) / 5
  kernel.defn @jacobi_2d_5pt(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a4: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %cst = arith.constant 0.20000000000000001 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64, %ov: f64):
      %s1 = arith.addf %v0, %v1 : f64
      %s2 = arith.addf %s1, %v2 : f64
      %s3 = arith.addf %s2, %v3 : f64
      %s4 = arith.addf %s3, %v4 : f64
      %r  = arith.mulf %s4, %cst : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // HEAT 3D 7-point: out = c + (l-2c+r + d-2c+u + b-2c+f)/8.
  // Operand order from matcher: x-pair (a0,a2), center (a1), y-pair (a3,a4),
  // z-pair (a5,a6).
  kernel.defn @heat_3d_7pt(
      %a0: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a1: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a2: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a3: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a4: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a5: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a6: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %out: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    %coef = arith.constant 0.125 : f64
    %two  = arith.constant 2.000000e+00 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4, %a5, %a6
          : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>)
      outs(%out : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64,
         %v5: f64, %v6: f64, %ov: f64):
      %t2c = arith.mulf %v1, %two : f64
      %x_diff = arith.subf %v0, %t2c : f64
      %x_lap  = arith.addf %x_diff, %v2 : f64
      %x_sc   = arith.mulf %x_lap, %coef : f64
      %y_diff = arith.subf %v3, %t2c : f64
      %y_lap  = arith.addf %y_diff, %v4 : f64
      %y_sc   = arith.mulf %y_lap, %coef : f64
      %z_diff = arith.subf %v5, %t2c : f64
      %z_lap  = arith.addf %z_diff, %v6 : f64
      %z_sc   = arith.mulf %z_lap, %coef : f64
      %xy     = arith.addf %x_sc, %y_sc : f64
      %xyz    = arith.addf %xy, %z_sc : f64
      %r      = arith.addf %xyz, %v1 : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D H-field update: out -= 0.5 * (in0 - in1).
  kernel.defn @fdtd_update_2in(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %coef = arith.constant 5.000000e-01 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %ov: f64):
      %diff = arith.subf %v0, %v1 : f64
      %sc   = arith.mulf %diff, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D E-field update: out -= 0.7 * (in0 - in1 + in2 - in3).
  kernel.defn @fdtd_E_update(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %coef = arith.constant 6.999999999999999e-01 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %ov: f64):
      %d1   = arith.subf %v0, %v1 : f64
      %a    = arith.addf %d1, %v2 : f64
      %d2   = arith.subf %a, %v3 : f64
      %sc   = arith.mulf %d2, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D source-injection: out[j] = source (broadcast 0-D memref over 1D).
  // Matcher emits this when the input's indexing map is `() -> ()` (scalar
  // access).
  kernel.defn @broadcast_scalar_to_vec(
      %src: memref<f64, strided<[], offset: ?>>,
      %out: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : memref<f64, strided<[], offset: ?>>)
      outs(%out : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    }
    kernel.yield
  }

  // cublasDcopy: 1D-to-1D identity copy (out[i] = in[i]). Used by doitgen
  // for write-back of the scratch buffer.
  kernel.defn @cublasDcopy(
      %src: memref<?xf64, strided<[1]>>,
      %out: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : memref<?xf64, strided<[1]>>)
      outs(%out : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    }
    kernel.yield
  }

  // CENTERED-SUM-SQUARES: out[j] = sum_i (X[i,j] - mean[j])^2.
  // Variance accumulation (without the 1/N division — that's a separate
  // elemwise_div_scalar in correlation).
  kernel.defn @centered_sum_squares(%X: tensor<?x?xf64>,
                                     %mean: tensor<?xf64>,
                                     %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%X, %mean : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%in: f64, %m: f64, %out: f64):
      %d = arith.subf %in, %m : f64
      %p = arith.mulf %d, %d : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // ============================================================
  // Tensor-form stencil defns (multi-root debufferize emits these).
  // Identical bodies to the memref-form stencils above, but with plain
  // `tensor<?...xf64>` operand/result types — the polygeist.submap chain
  // that encodes the offsets is opaque to the lowerer, so the defns can
  // treat each input as a plain tensor of the same rank.
  // ============================================================

  // JACOBI 1D 3-point, tensor form.
  kernel.defn @jacobi_1d_3pt_tensor(
      %a: tensor<?xf64>, %b: tensor<?xf64>, %c: tensor<?xf64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %cst = arith.constant 0.33333333333333331 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b, %c : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%av: f64, %bv: f64, %cv: f64, %ov: f64):
      %s1 = arith.addf %av, %bv : f64
      %s2 = arith.addf %s1, %cv : f64
      %r  = arith.mulf %s2, %cst : f64
      linalg.yield %r : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // JACOBI 2D 5-point, tensor form.
  kernel.defn @jacobi_2d_5pt_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>, %a2: tensor<?x?xf64>,
      %a3: tensor<?x?xf64>, %a4: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 0.20000000000000001 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4
          : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>,
            tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64, %ov: f64):
      %s1 = arith.addf %v0, %v1 : f64
      %s2 = arith.addf %s1, %v2 : f64
      %s3 = arith.addf %s2, %v3 : f64
      %s4 = arith.addf %s3, %v4 : f64
      %r  = arith.mulf %s4, %cst : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // HEAT 3D 7-point, tensor form.
  kernel.defn @heat_3d_7pt_tensor(
      %a0: tensor<?x?x?xf64>, %a1: tensor<?x?x?xf64>, %a2: tensor<?x?x?xf64>,
      %a3: tensor<?x?x?xf64>, %a4: tensor<?x?x?xf64>, %a5: tensor<?x?x?xf64>,
      %a6: tensor<?x?x?xf64>,
      %out_init: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %coef = arith.constant 0.125 : f64
    %two  = arith.constant 2.000000e+00 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4, %a5, %a6
          : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>,
            tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>,
            tensor<?x?x?xf64>)
      outs(%out_init : tensor<?x?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64,
         %v5: f64, %v6: f64, %ov: f64):
      %t2c = arith.mulf %v1, %two : f64
      %x_diff = arith.subf %v0, %t2c : f64
      %x_lap  = arith.addf %x_diff, %v2 : f64
      %x_sc   = arith.mulf %x_lap, %coef : f64
      %y_diff = arith.subf %v3, %t2c : f64
      %y_lap  = arith.addf %y_diff, %v4 : f64
      %y_sc   = arith.mulf %y_lap, %coef : f64
      %z_diff = arith.subf %v5, %t2c : f64
      %z_lap  = arith.addf %z_diff, %v6 : f64
      %z_sc   = arith.mulf %z_lap, %coef : f64
      %xy     = arith.addf %x_sc, %y_sc : f64
      %xyz    = arith.addf %xy, %z_sc : f64
      %r      = arith.addf %xyz, %v1 : f64
      linalg.yield %r : f64
    } -> tensor<?x?x?xf64>
    kernel.yield %r : tensor<?x?x?xf64>
  }

  // FDTD-2D H-field update, tensor form.
  kernel.defn @fdtd_update_2in_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %coef = arith.constant 5.000000e-01 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1 : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %ov: f64):
      %diff = arith.subf %v0, %v1 : f64
      %sc   = arith.mulf %diff, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // Broadcast a 0-D tensor (scalar) over a 1D tensor — tensor-form twin
  // of @broadcast_scalar_to_vec. Used by multi-root fdtd-2d's source-
  // injection step where polygeist.submap produces a rank-0 tensor<f64>.
  kernel.defn @broadcast_scalar_to_vec_tensor(
      %src: tensor<f64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : tensor<f64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // cublasDcopy, tensor form (1D identity copy). Used by multi-root
  // fdtd-2d's source-injection step.
  kernel.defn @cublasDcopy_tensor(
      %src: tensor<?xf64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : tensor<?xf64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // FDTD-2D E-field update, tensor form.
  kernel.defn @fdtd_E_update_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>,
      %a2: tensor<?x?xf64>, %a3: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %coef = arith.constant 6.999999999999999e-01 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3
          : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %ov: f64):
      %d1   = arith.subf %v0, %v1 : f64
      %a    = arith.addf %d1, %v2 : f64
      %d2   = arith.subf %a, %v3 : f64
      %sc   = arith.mulf %d2, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // Conv2D 9-tap weighted (3x3 stencil).
  // Operands: 9 input subviews (memref form) of one source tensor (one per
  // 3x3 neighbour position) + 1 output subview. The 9 scalar weights live
  // *inside* the matched linalg.generic body, not in the kernel.launch
  // operand list — surfacing them is a matcher-extension TODO. For the
  // --lower-kernel-launch-to-cublas dispatch this defn is just a symbol
  // carrier (the cuDNN runtime shim hardcodes the polybench weights);
  // body is no-op so the verifier passes.
  kernel.defn @cudnnConvolution2D_9tap(
      %A0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %w0: f64, %w1: f64, %w2: f64,
      %w3: f64, %w4: f64, %w5: f64,
      %w6: f64, %w7: f64, %w8: f64) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_tensor(
      %A0: tensor<?x?xf64>, %A1: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
      %A3: tensor<?x?xf64>, %A4: tensor<?x?xf64>, %A5: tensor<?x?xf64>,
      %A6: tensor<?x?xf64>, %A7: tensor<?x?xf64>, %A8: tensor<?x?xf64>,
      %C:  tensor<?x?xf64>,
      %w0: f64, %w1: f64, %w2: f64,
      %w3: f64, %w4: f64, %w5: f64,
      %w6: f64, %w7: f64, %w8: f64) -> tensor<?x?xf64> {
    kernel.yield %C : tensor<?x?xf64>
  }

  // FP32 variant of the conv2d 9-tap defn. Same structure as the f64 one
  // but with f32 memrefs + f32 weights. Selected by the rewriter when the
  // matched body's operand types are f32 (it emits @cudnnConvolution2D_9tap_f32
  // as the launch symbol). Phase 2 of the cuDNN conv generalization.
  kernel.defn @cudnnConvolution2D_9tap_f32(
      %A0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %w0: f32, %w1: f32, %w2: f32,
      %w3: f32, %w4: f32, %w5: f32,
      %w6: f32, %w7: f32, %w8: f32) {
    kernel.yield
  }

  // Conv2D 25-tap weighted (5x5 stencil), surfaced exactly like the 9-tap
  // path: 25 shifted input subviews, one output interior subview, then 25
  // scalar filter weights in row-major order.
  kernel.defn @cudnnConvolution2D_25tap(
      %A0:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A1:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A2:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A3:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A4:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A5:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A6:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A7:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A8:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A9:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A10: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A11: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A12: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A13: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A14: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A15: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A16: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A17: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A18: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A19: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A20: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A21: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A22: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A23: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A24: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %C:   memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %w0:  f64, %w1:  f64, %w2:  f64, %w3:  f64, %w4:  f64,
      %w5:  f64, %w6:  f64, %w7:  f64, %w8:  f64, %w9:  f64,
      %w10: f64, %w11: f64, %w12: f64, %w13: f64, %w14: f64,
      %w15: f64, %w16: f64, %w17: f64, %w18: f64, %w19: f64,
      %w20: f64, %w21: f64, %w22: f64, %w23: f64, %w24: f64) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_25tap_f32(
      %A0:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A1:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A2:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A3:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A4:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A5:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A6:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A7:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A8:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A9:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A10: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A11: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A12: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A13: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A14: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A15: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A16: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A17: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A18: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A19: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A20: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A21: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A22: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A23: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A24: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %C:   memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %w0:  f32, %w1:  f32, %w2:  f32, %w3:  f32, %w4:  f32,
      %w5:  f32, %w6:  f32, %w7:  f32, %w8:  f32, %w9:  f32,
      %w10: f32, %w11: f32, %w12: f32, %w13: f32, %w14: f32,
      %w15: f32, %w16: f32, %w17: f32, %w18: f32, %w19: f32,
      %w20: f32, %w21: f32, %w22: f32, %w23: f32, %w24: f32) {
    kernel.yield
  }

  // Generalized odd-square weighted Conv2D stencil. The matcher proves the
  // original linalg inputs are same-base shifted subviews, then packs the
  // row-major KxK weights into %W and passes only the top-left input subview,
  // output interior subview, packed weights, and K. This avoids adding one
  // kernel.defn per tap count.
  kernel.defn @cudnnConvolution2D_ntap(
      %A: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %C: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %W: memref<?xf64>,
      %K: i32) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_ntap_f32(
      %A: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %C: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %W: memref<?xf32>,
      %K: i32) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_ntap_tensor(
      %A: tensor<?x?xf64>,
      %C: tensor<?x?xf64>,
      %W: tensor<?xf64>,
      %K: i32) -> tensor<?x?xf64> {
    kernel.yield %C : tensor<?x?xf64>
  }

  kernel.defn @cudnnConvolution2D_ntap_f32_tensor(
      %A: tensor<?x?xf32>,
      %C: tensor<?x?xf32>,
      %W: tensor<?xf32>,
      %K: i32) -> tensor<?x?xf32> {
    kernel.yield %C : tensor<?x?xf32>
  }

  // Generalized packed-weight Conv3D stencil. The matcher proves the raised
  // reduction uses a dense rank-6 window view, then passes the haloed 3D input
  // tensor, dense 3D output tensor, rank-3 filter tensor, and filter width.
  kernel.defn @cudnnConvolution3D_ntap_tensor(
      %A: tensor<?x?x?xf64>,
      %C: tensor<?x?x?xf64>,
      %W: tensor<?x?x?xf64>,
      %K: i32) -> tensor<?x?x?xf64> {
    kernel.yield %C : tensor<?x?x?xf64>
  }

  kernel.defn @cudnnConvolution3D_ntap_f32_tensor(
      %A: tensor<?x?x?xf32>,
      %C: tensor<?x?x?xf32>,
      %W: tensor<?x?x?xf32>,
      %K: i32) -> tensor<?x?x?xf32> {
    kernel.yield %C : tensor<?x?x?xf32>
  }

  // Flattened dense-grid form of the standard 3D seven-point axial stencil.
  // The adapter materializes the sparse 3x3x3 filter and dispatches cuDNN;
  // it does not implement the stencil arithmetic itself.
  kernel.defn @cudnnStencil3D7pt_f32_flat_tensor(
      %A: tensor<?xf32>, %C: tensor<?xf32>,
      %center_scale: f32, %neighbor_scale: f32,
      %ny: index, %nx: index,
      %out_x: index, %out_y: index, %out_z: index) -> tensor<?xf32> {
    kernel.yield %C : tensor<?xf32>
  }

  // Multi-channel, single-batch valid Conv3D. The rank-8 operand is the
  // logical [OC,OD,OH,OW,IC,KD,KH,KW] window; ABI lowering recovers the
  // underlying rank-4/5 NCDHW input before calling cuDNN.
  kernel.defn @cudnnConvolution3D_f32(
      %window: tensor<?x?x?x?x?x?x?x?xf32>,
      %filter: tensor<?x?x?x?x?xf32>,
      %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?xf32>
  }

  kernel.defn @cudnnConvolution3D_f32_bias(
      %window: tensor<?x?x?x?x?x?x?x?xf32>,
      %filter: tensor<?x?x?x?x?xf32>, %bias: tensor<?xf32>,
      %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?xf32>
  }
  kernel.defn @cudnnConvolution1D_f32_bias(
      %windows: tensor<?x?x?x?x?xf32>,
      %filter: tensor<?x?x?xf32>, %bias: tensor<?xf32>,
      %output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %output : tensor<?x?x?xf32>
  }
  kernel.defn @cudnnConvolution2D_f32_dilated(
      %windows: tensor<?x?x?x?x?x?xf32>,
      %filter: tensor<?x?x?x?xf32>, %output: tensor<?x?x?xf32>)
      -> tensor<?x?x?xf32> { kernel.yield %output : tensor<?x?x?xf32> }

  // 1D complex FFT ABI declarations. Complex values are represented as
  // interleaved real/imag pairs in the trailing dimension of size 2. The
  // runtime follows cuFFT semantics: inverse transforms are unnormalized.
  kernel.defn @cufftZ2Z_1D_tensor(
      %A: tensor<?x2xf64>,
      %C: tensor<?x2xf64>,
      %inverse: i32) -> tensor<?x2xf64> {
    kernel.yield %C : tensor<?x2xf64>
  }

  kernel.defn @cufftC2C_1D_tensor(
      %A: tensor<?x2xf32>,
      %C: tensor<?x2xf32>,
      %inverse: i32) -> tensor<?x2xf32> {
    kernel.yield %C : tensor<?x2xf32>
  }

  // Separable 3D tensor product: ai,bj,ck,ijk->abc. The operands are the
  // rank-6 broadcast/submap views produced by raising; ABI lowering unwraps
  // them to the shared psi buffer, the u buffer, and the output buffer.
  kernel.defn @cutensornetTensorProduct3D_f32_tensor(
      %psiA: tensor<?x?x?x?x?x?xf32>,
      %psiB: tensor<?x?x?x?x?x?xf32>,
      %psiC: tensor<?x?x?x?x?x?xf32>,
      %u: tensor<?x?x?x?x?x?xf32>,
      %out: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?x?x?xf32>
  }

  kernel.defn @cutensornetTensorProduct3D_f64_tensor(
      %psiA: tensor<?x?x?x?x?x?xf64>,
      %psiB: tensor<?x?x?x?x?x?xf64>,
      %psiC: tensor<?x?x?x?x?x?xf64>,
      %u: tensor<?x?x?x?x?x?xf64>,
      %out: tensor<?x?x?x?x?x?xf64>) -> tensor<?x?x?x?x?x?xf64> {
    kernel.yield %out : tensor<?x?x?x?x?x?xf64>
  }

  // Layout-aware two-input FP64 Einstein contractions. The matcher attaches
  // the original linalg indexing maps to each launch; ABI lowering combines
  // them with polygeist.submap strides and routes the operation to
  // cuTensorNet. The unranked signature is the generic route; ranked legacy
  // symbols remain accepted for compatibility with existing matched files.
  kernel.defn @cutensornetContraction2_f64(
      %A: tensor<*xf64>,
      %B: tensor<*xf64>,
      %C: tensor<*xf64>) -> tensor<*xf64> {
    kernel.yield %C : tensor<*xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r4r5r4(
      %A: tensor<?x?x?x?xf64>,
      %B: tensor<?x?x?x?x?xf64>,
      %C: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %C : tensor<?x?x?x?xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r5r4r4(
      %A: tensor<?x?x?x?x?xf64>,
      %B: tensor<?x?x?x?xf64>,
      %C: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %C : tensor<?x?x?x?xf64>
  }

  kernel.defn @cutensornetContraction2_f64_r5r5r4(
      %A: tensor<?x?x?x?x?xf64>,
      %B: tensor<?x?x?x?x?xf64>,
      %C: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %C : tensor<?x?x?x?xf64>
  }

  // cuDNN backend operation graph: relu(alpha * x + bias).
  kernel.defn @cudnnPointwiseAffineRelu_f32(
      %x: tensor<?xf32>, %bias: tensor<?xf32>, %out: tensor<?xf32>,
      %alpha: f32) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  // Parameterized rank-1 f32 pointwise DAG. The launch carries compact graph
  // bytecode as attributes; unused tensor/scalar ABI slots are ignored.
  kernel.defn @cudnnPointwiseGraph_f32(
      %in0: tensor<?xf32>, %in1: tensor<?xf32>,
      %in2: tensor<?xf32>, %in3: tensor<?xf32>,
      %out: tensor<?xf32>,
      %s0: f32, %s1: f32, %s2: f32, %s3: f32,
      %s4: f32, %s5: f32, %s6: f32, %s7: f32) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  kernel.defn @cubInclusiveSum1D_f32_tensor(
      %input: tensor<?xf32>, %final: tensor<f32>,
      %output: tensor<?xf32>) -> (tensor<f32>, tensor<?xf32>) {
    kernel.yield %final, %output : tensor<f32>, tensor<?xf32>
  }

  kernel.defn @cubSegmentedInclusiveProduct2D_f32_tensor(
      %input: tensor<?x?xf32>, %output: tensor<?x?xf32>,
      %final: tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) {
    kernel.yield %output, %final : tensor<?x?xf32>, tensor<?xf32>
  }

  kernel.defn @cubExclusiveSum1D_i32_memref(
      %input: memref<?xi32>, %output: memref<?xi32>) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_f16(
      %A0: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %w0: f16, %w1: f16, %w2: f16,
      %w3: f16, %w4: f16, %w5: f16,
      %w6: f16, %w7: f16, %w8: f16) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_bf16(
      %A0: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %w0: bf16, %w1: bf16, %w2: bf16,
      %w3: bf16, %w4: bf16, %w5: bf16,
      %w6: bf16, %w7: bf16, %w8: bf16) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_i32(
      %A0: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %w0: i32, %w1: i32, %w2: i32,
      %w3: i32, %w4: i32, %w5: i32,
      %w6: i32, %w7: i32, %w8: i32) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_i16(
      %A0: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %w0: i16, %w1: i16, %w2: i16,
      %w3: i16, %w4: i16, %w5: i16,
      %w6: i16, %w7: i16, %w8: i16) {
    kernel.yield
  }
}
