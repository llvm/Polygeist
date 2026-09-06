//===- LowerKernelLaunchToCuBLAS.cpp - kernel.launch → cuBLAS ABI -------===//
//
// Phase-2 *ABI* lowering. Distinct from the canonical-defn lowering in
// `LowerKernelLaunch.cpp` (which inlines a reference linalg.generic body):
// this pass replaces each recognised `kernel.launch @<libsym>(...)` with a
// `func.call` to the matching runtime shim ABI function declared in
// `runtime/polygeist_cublas_rt.h`. Link the shim object file (CPU stub
// for validation, cuBLAS-backed for hardware) to produce an executable.
//
// SUPPORTED LIBRARY SYMBOLS (extend by adding to `kLowerings`):
//   @cublasDgemm  →  polygeist_cublas_dgemm(M, N, K, alpha, A, lda, B, ldb,
//                                              beta, C, ldc)
//
// EXPECTED INPUT IR:
//   `kernel.launch` ops live in TENSOR form (the matcher emits them in
//   tensor form by default). For each launch we synthesise:
//     - `bufferization.to_memref` for each tensor operand
//     - dim queries (static when possible, `memref.dim` when dynamic)
//     - the `func.call` to the shim ABI function
//     - `bufferization.to_tensor restrict writable` to recover the result
//   The forward declaration of each shim function is added to the module
//   if not already present.
//
// OUT-OF-SCOPE (follow-up work):
//   * Device-residency hoisting (eliminate H↔D copies between consecutive
//     launches). The current per-call copies in the CUDA backend dominate
//     for small matrices.
//   * Non-f64 element types.
//   * Other library symbols (axpy, axpby, gemv, scal, …).
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "KernelLaunchLoweringUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "lower-kernel-launch-to-cublas"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

// Symbol of the runtime ABI function for each supported library op. Add
// more entries here as the matcher's library grows.
struct ShimDecl {
  StringRef shimSymbol;     // e.g. "polygeist_cublas_dgemm"
  // Arg types for the func.func private declaration. Filled lazily based
  // on the launch's MLIR types so element types flow through.
};

static StringRef shimSymbolFor(StringRef libSym) {
  if (libSym == "cubHistogramEvenI32ShiftZero_memref")
    return "polygeist_cub_histogram_even_i32_shift_zero";
  if (libSym == "cublasDtrsvLowerRowMajor_memref")
    return "polygeist_cublas_dtrsv_lower_row_major";
  if (libSym == "cusolverDnDpotrfLowerRowMajor_memref")
    return "polygeist_cusolver_dpotrf_lower_row_major";
  if (libSym == "cusparseSpMV_CSR_f32_memref")
    return "polygeist_cusparse_spmv_csr_f32_sized";
  if (libSym == "cusparseSpMV_CSR_f64_memref")
    return "polygeist_cusparse_spmv_csr_f64_sized";
  if (libSym == "cusparseSpMM_CSR_f32_memref")
    return "polygeist_cusparse_spmm_csr_f32_sized";
  if (libSym == "cusparseSpMM_COO_f32_memref")
    return "polygeist_cusparse_spmm_coo_f32_sized";
  if (libSym == "cusparseSpMM_BSR_f32_memref")
    return "polygeist_cusparse_spmm_bsr_f32_sized";
  if (libSym == "cusparseSpMV_JDS_f32_memref")
    return "polygeist_cusparse_spmv_jds_f32_sized";
  if (libSym == "custenStencil2DXY_f64_memref" ||
      libSym == "custenStencil2DXY_f64_tensor")
    return "polygeist_custen_stencil2d_xy_f64";
  if (libSym.starts_with("cutensorUnary_") && libSym.ends_with("_f32"))
    return "polygeist_cutensor_unary_f32";
  if (libSym == "cudnnPointwiseAffineRelu_f32")
    return "polygeist_cudnn_pointwise_affine_relu_f32";
  if (libSym == "cudnnPointwiseGraph_f32")
    return "polygeist_cudnn_pointwise_graph_f32";
  if (libSym == "cubInclusiveSum1D_f32_tensor")
    return "polygeist_cub_inclusive_sum1d_f32";
  if (libSym == "cubSegmentedInclusiveProduct2D_f32_tensor")
    return "polygeist_cub_segmented_inclusive_product2d_f32";
  if (libSym == "cubExclusiveSum1D_i32_memref")
    return "polygeist_cub_exclusive_sum1d_i32";
  if (libSym == "cubCountNonzero1D_f32_tensor")
    return "polygeist_cub_count_nonzero1d_f32";
  if (libSym == "cubSegmentedCountNonzero2D_f32_tensor")
    return "polygeist_cub_segmented_count_nonzero2d_f32";
  if (libSym == "cubEqualAll1D_f32_tensor")
    return "polygeist_cub_equal_all1d_f32";
  if (libSym == "cudnnReduceSum_f32" ||
      libSym == "cudnnReduceProduct_f32" ||
      libSym == "cudnnReduceMin_f32" ||
      libSym == "cudnnReduceMax_f32" ||
      libSym == "cudnnReduceMinMax_f32")
    return "polygeist_cudnn_reduce_f32";
  if (libSym == "cudnnReduceSum_f64")
    return "polygeist_cudnn_reduce_f64";
  if (libSym == "cudnnReduceTrace_f32")
    return "polygeist_cudnn_reduce_diagonal_f32";
  if (libSym == "cubSegmentedPrefixSum_f32")
    return "polygeist_cub_segmented_prefix_sum_f32";
  if (libSym == "cubSegmentedPrefixLogicalAnd_i32")
    return "polygeist_cub_segmented_prefix_logical_and_i32";
  if (libSym.starts_with("cutensorPermute_f32_r") &&
      libSym.ends_with("_tensor"))
    return "polygeist_cutensor_permute_f32";
  if (libSym.starts_with("cubSegmented") && libSym.ends_with("_i32"))
    return "polygeist_cub_segmented_reduce_i32";
  if (libSym == "cubSegmentedLogicalSelect_i32_tensor")
    return "polygeist_cub_segmented_reduce_i32";
  if (libSym == "cublasDgemm") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_simple") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_subtract") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_strided_batched_subtract")
    return "polygeist_cublas_dgemm_strided_batched_subtract";
  if (libSym == "cublasDgemv_strided_batched_subtract")
    return "polygeist_cublas_dgemv_strided_batched_subtract";
  if (libSym == "cublasDgemm_alpha_only") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_zero") return "polygeist_cublas_dgemm";
  if (libSym == "cublasSgemm_nn" || libSym == "cublasSgemm_nn_zero" ||
      libSym == "cublasSgemm_nt_zero" || libSym == "cublasSgemm_tn_zero" ||
      libSym == "cublasSgemm_tt_zero" ||
      libSym == "cublasSgemm_nt" ||
      libSym == "cublasSgemm_tn" || libSym == "cublasSgemm_tt")
    return "polygeist_cublas_sgemm_transpose";
  if (libSym.starts_with("cublasSgemm_") &&
      (libSym.ends_with("_alpha") || libSym.ends_with("_alpha_beta")))
    return "polygeist_cublas_sgemm_transpose";
  if (libSym == "cublasSgemm_strided_batched_nn_zero")
    return "polygeist_cublas_sgemm_strided_batched";
  if (libSym == "cublasSgemm_broadcast3d_simple")
    return "polygeist_cublas_sgemm";
  if (libSym == "cublasSgemm_broadcast3d_memref")
    return "polygeist_cublas_sgemm";
  if (libSym == "cublasSgemm_flat_colmajor_nt_alpha_beta")
    return "polygeist_cublas_sgemm_transpose";
  if (libSym == "cublasSgemm_strided_batched_broadcast_rhs")
    return "polygeist_cublas_sgemm_strided_batched_broadcast_rhs";
  if (libSym == "cublasDgeam_scale2D") return "polygeist_cublas_dscal_2d";
  if (libSym == "memset_zero_2D") return "polygeist_cublas_memset_zero_2d";
  if (libSym == "memset_zero_2D_f32")
    return "polygeist_cublas_memset_zero_2d_f32";
  if (libSym == "memset_zero_1D") return "polygeist_cublas_memset_zero_1d";
  if (libSym == "memset_zero_1D_f32")
    return "polygeist_cublas_memset_zero_1d_f32";
  if (libSym == "cublasDgemv") return "polygeist_cublas_dgemv";
  if (libSym == "cublasDgemv_T") return "polygeist_cublas_dgemv_T";
  if (libSym == "cublasDgemv_T_zero") return "polygeist_cublas_dgemv_T";
  if (libSym == "cublasDgemv_subtract") return "polygeist_cublas_dgemv";
  if (libSym == "cublasDgemv_subtract_T") return "polygeist_cublas_dgemv_T";
  if (libSym == "cublasSgemv") return "polygeist_cublas_sgemv";
  if (libSym == "cublasSgemv_T") return "polygeist_cublas_sgemv_T";
  if (libSym == "cublasDgemv_alpha") return "polygeist_cublas_dgemv_alpha";
  if (libSym == "cublasDaxpby") return "polygeist_cublas_daxpby";
  if (libSym == "cublasSaxpby") return "polygeist_cublas_saxpby";
  if (libSym == "cublasSscal") return "polygeist_cublas_sscal";
  if (libSym == "cublasDaxpy_unit") return "polygeist_cublas_daxpy_unit";
  if (libSym == "cublasDger_rank2") return "polygeist_cublas_dger_rank2";
  if (libSym == "cublasDgemm_outer_product")
    return "polygeist_cublas_dgemm_outer_product";
  if (libSym == "cudnnConvolution2D_9tap")
    return "polygeist_cudnn_conv2d_polybench9tap";
  if (libSym == "cudnnConvolution2D_9tap_f32")
    return "polygeist_cudnn_conv2d_3x3_f32";
  if (libSym == "cudnnConvolution2D_9tap_f16")
    return "polygeist_cudnn_conv2d_3x3_f16";
  if (libSym == "cudnnConvolution2D_9tap_bf16")
    return "polygeist_cudnn_conv2d_3x3_bf16";
  if (libSym == "cudnnConvolution2D_9tap_i32")
    return "polygeist_cudnn_conv2d_3x3_i32";
  if (libSym == "cudnnConvolution2D_25tap")
    return "polygeist_cudnn_conv2d_5x5_f64";
  if (libSym == "cudnnConvolution2D_25tap_f32")
    return "polygeist_cudnn_conv2d_5x5_f32";
  if (libSym == "cudnnConvolution2D_ntap")
    return "polygeist_cudnn_conv2d_ntap_f64";
  if (libSym == "cudnnConvolution2D_ntap_f32")
    return "polygeist_cudnn_conv2d_ntap_f32";
  if (libSym == "cudnnConvolution2D_ntap_tensor")
    return "polygeist_cudnn_conv2d_ntap_f64";
  if (libSym == "cudnnConvolution2D_ntap_f32_tensor")
    return "polygeist_cudnn_conv2d_ntap_f32";
  if (libSym == "cudnnConvolution3D_ntap_tensor")
    return "polygeist_cudnn_conv3d_ntap_f64";
  if (libSym == "cudnnConvolution3D_ntap_f32_tensor")
    return "polygeist_cudnn_conv3d_ntap_f32";
  if (libSym == "cudnnStencil3D7pt_f32_flat_tensor")
    return "polygeist_cudnn_stencil3d_7pt_f32_flat";
  if (libSym == "cudnnConvolution3D_f32" ||
      libSym == "cudnnConvolution3D_f32_bias")
    return "polygeist_cudnn_conv3d_channels_f32";
  if (libSym == "cudnnConvolution1D_f32_bias")
    return "polygeist_cudnn_conv1d_bias_f32";
  if (libSym == "cudnnConvolution2D_f32_dilated")
    return "polygeist_cudnn_conv2d_dilated_f32";
  if (libSym == "cublasGemmEx_i8_i32_tensor")
    return "polygeist_cublas_gemmex_i8_i32";
  if (libSym == "cublasSnrm2_f32_memref")
    return "polygeist_cublas_snrm2_f32";
  if (libSym == "cublasJointMaxAbsProduct_f32_memref")
    return "polygeist_cublas_joint_maxabs_product_f32";
  if (libSym == "cudnnFeatureMaskScale_f32_tensor")
    return "polygeist_cudnn_feature_mask_scale_f32";
  if (libSym == "cudnnConvolutionTranspose2D_f32_memref")
    return "polygeist_cudnn_conv_transpose2d_f32";
  if (libSym == "cudnnConvolutionTranspose3D_f32_memref")
    return "polygeist_cudnn_conv_transpose3d_f32";
  if (libSym == "cudnnConvolutionBackwardFilter3D_f32_memref")
    return "polygeist_cudnn_conv_backward_filter3d_f32";
  if (libSym == "cudnnDepthwiseConvolution2D_f32_memref")
    return "polygeist_cudnn_depthwise_conv2d_f32";
  if (libSym == "cutensorKroneckerProduct2D_f32_memref")
    return "polygeist_cutensor_kronecker_product2d_f32";
  if (libSym == "cudnnBinaryCrossEntropyMean_f32_memref")
    return "polygeist_cudnn_binary_cross_entropy_mean_f32";
  if (libSym == "cudnnConvolutionTBC_f32_memref")
    return "polygeist_cudnn_conv_tbc_f32";
  if (libSym == "cudnnConvolutionTBCBackward_f32_memref")
    return "polygeist_cudnn_conv_tbc_backward_f32";
  if (libSym == "cudnnTransformBiasRescaleQKV_f32_memref")
    return "polygeist_cudnn_transform_bias_rescale_qkv_f32";
  if (libSym == "cudnnAddrElementwise_f32_memref")
    return "polygeist_cudnn_addr_elementwise_f32";
  if (libSym == "cudnnLogSigmoid_f32_memref")
    return "polygeist_cudnn_log_sigmoid_f32";
  if (libSym == "cubSegmentedLogicalAnd_i32_memref" ||
      libSym == "cubSegmentedLogicalSelect_i32_memref")
    return "polygeist_cub_segmented_reduce_i32";
  if (libSym == "cubSegmentedPrefixSum_f32_memref")
    return "polygeist_cub_segmented_prefix_sum_f32";
  if (libSym == "cubSegmentedPrefixLogicalAnd_i32_memref")
    return "polygeist_cub_segmented_prefix_logical_and_i32";
  if (libSym == "cubSegmentedSum_f32_memref" ||
      libSym == "cubSegmentedMin_f32_memref" ||
      libSym == "cubSegmentedMax_f32_memref")
    return "polygeist_cub_segmented_reduce_f32";
  if (libSym == "cubSegmentedSum_f64_memref")
    return "polygeist_cub_segmented_reduce_f64";
  if (libSym == "cubSegmentedBitXor_i32_memref")
    return "polygeist_cub_segmented_reduce_i32";
  if (libSym == "cublasSdot_memref" || libSym == "cublasDdot_memref")
    return libSym == "cublasSdot_memref" ? "polygeist_cublas_dot_f32"
                                         : "polygeist_cublas_dot_f64";
  if (libSym == "cubSegmentedArgMax_f32_i32_memref" ||
      libSym == "cubSegmentedArgMin_f32_i32_memref")
    return "polygeist_cub_segmented_argreduce_f32";
  if (libSym == "cublasSgemvTZero_memref")
    return "polygeist_cublas_sgemv_T";
  if (libSym == "cudnnSinc_f32_memref")
    return "polygeist_cudnn_sinc_f32";
  if (libSym == "cubSegmentedSortDescending_f32_i32_memref" ||
      libSym == "cubSegmentedTopKDescending_f32_i32_memref")
    return "polygeist_cub_segmented_sort_descending_f32_i32";
  if (libSym == "cubSegmentReduceLengths_f32_memref")
    return "polygeist_cub_segment_reduce_lengths_f32";
  if (libSym == "cufftZ2Z_1D_tensor")
    return "polygeist_cufft_z2z_1d";
  if (libSym == "cufftC2C_1D_tensor")
    return "polygeist_cufft_c2c_1d";
  if (libSym == "cutensornetTensorProduct3D_f32_tensor")
    return "polygeist_cutensornet_tensor_product_3d_f32";
  if (libSym == "cutensornetTensorProduct3D_f64_tensor")
    return "polygeist_cutensornet_tensor_product_3d_f64";
  if (libSym == "cutensornetContraction2_f64" ||
      libSym == "cutensornetContraction2_f64_r4r5r4" ||
      libSym == "cutensornetContraction2_f64_r5r4r4" ||
      libSym == "cutensornetContraction2_f64_r5r5r4")
    return "polygeist_cutensornet_contraction2_f64";
  if (libSym.starts_with("cutensornetNetwork_f32"))
    return "polygeist_cutensornet_network_f32";
  if (libSym.starts_with("cutensornetNetwork_f64"))
    return "polygeist_cutensornet_network_f64";
  // NOTE: cudnnConvolution2D_9tap_i{8,16} are intentionally absent — those
  // launches route to PVA Solutions' libpva_operator and are lowered by
  // a separate pass (see LowerKernelLaunchToPVA.cpp). cuDNN itself has
  // no working standalone INT8/INT16 forward-conv kernel on Orin.
  // Extracted-darknet batched CNN-block primitives. All four take their
  // 4D tensors through `polygeist.submap` views (the implicit im2col for
  // conv, the broadcast onto the 4D iteration domain for batchnorm, etc.)
  // — the lowering walks each submap operand back to the underlying base
  // memref before extracting the data pointer.
  if (libSym == "cudnnConvolutionFwd_batched")
    return "polygeist_cudnn_conv2d_batched";
  if (libSym == "cudnnConvolution2DWindow_f32")
    return "polygeist_cudnn_conv2d_uniform_window_f32";
  if (libSym.starts_with("cudnnAdaptivePool_f32_") ||
      libSym.starts_with("cudnnAveragePool_f32_") ||
      libSym == "cudnnAvgPoolWindow_f32")
    return "polygeist_cudnn_adaptive_pool_f32";
  if (libSym == "cudnnBatchNormBackward_f32_full" ||
      libSym == "cudnnBatchNormBackward_f32_dx")
    return "polygeist_cudnn_batchnorm_backward_f32";
  if (libSym == "cudnnConvolutionFwd_im2col_gemm")
    return "polygeist_cudnn_conv2d_im2col_gemm_f32";
  if (libSym == "cudnnMaxPoolFwd_batched")
    return "polygeist_cudnn_maxpool_batched";
  if (libSym == "cudnnBatchNormalizationForwardInference")
    return "polygeist_cudnn_batchnorm_inference";
  if (libSym == "cudnnAddTensor_batched")
    return "polygeist_cudnn_add_tensor_batched";
  if (libSym == "cudnnConvBnReluFwdFused")
    return "polygeist_cudnn_conv_bn_relu_fused";
  if (libSym == "cudnnConvBiasReluAddFwdFused")
    return "polygeist_cudnn_conv_bias_relu_add_fused";
  if (libSym == "whisperExpShiftSum_f32_tensor")
    return "polygeist_whisper_exp_shift_sum_f32";
  if (libSym == "cublasSdot")
    return "polygeist_cublas_dot_f32";
  if (libSym == "cublasDdot")
    return "polygeist_cublas_dot_f64";
  if (libSym == "cudnnSoftmaxForward")
    return "polygeist_cudnn_softmax_forward_f32";
  if (libSym == "cudnnSoftmaxForward_tensor")
    return "polygeist_cudnn_softmax_forward_f32";
  if (libSym == "cudnnSoftmaxForwardOut_tensor")
    return "polygeist_cudnn_softmax_forward_out_f32";
  if (libSym == "cudaCopy1D_f32_tensor" ||
      libSym == "cudaCopy2D_f32_tensor" ||
      libSym == "cudaCopy3D_f32_tensor" ||
      libSym == "cudaCopy6D_f32_tensor")
    return "polygeist_cuda_copy_f32";
  if (libSym == "cublasBroadcastAxis0_f32" ||
      libSym == "cublasBroadcastAxis1_f32")
    return "polygeist_cublas_broadcast_1d_to_2d_f32";
  if (libSym == "cudaAdd_f32_tensor")
    return "polygeist_cuda_add_f32";
  if (libSym == "cudaMaskSelect_f32_tensor")
    return "polygeist_cuda_mask_select_f32";
  if (libSym == "cudaSwiGLU_f32_tensor")
    return "polygeist_cuda_swiglu_f32";
  if (libSym == "cudaRopeMulMulSub_f32_tensor" ||
      libSym == "cudaRopeMulMulAdd_f32_tensor")
    return "polygeist_cuda_rope_mulmul_f32";
  if (libSym == "cublasLtMatmulBiasReluFused")
    return "polygeist_cublaslt_matmul_bias_relu";
  if (libSym == "cublasDsyrk_alias")
    return "polygeist_cublas_dsyrk";
  if (libSym == "cublasGemmFor1x1Conv")
    return "polygeist_cublas_sgemm_1x1conv";
  return StringRef();
}

static std::optional<int32_t> cutensorUnaryOpId(StringRef libSym) {
  if (!libSym.starts_with("cutensorUnary_") || !libSym.ends_with("_f32"))
    return std::nullopt;
  StringRef op = libSym.drop_front(14).drop_back(4);
  static constexpr StringLiteral names[] = {
      "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
      "cos", "cosh", "exp", "floor", "log", "mish", "neg", "reciprocal",
      "relu", "sigmoid", "silu", "sin", "sinh", "sqrt", "tan", "tanh"};
  for (int32_t i = 0;
       i < static_cast<int32_t>(sizeof(names) / sizeof(names[0])); ++i)
    if (op == names[i])
      return i;
  return std::nullopt;
}

// `ensureShimDecl` and `memrefBasePtr` are shared with the PVA lowering
// pass; their definitions live in KernelLaunchLoweringUtils.cpp.
using mlir::polygeist::ensureShimDecl;
using mlir::polygeist::memrefBasePtr;

// Return an SSA value for the `axis` dimension of memref `m`, as `i32`.
// We use i32 because the shim functions accept int32_t for M/N/K/lda/...
// Static dims emit `arith.constant`; dynamic dims emit `memref.dim`.
static Value memrefDimAsI32(OpBuilder &b, Location loc, Value m, int64_t axis) {
  auto mrType = cast<MemRefType>(m.getType());
  if (!mrType.isDynamicDim(axis)) {
    int64_t v = mrType.getDimSize(axis);
    return b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                        b.getI32IntegerAttr((int32_t)v));
  }
  // A Polygeist submap carries the logical view extents explicitly.  Querying
  // memref.dim on the opaque view strands the submap after kernel.launch has
  // been replaced by a runtime call (upstream mlir-opt cannot lower the
  // Polygeist dialect).  Use that explicit extent directly instead.
  if (auto submap = m.getDefiningOp<polygeist::SubmapOp>()) {
    if (axis >= 0 && static_cast<unsigned>(axis) < submap.getSizes().size())
      return b.create<arith::IndexCastOp>(loc, b.getI32Type(),
                                          submap.getSizes()[axis]);
  }
  Value idx = b.create<arith::ConstantIndexOp>(loc, axis);
  Value dimIdx = b.create<memref::DimOp>(loc, m, idx);
  return b.create<arith::IndexCastOp>(loc, b.getI32Type(), dimIdx);
}

static Value memrefNumElementsAsI32(OpBuilder &b, Location loc, Value m) {
  auto mrType = cast<MemRefType>(m.getType());
  Value total = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                            b.getI32IntegerAttr(1));
  for (int64_t axis = 0; axis < mrType.getRank(); ++axis)
    total = b.create<arith::MulIOp>(loc, total,
                                    memrefDimAsI32(b, loc, m, axis));
  return total;
}

static Value valueAsI32(OpBuilder &b, Location loc, Value v);

static Value integerLikeAsI64(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) {
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
      Value src = cast.getIn();
      if (isa<IntegerType>(src.getType()))
        return integerLikeAsI64(b, loc, src);
    }
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), v);
  }
  if (v.getType().isInteger(64))
    return v;
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() > 64)
      return b.create<arith::TruncIOp>(loc, b.getI64Type(), v);
    return b.create<arith::ExtSIOp>(loc, b.getI64Type(), v);
  }
  return v;
}

static Value opFoldResultAsI64(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t v = cast<IntegerAttr>(attr).getInt();
    return b.create<arith::ConstantOp>(loc, b.getI64Type(),
                                       b.getI64IntegerAttr(v));
  }
  return integerLikeAsI64(b, loc, cast<Value>(ofr));
}

static Value opFoldResultAsI32(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t v = cast<IntegerAttr>(attr).getInt();
    return b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                       b.getI32IntegerAttr((int32_t)v));
  }
  return valueAsI32(b, loc, cast<Value>(ofr));
}

static Value valueAsI32(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex())
    return b.create<arith::IndexCastOp>(loc, b.getI32Type(), v);
  if (v.getType().isInteger(32))
    return v;
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() > 32)
      return b.create<arith::TruncIOp>(loc, b.getI32Type(), v);
    return b.create<arith::ExtSIOp>(loc, b.getI32Type(), v);
  }
  return v;
}

// Recover the backing memref whenever a tensor is only an ABI/view wrapper.
// Library calls are opaque to one-shot-bufferize: blindly emitting
// bufferization.to_memref here can allocate and copy an entire operand before
// the call.  It also makes an otherwise valid cudaMalloc-backed C ABI unsafe,
// because that compiler-generated copy executes on the host.  Keep direct
// to_tensor values and extract_slice views as aliases of their source memrefs;
// materialize only genuine tensor SSA values with no recoverable provenance.
static Value tensorToMemref(OpBuilder &b, Location loc, Value t) {
  Value stripped = t;
  for (int hops = 0; hops < 8; ++hops) {
    if (auto cast = stripped.getDefiningOp<tensor::CastOp>()) {
      stripped = cast.getSource();
      continue;
    }
    break;
  }
  if (auto toTensor = stripped.getDefiningOp<bufferization::ToTensorOp>())
    return toTensor.getMemref();
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    Value source = slice.getSource();
    for (int hops = 0; hops < 8; ++hops) {
      if (auto cast = source.getDefiningOp<tensor::CastOp>()) {
        source = cast.getSource();
        continue;
      }
      break;
    }
    if (auto toTensor =
            source.getDefiningOp<bufferization::ToTensorOp>()) {
      auto srcType = cast<MemRefType>(toTensor.getMemref().getType());
      auto resultType = cast<MemRefType>(
          memref::SubViewOp::inferRankReducedResultType(
              slice.getType().getShape(), srcType, slice.getMixedOffsets(),
              slice.getMixedSizes(), slice.getMixedStrides()));
      return b.create<memref::SubViewOp>(
          loc, resultType, toTensor.getMemref(), slice.getMixedOffsets(),
          slice.getMixedSizes(), slice.getMixedStrides());
    }
  }
  auto tt = cast<RankedTensorType>(t.getType());
  auto memrefType = MemRefType::get(tt.getShape(), tt.getElementType());
  return b.create<bufferization::ToMemrefOp>(loc, memrefType, t);
}

static Value valueToMemref(OpBuilder &b, Location loc, Value v) {
  if (isa<MemRefType>(v.getType()))
    return v;
  return tensorToMemref(b, loc, v);
}

// Snapshot a memref that has just been mutated through an opaque LLVM pointer.
// The pointer passed to an external runtime call hides the mutation from
// one-shot-bufferize.  Without an explicit memory operation, bufferization may
// legally reuse the same allocation for a later tensor.empty even while the
// launch result is live.  A copy into a fresh memref makes the produced SSA
// value and its lifetime visible to buffer alias analysis.
static Value snapshotOpaqueCallResult(OpBuilder &b, Location loc, Value source) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<Value, 4> dynamicSizes;
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    if (!type.isDynamicDim(dim))
      continue;
    Value axis = b.create<arith::ConstantIndexOp>(loc, dim);
    dynamicSizes.push_back(b.create<memref::DimOp>(loc, source, axis));
  }
  Value snapshot = b.create<memref::AllocOp>(loc, type, dynamicSizes);
  b.create<memref::CopyOp>(loc, source, snapshot);
  return snapshot;
}

static ShapedType getRankedShapedType(Value v) {
  if (auto t = dyn_cast<RankedTensorType>(v.getType()))
    return t;
  if (auto m = dyn_cast<MemRefType>(v.getType()))
    return m;
  return ShapedType();
}

static Value stripTensorCasts(Value v) {
  for (int hops = 0; hops < 8; ++hops) {
    if (auto cast = v.getDefiningOp<tensor::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

static bufferization::ToTensorOp sourceToTensorOp(Value tensorValue) {
  Value v = stripTensorCasts(tensorValue);
  for (int hops = 0; hops < 8; ++hops) {
    if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>())
      return toTensor;
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto loop = dyn_cast_or_null<affine::AffineForOp>(
          blockArg.getOwner()->getParentOp());
      unsigned number = blockArg.getArgNumber();
      if (loop && number > 0 && number <= loop.getInits().size()) {
        v = stripTensorCasts(loop.getInits()[number - 1]);
        continue;
      }
    }
    break;
  }
  return nullptr;
}

// Follow destination-style tensor updates to the ABI buffer they logically
// update.  This is intentionally output-only: following tensor.insert for a
// read operand would discard the inserted value.  Reduction matchers commonly
// initialize an output element with tensor.insert and then pass a rank-reduced
// extract_slice of that value to a library call which overwrites it.
static bufferization::ToTensorOp destinationToTensorOp(Value tensorValue) {
  Value v = stripTensorCasts(tensorValue);
  for (int hops = 0; hops < 8; ++hops) {
    if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>())
      return toTensor;
    if (auto insert = v.getDefiningOp<tensor::InsertOp>()) {
      v = stripTensorCasts(insert.getDest());
      continue;
    }
    if (auto insertSlice = v.getDefiningOp<tensor::InsertSliceOp>()) {
      v = stripTensorCasts(insertSlice.getDest());
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto loop = dyn_cast_or_null<affine::AffineForOp>(
          blockArg.getOwner()->getParentOp());
      unsigned number = blockArg.getArgNumber();
      if (loop && number > 0 && number <= loop.getInits().size()) {
        v = stripTensorCasts(loop.getInits()[number - 1]);
        continue;
      }
    }
    break;
  }
  return nullptr;
}

static Value sliceSourceMemref(Value tensorValue) {
  Value v = stripTensorCasts(tensorValue);
  auto slice = v.getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice) return Value();
  auto toTensor = sourceToTensorOp(slice.getSource());
  if (!toTensor) return Value();
  return toTensor.getMemref();
}

static Value valueToMemrefPreservingSlice(OpBuilder &b, Location loc, Value v);
static Value memrefToTensor(OpBuilder &b, Location loc, Value m,
                            Type tensorType);

// Return the address of logical element zero of a flattened affine submap.
// C frontends commonly describe a 3-D view over a flat pointer with a map
// such as ((d2 + 1) * ny + d1 + 1) * nx + d0 + 1.  Materializing that tensor
// view leaves polygeist.submap operations after an opaque library call.  The
// runtime only needs the starting address, so evaluate the map at zero view
// coordinates and add that element offset to the flat ABI buffer directly.
static Value pointerForFlatAffineSubmap(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
  auto submap = stripped.getDefiningOp<polygeist::SubmapOp>();
  if (!submap || submap.getMap().getNumResults() != 1)
    return Value();
  auto baseTensor = sourceToTensorOp(submap.getBase());
  if (!baseTensor)
    return Value();
  Value base = baseTensor.getMemref();
  auto baseTy = dyn_cast<MemRefType>(base.getType());
  if (!baseTy || baseTy.getRank() != 1)
    return Value();

  SmallVector<Value> mapOperands;
  mapOperands.reserve(submap.getMap().getNumInputs());
  for (unsigned i = 0; i < submap.getMap().getNumDims(); ++i)
    mapOperands.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
  mapOperands.append(submap.getSymbols().begin(), submap.getSymbols().end());
  Value viewOffset = b.create<affine::AffineApplyOp>(
      loc, submap.getMap(), mapOperands);

  Value alignedIdx =
      b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, base);
  Value alignedI64 =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);
  auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, base);
  Value baseOffset = integerLikeAsI64(b, loc, metadata.getOffset());
  Value totalOffset = b.create<arith::AddIOp>(
      loc, baseOffset, integerLikeAsI64(b, loc, viewOffset));
  unsigned bits = baseTy.getElementType().getIntOrFloatBitWidth();
  Value eltBytes = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
  Value byteOffset = b.create<arith::MulIOp>(loc, totalOffset, eltBytes);
  Value address = b.create<arith::AddIOp>(loc, alignedI64, byteOffset);
  return b.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(b.getContext()), address);
}

static Value pointerForTensorOrMemref(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
  if (Value pointer = pointerForFlatAffineSubmap(b, loc, stripped))
    return pointer;
  if (auto toTensor = sourceToTensorOp(stripped))
    return memrefBasePtr(b, loc, toTensor.getMemref());
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (auto toTensor = sourceToTensorOp(slice.getSource())) {
      Value base = toTensor.getMemref();
      auto baseTy = cast<MemRefType>(base.getType());
      Value alignedIdx =
          b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, base);
      Value alignedI64 = b.create<arith::IndexCastOp>(
          loc, b.getI64Type(), alignedIdx);
      auto md = b.create<memref::ExtractStridedMetadataOp>(loc, base);
      Value linear = integerLikeAsI64(b, loc, md.getOffset());
      auto offsets = slice.getMixedOffsets();
      for (int64_t i = 0, e = offsets.size(); i < e; ++i) {
        Value off = opFoldResultAsI64(b, loc, offsets[i]);
        Value stride = integerLikeAsI64(b, loc, md.getStrides()[i]);
        Value scaled = b.create<arith::MulIOp>(loc, off, stride);
        linear = b.create<arith::AddIOp>(loc, linear, scaled);
      }
      unsigned bits = baseTy.getElementType().getIntOrFloatBitWidth();
      Value eltBytes = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
      Value byteOff = b.create<arith::MulIOp>(loc, linear, eltBytes);
      Value byteAddr = b.create<arith::AddIOp>(loc, alignedI64, byteOff);
      auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
      return b.create<LLVM::IntToPtrOp>(loc, ptrTy, byteAddr);
    }
  }

  Value mr = valueToMemrefPreservingSlice(b, loc, v);
  return memrefBasePtr(b, loc, mr);
}

// Return the address of logical element zero, including a subview's strided
// metadata offset. memrefBasePtr intentionally returns the allocation base,
// which is insufficient for an interior pointwise destination.
static Value memrefDataPtr(OpBuilder &b, Location loc, Value mr) {
  // Compute the address of logical element zero directly from a Polygeist
  // submap.  Runtime library ABIs need only the pointer and logical extents;
  // keeping the semantic view alive after launch lowering prevents the rest
  // of the module from reaching standard MLIR.  This handles identity views,
  // fixed-offset slices, and rank-reduced/broadcast scalar views uniformly.
  if (auto submap = mr.getDefiningOp<polygeist::SubmapOp>()) {
    Value base = submap.getBase();
    auto baseType = dyn_cast<MemRefType>(base.getType());
    AffineMap map = submap.getMap();
    if (baseType && map.getNumResults() ==
                        static_cast<unsigned>(baseType.getRank())) {
      SmallVector<Value> mapOperands;
      mapOperands.reserve(map.getNumInputs());
      for (unsigned i = 0; i < map.getNumDims(); ++i)
        mapOperands.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
      mapOperands.append(submap.getSymbols().begin(),
                         submap.getSymbols().end());

      Value alignedIdx =
          b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, base);
      Value address =
          b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);
      auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, base);
      Value linear = integerLikeAsI64(b, loc, metadata.getOffset());
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        AffineMap coordinateMap = AffineMap::get(
            map.getNumDims(), map.getNumSymbols(), map.getResult(i),
            b.getContext());
        Value coordinate =
            b.create<affine::AffineApplyOp>(loc, coordinateMap, mapOperands);
        Value stride = integerLikeAsI64(b, loc, metadata.getStrides()[i]);
        Value scaled = b.create<arith::MulIOp>(
            loc, integerLikeAsI64(b, loc, coordinate), stride);
        linear = b.create<arith::AddIOp>(loc, linear, scaled);
      }
      unsigned bits = baseType.getElementType().getIntOrFloatBitWidth();
      Value eltBytes = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
      Value byteOffset = b.create<arith::MulIOp>(loc, linear, eltBytes);
      address = b.create<arith::AddIOp>(loc, address, byteOffset);
      return b.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(b.getContext()), address);
    }
  }
  auto type = cast<MemRefType>(mr.getType());
  Value alignedIdx =
      b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, mr);
  Value alignedI64 =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);
  auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, mr);
  Value offset = integerLikeAsI64(b, loc, metadata.getOffset());
  unsigned bits = type.getElementType().getIntOrFloatBitWidth();
  Value eltBytes = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
  Value byteOffset = b.create<arith::MulIOp>(loc, offset, eltBytes);
  Value address = b.create<arith::AddIOp>(loc, alignedI64, byteOffset);
  return b.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(b.getContext()), address);
}

static Value dimForTensorOrMemrefAsI32(OpBuilder &b, Location loc, Value v,
                                       int64_t axis) {
  Value stripped = stripTensorCasts(v);
  if (auto toTensor = sourceToTensorOp(stripped))
    return memrefDimAsI32(b, loc, toTensor.getMemref(), axis);
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    if ((int64_t)slice.getType().getRank() == (int64_t)slice.getMixedSizes().size())
      return opFoldResultAsI32(b, loc, slice.getMixedSizes()[axis]);
  }
  Value mr = valueToMemrefPreservingSlice(b, loc, v);
  return memrefDimAsI32(b, loc, mr, axis);
}

// Bufferize a tensor value, preserving extract_slice views as memref.subview.
// This avoids handing dynamic tensor.extract_slice / tensor.insert_slice to
// one-shot-bufferize after the launch has already been lowered to a call.
static Value valueToMemrefPreservingSlice(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
  if (auto toTensor = sourceToTensorOp(stripped))
    return toTensor.getMemref();
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (auto toTensor = sourceToTensorOp(slice.getSource())) {
      auto srcType = cast<MemRefType>(toTensor.getMemref().getType());
      auto resultType = cast<MemRefType>(
          memref::SubViewOp::inferRankReducedResultType(
              slice.getType().getShape(), srcType, slice.getMixedOffsets(),
              slice.getMixedSizes(), slice.getMixedStrides()));
      return b.create<memref::SubViewOp>(
          loc, resultType, toTensor.getMemref(), slice.getMixedOffsets(),
          slice.getMixedSizes(), slice.getMixedStrides());
    }
  }
  if (isa<MemRefType>(v.getType()))
    return v;
  return tensorToMemref(b, loc, v);
}

static Value valueToOutputMemrefPreservingSlice(OpBuilder &b, Location loc,
                                                 Value v) {
  Value stripped = stripTensorCasts(v);
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (auto toTensor = destinationToTensorOp(slice.getSource())) {
      auto srcType = cast<MemRefType>(toTensor.getMemref().getType());
      auto resultType = cast<MemRefType>(
          memref::SubViewOp::inferRankReducedResultType(
              slice.getType().getShape(), srcType, slice.getMixedOffsets(),
              slice.getMixedSizes(), slice.getMixedStrides()));
      return b.create<memref::SubViewOp>(
          loc, resultType, toTensor.getMemref(), slice.getMixedOffsets(),
          slice.getMixedSizes(), slice.getMixedStrides());
    }
  }
  return valueToMemrefPreservingSlice(b, loc, v);
}

static Value tensorForOutputSliceSource(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
  auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice)
    return Value();
  auto toTensor = destinationToTensorOp(slice.getSource());
  if (!toTensor)
    return Value();
  auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
  if (!sourceType)
    return Value();
  return memrefToTensor(b, loc, toTensor.getMemref(), sourceType);
}

// Inverse of the above — wrap a memref back into a tensor for downstream
// SSA uses. The `restrict` + `writable` attributes promise this is the
// only alias of the memref, which is true for fresh launch results.
static Value memrefToTensor(OpBuilder &b, Location loc, Value m, Type tensorType) {
  auto mrType = dyn_cast<MemRefType>(m.getType());
  auto rankedTensor = dyn_cast<RankedTensorType>(tensorType);
  if (mrType && rankedTensor) {
    auto expected = MemRefType::get(
        rankedTensor.getShape(), rankedTensor.getElementType(),
        mrType.getLayout(), mrType.getMemorySpace());
    if (expected != mrType &&
        memref::CastOp::areCastCompatible(mrType, expected))
      m = b.create<memref::CastOp>(loc, expected, m);
  }
  auto t = b.create<bufferization::ToTensorOp>(
      loc, tensorType, m, /*restrict=*/true, /*writable=*/true);
  return t.getResult();
}

static Value tensorForSliceSource(OpBuilder &b, Location loc, Value tensorValue) {
  Value v = stripTensorCasts(tensorValue);
  auto slice = v.getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice) return Value();
  Value src = stripTensorCasts(slice.getSource());
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  Value srcMr = sliceSourceMemref(v);
  if (!srcTy || !srcMr) return Value();
  return memrefToTensor(b, loc, srcMr, srcTy);
}

static void rewireTensorSliceLaunchResult(LaunchOp launch,
                                          Value updatedViewTensor,
                                          Value updatedBaseTensor,
                                          unsigned resultIndex = 0) {
  if (launch.getNumResults() <= resultIndex) return;
  Value res = launch.getResult(resultIndex);
  SmallVector<tensor::InsertSliceOp> inserts;
  SmallVector<tensor::CastOp> resultCasts;
  if (updatedBaseTensor) {
    // Dynamic library signatures commonly expose an unranked launch result,
    // so the destination-style write-back is reached through one or more
    // tensor.cast operations.  Treat those casts as transparent when finding
    // the terminal insert_slice; otherwise one-shot bufferization later
    // materializes a full output snapshot and copy even though the runtime
    // already wrote the destination allocation in place.
    SmallVector<Value> worklist{res};
    llvm::SmallPtrSet<Operation *, 4> seenCasts;
    for (size_t i = 0; i < worklist.size(); ++i) {
      Value candidate = worklist[i];
      for (Operation *user : candidate.getUsers()) {
        if (auto insert = dyn_cast<tensor::InsertSliceOp>(user)) {
          if (insert.getSource() == candidate)
            inserts.push_back(insert);
          continue;
        }
        if (auto cast = dyn_cast<tensor::CastOp>(user)) {
          if (cast.getSource() == candidate && seenCasts.insert(cast).second) {
            resultCasts.push_back(cast);
            worklist.push_back(cast.getResult());
          }
        }
      }
    }
  }
  for (auto insert : inserts) {
    insert.getResult().replaceAllUsesWith(updatedBaseTensor);
    insert.erase();
  }
  for (tensor::CastOp cast : llvm::reverse(resultCasts))
    if (cast.getResult().use_empty())
      cast.erase();
  if (!res.use_empty() && updatedViewTensor)
    res.replaceAllUsesWith(updatedViewTensor);
}

// Walk a SSA value back through `polygeist.submap` / `polygeist.submapInverse`
// to its underlying base tensor. The matcher's launches feed operands
// through view chains (the 7D strided-window for conv im2col, the 4D
// broadcast of a 1D per-channel vector for batchnorm, etc.). Earlier
// matched launches in the same function can ALSO have introduced a
// submapInverse via their own in-place semantics — composing two
// launches whose outputs alias makes the chain ≥ 2 levels deep.
//
// Rules:
//   • polygeist.submap → walk to its `base`
//   • polygeist.submapInverse → walk to its FIRST operand (the base
//     tensor it scatters back into; conceptually, after the inverse
//     scatter, the underlying base IS the up-to-date tensor).
// Returns `v` unchanged if neither defining op applies, including when
// `v` is a function argument or a bufferization.to_tensor.
static Value resolveSubmapBase(Value v) {
  for (int hops = 0; hops < 16; ++hops) {
    if (auto submap = v.getDefiningOp<polygeist::SubmapOp>()) {
      v = submap.getBase();
      continue;
    }
    if (auto inv = v.getDefiningOp<polygeist::SubmapInverseOp>()) {
      // First operand is the underlying base; SubmapInverseOp doesn't
      // expose a getBase() accessor, so use getOperand(0).
      v = inv.getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

// After lowering an in-place launch (the runtime shim mutates the output
// memref directly), we need to wire downstream consumers to the new
// "updated base tensor" SSA. There are two patterns:
//
//   (a) Output operand was a polygeist.submap view of the underlying 4D
//       base. The launch's result has the *view* type and is consumed by
//       polygeist.submapInverse(base, result, ...) which scatters back
//       to a 4D tensor. We replace the submapInverse's result with the
//       updated 4D base tensor and erase the inverse.
//
//   (b) Output operand was already the 4D base tensor (no submap on the
//       output). The launch's result has the 4D base type, consumed
//       directly by bufferization.to_memref / etc. We replace
//       launch.getResult(0) uses with the updated base tensor.
//
// The caller's `updatedBaseTensor` is a `bufferization.to_tensor` of the
// freshly-bufferised output memref — same 4D type as the base.
static void rewireLaunchResult(LaunchOp launch, Value updatedBaseTensor) {
  if (launch.getNumResults() == 0) return;
  Value res = launch.getResult(0);

  // Case (a): submapInverse consumer — replace its result instead, so
  // we collapse both the inverse and the launch out of the IR.
  SmallVector<polygeist::SubmapInverseOp> inverses;
  for (Operation *user : res.getUsers()) {
    if (auto inv = dyn_cast<polygeist::SubmapInverseOp>(user))
      inverses.push_back(inv);
  }
  for (auto inv : inverses) {
    inv.getResult().replaceAllUsesWith(updatedBaseTensor);
    inv.erase();
  }

  // Case (b): any remaining consumers of the launch result expect the
  // launch's result type. If the launch result is the same type as the
  // base tensor (output wasn't a submap), this `replaceAllUsesWith` is
  // type-safe and wires to_memref / memref.copy / etc. to the
  // bufferized base. If the launch result is a *view* type and there
  // are still consumers other than the inverses we just erased, the
  // caller's invariants are violated — fail loudly so we notice.
  if (!res.use_empty()) {
    if (res.getType() != updatedBaseTensor.getType()) {
      launch.emitWarning(
          "lowering: launch result has view type with non-submapInverse "
          "consumer; downstream verifier may complain about the type "
          "of the in-place updated tensor");
    }
    res.replaceAllUsesWith(updatedBaseTensor);
  }
}

// Variant of rewireLaunchResult for an in-place launch whose destination is a
// polygeist.submap.  The runtime mutates (and the caller snapshots) the base
// allocation, but ordinary SSA consumers of the launch result still expect
// the shaped view type.  Recreate that same view over the updated base while
// continuing to bypass submapInverse consumers with the full updated base.
static LogicalResult rewireSubmapLaunchResult(LaunchOp launch,
                                               Value updatedViewTensor,
                                               Value updatedBaseTensor) {
  if (launch.getNumResults() == 0)
    return success();
  Value res = launch.getResult(0);

  SmallVector<polygeist::SubmapInverseOp> inverses;
  for (Operation *user : res.getUsers())
    if (auto inverse = dyn_cast<polygeist::SubmapInverseOp>(user))
      inverses.push_back(inverse);
  for (polygeist::SubmapInverseOp inverse : inverses) {
    inverse.getResult().replaceAllUsesWith(updatedBaseTensor);
    inverse.erase();
  }

  if (!res.use_empty()) {
    if (!updatedViewTensor || res.getType() != updatedViewTensor.getType())
      return launch.emitError(
          "lowering: cannot reconnect a submap launch result to its updated "
          "view type");
    res.replaceAllUsesWith(updatedViewTensor);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Per-library lowerings
//===----------------------------------------------------------------------===//

// kernel.launch @cublasDgemm(%A, %B, %C, %beta, %alpha)
//   : (tensor<MxKxf64>, tensor<KxNxf64>, tensor<MxNxf64>, f64, f64)
//   -> tensor<MxNxf64>
//
// Lowers to:
//   %A_mr = bufferization.to_memref %A
//   %B_mr = bufferization.to_memref %B
//   %C_mr = bufferization.to_memref %C
//   %M, %N, %K, %lda, %ldb, %ldc = ... (i32 dim queries)
//   func.call @polygeist_cublas_dgemm(%M, %N, %K, %alpha,
//                                        %A_mr, %lda, %B_mr, %ldb,
//                                        %beta, %C_mr, %ldc)
//   %out = bufferization.to_tensor %C_mr restrict writable
//   replaceAllUsesWith(launch.getResult(0), %out)
static LogicalResult lowerDgemm(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 5)
    return launch.emitError("cublasDgemm lowering: expected 5 operands "
                            "(A, B, C, beta, alpha), got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cublasDgemm lowering: expected 1 result");

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  Value beta = launch.getOperand(3);
  Value alpha = launch.getOperand(4);

  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct)
    return launch.emitError(
        "cublasDgemm lowering: A/B/C operands must be ranked tensors");
  if (At.getRank() != 2 || Bt.getRank() != 2 || Ct.getRank() != 2)
    return launch.emitError(
        "cublasDgemm lowering: A/B/C must be 2D tensors");
  if (!At.getElementType().isF64() || !Bt.getElementType().isF64() ||
      !Ct.getElementType().isF64())
    return launch.emitError(
        "cublasDgemm lowering: only f64 element type supported");
  if (!beta.getType().isF64() || !alpha.getType().isF64())
    return launch.emitError(
        "cublasDgemm lowering: alpha/beta must be f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();

  // Bufferize tensors → memrefs (whose ABI carries the data pointer when
  // lowered to LLVM). Do this BEFORE dim queries so we can use memref.dim.
  Value A_mr = tensorToMemref(b, loc, A);
  Value B_mr = tensorToMemref(b, loc, B);
  Value C_mr = valueToOutputMemrefPreservingSlice(b, loc, C);

  // Materialise dim queries on the memrefs (static shape → arith.constant,
  // dynamic shape → memref.dim).
  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value K = memrefDimAsI32(b, loc, A_mr, 1);
  Value N = memrefDimAsI32(b, loc, B_mr, 1);
  // Row-major leading dims: lda = K, ldb = N, ldc = N.
  Value lda = K;
  Value ldb = N;
  Value ldc = N;

  // CRITICAL: do NOT pass memrefs to the C shim — MLIR's --convert-func-to-llvm
  // would expand each memref into 7 LLVM args (alloc-ptr, aligned-ptr, offset,
  // sizes×2, strides×2), but the C shim signature is (M,N,K,alpha,A*,lda,...)
  // with one pointer per matrix. The reg/stack layouts would not match and the
  // shim would read garbage. Extract raw `!llvm.ptr` and pass those.
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value B_ptr = memrefBasePtr(b, loc, B_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  // Forward-declare the shim function with raw-pointer arg types.
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),  // M, N, K
      b.getF64Type(),                                   // alpha
      ptrTy, b.getI32Type(),                            // A*, lda
      ptrTy, b.getI32Type(),                            // B*, ldb
      b.getF64Type(),                                   // beta
      ptrTy, b.getI32Type(),                            // C*, ldc
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dgemm",
                                       argTypes, b);

  SmallVector<Value> callOperands = {M, N, K, alpha, A_ptr, lda, B_ptr, ldb,
                                     beta, C_ptr, ldc};
  b.create<func::CallOp>(loc, shim, callOperands);

  // Recover the result tensor SSA from C_mr (C was updated in place).
  Value resultTensor = memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, resultTensor, tensorForOutputSliceSource(b, loc, C));
  launch.erase();
  return success();
}

// Shared helper: lower a gemm-shape launch with optionally-implicit
// alpha/beta. Variants:
//   @cublasDgemm           operands (A, B, C, beta, alpha)        — full form
//   @cublasDgemm_simple    operands (A, B, C)                     — α=1, β=1
//   @cublasDgemm_alpha_only operands (A, B, C, alpha)             — β=1
// All three lower to the same polygeist_cublas_dgemm runtime call.
static LogicalResult lowerDgemmVariant(LaunchOp launch, ModuleOp module,
                                          StringRef variant) {
  unsigned expected = (variant == "cublasDgemm") ? 5
                    : (variant == "cublasDgemm_alpha_only") ? 4
                    : 3;
  if (launch.getNumOperands() != expected)
    return launch.emitError(variant)
           << " lowering: expected " << expected
           << " operands, got " << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(variant) << " lowering: expected 1 result";

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  Value beta, alpha;
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value one = b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                          b.getF64FloatAttr(1.0));
  if (variant == "cublasDgemm") {
    beta = launch.getOperand(3);
    alpha = launch.getOperand(4);
  } else if (variant == "cublasDgemm_alpha_only") {
    beta = one;
    alpha = launch.getOperand(3);
  } else {  // _simple, subtract, or zero-initialized
    beta = variant == "cublasDgemm_zero"
        ? b.create<arith::ConstantOp>(loc, b.getF64Type(), b.getF64FloatAttr(0.0))
        : one;
    alpha = variant == "cublasDgemm_subtract"
                ? b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                               b.getF64FloatAttr(-1.0))
                : one;
  }

  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 2 || Bt.getRank() != 2 ||
      Ct.getRank() != 2)
    return launch.emitError(variant)
           << " lowering: A/B/C must be 2D ranked tensors";
  if (!At.getElementType().isF64() || !Bt.getElementType().isF64() ||
      !Ct.getElementType().isF64())
    return launch.emitError(variant)
           << " lowering: only f64 supported";

  Value A_mr = tensorToMemref(b, loc, A);
  Value B_mr = tensorToMemref(b, loc, B);
  Value C_mr = tensorToMemref(b, loc, C);
  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value K = memrefDimAsI32(b, loc, A_mr, 1);
  Value N = memrefDimAsI32(b, loc, B_mr, 1);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value B_ptr = memrefBasePtr(b, loc, B_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getF64Type(),
      ptrTy, b.getI32Type(),
      ptrTy, b.getI32Type(),
      b.getF64Type(),
      ptrTy, b.getI32Type(),
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dgemm",
                                       argTypes, b);
  SmallVector<Value> callOperands = {M, N, K, alpha, A_ptr, K /*lda*/,
                                     B_ptr, N /*ldb*/, beta, C_ptr,
                                     N /*ldc*/};
  b.create<func::CallOp>(loc, shim, callOperands);

  Value resultTensor = memrefToTensor(b, loc, C_mr,
                                       launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(resultTensor);
  launch.erase();
  return success();
}

// FP32 row-major GEMM. The two-letter suffix is the semantic transpose state
// proved from linalg indexing maps by the matcher (nn, nt, tn, or tt).
static LogicalResult lowerSgemmTranspose(LaunchOp launch, ModuleOp module,
                                         StringRef variant) {
  StringRef suffix = variant.drop_front(StringRef("cublasSgemm_").size());
  enum class ScalarMode { Simple, AlphaOnly, AlphaBeta };
  ScalarMode scalarMode = ScalarMode::Simple;
  if (suffix.consume_back("_alpha_beta"))
    scalarMode = ScalarMode::AlphaBeta;
  else if (suffix.consume_back("_alpha"))
    scalarMode = ScalarMode::AlphaOnly;
  bool zeroInit = suffix.consume_back("_zero");
  unsigned expectedOperands = scalarMode == ScalarMode::AlphaBeta ? 5
                            : scalarMode == ScalarMode::AlphaOnly ? 4 : 3;
  if (launch.getNumOperands() != expectedOperands ||
      launch.getNumResults() != 1)
    return launch.emitError(variant)
           << ": expected " << expectedOperands << " operands and one result";
  if (suffix.size() != 2 || (suffix[0] != 'n' && suffix[0] != 't') ||
      (suffix[1] != 'n' && suffix[1] != 't'))
    return launch.emitError(variant) << ": invalid transpose suffix";
  bool transA = suffix[0] == 't';
  bool transB = suffix[1] == 't';
  Value A = launch.getOperand(0), B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 2 || Bt.getRank() != 2 ||
      Ct.getRank() != 2 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError(variant) << ": A/B/C must be rank-2 f32 tensors";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, A);
  Value B_mr = tensorToMemref(b, loc, B);
  Value C_mr = tensorToMemref(b, loc, C);
  Value M = memrefDimAsI32(b, loc, C_mr, 0);
  Value N = memrefDimAsI32(b, loc, C_mr, 1);
  Value K = memrefDimAsI32(b, loc, A_mr, transA ? 0 : 1);
  Value lda = memrefDimAsI32(b, loc, A_mr, 1);
  Value ldb = memrefDimAsI32(b, loc, B_mr, 1);
  Value ldc = memrefDimAsI32(b, loc, C_mr, 1);
  Value transAVal = b.create<arith::ConstantIntOp>(loc, transA, 32);
  Value transBVal = b.create<arith::ConstantIntOp>(loc, transB, 32);
  Value one = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                          b.getF32FloatAttr(1.0));
  Value alpha = one;
  Value beta = one;
  if (scalarMode == ScalarMode::AlphaBeta) {
    beta = launch.getOperand(3);
    alpha = launch.getOperand(4);
  } else if (scalarMode == ScalarMode::AlphaOnly) {
    alpha = launch.getOperand(3);
  } else if (zeroInit) {
    beta = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                       b.getF32FloatAttr(0.0));
  }
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), b.getF32Type(),
      ptrTy, b.getI32Type(), ptrTy, b.getI32Type(), b.getF32Type(),
      ptrTy, b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cublas_sgemm_transpose", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{
      M, N, K, transAVal, transBVal, alpha,
      memrefBasePtr(b, loc, A_mr), lda, memrefBasePtr(b, loc, B_mr), ldb,
      beta, memrefBasePtr(b, loc, C_mr), ldc});
  Value out = memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerSgemmStridedBatched(LaunchOp launch,
                                               ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError("batched SGEMM: expected A, B, C and one result");
  Value A = launch.getOperand(0), B = launch.getOperand(1), C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 3 ||
      Ct.getRank() != 3 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError("batched SGEMM: operands must be rank-3 f32 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value Am = tensorToMemref(b, loc, A), Bm = tensorToMemref(b, loc, B);
  Value Cm = tensorToMemref(b, loc, C);
  Value batch = memrefDimAsI32(b, loc, Cm, 0);
  Value M = memrefDimAsI32(b, loc, Cm, 1);
  Value N = memrefDimAsI32(b, loc, Cm, 2);
  Value K = memrefDimAsI32(b, loc, Am, 2);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type(), b.getI32Type(), b.getI32Type(),
                             b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cublas_sgemm_strided_batched", types, b);
  b.create<func::CallOp>(loc, shim, ValueRange{batch, M, N, K,
      memrefBasePtr(b, loc, Am), memrefBasePtr(b, loc, Bm),
      memrefBasePtr(b, loc, Cm)});
  Value out = memrefToTensor(b, loc, Cm, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerDgemmStridedBatchedSubtract(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "batched DGEMM subtract: expected A, B, C and one result");
  Value A = launch.getOperand(0), B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 3 ||
      Ct.getRank() != 3 || !At.getElementType().isF64() ||
      !Bt.getElementType().isF64() || !Ct.getElementType().isF64())
    return launch.emitError(
        "batched DGEMM subtract: operands must be rank-3 f64 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value Am = tensorToMemref(b, loc, A);
  Value Bm = tensorToMemref(b, loc, B);
  Value Cm = tensorToMemref(b, loc, C);
  Value batch = memrefDimAsI32(b, loc, Cm, 0);
  Value M = memrefDimAsI32(b, loc, Cm, 1);
  Value N = memrefDimAsI32(b, loc, Cm, 2);
  Value K = memrefDimAsI32(b, loc, Am, 2);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type(), b.getI32Type(), b.getI32Type(),
                             b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_dgemm_strided_batched_subtract", types, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{batch, M, N, K, memrefBasePtr(b, loc, Am),
                 memrefBasePtr(b, loc, Bm), memrefBasePtr(b, loc, Cm)});
  Value out = memrefToTensor(b, loc, Cm, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerDgemvStridedBatchedSubtract(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "batched DGEMV subtract: expected A, X, Y and one result");
  Value A = launch.getOperand(0), X = launch.getOperand(1);
  Value Y = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Xt = dyn_cast<RankedTensorType>(X.getType());
  auto Yt = dyn_cast<RankedTensorType>(Y.getType());
  if (!At || !Xt || !Yt || At.getRank() != 3 || Xt.getRank() != 2 ||
      Yt.getRank() != 2 || !At.getElementType().isF64() ||
      !Xt.getElementType().isF64() || !Yt.getElementType().isF64())
    return launch.emitError(
        "batched DGEMV subtract: expected rank-3/2/2 f64 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value Am = tensorToMemref(b, loc, A);
  Value Xm = tensorToMemref(b, loc, X);
  Value Ym = tensorToMemref(b, loc, Y);
  Value batch = memrefDimAsI32(b, loc, Ym, 0);
  Value M = memrefDimAsI32(b, loc, Ym, 1);
  Value K = memrefDimAsI32(b, loc, Am, 2);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type(), b.getI32Type(), b.getI32Type(),
                             ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_dgemv_strided_batched_subtract", types, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{batch, M, K, memrefBasePtr(b, loc, Am),
                 memrefBasePtr(b, loc, Xm), memrefBasePtr(b, loc, Ym)});
  Value out = memrefToTensor(b, loc, Ym, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConv2DNtapTensor(LaunchOp launch,
                                                ModuleOp module,
                                                StringRef shimSymbol) {
  if (launch.getNumOperands() != 4)
    return launch.emitError("cudnnConvolution2D_ntap_tensor: expected 4 "
                            "operands (input slice, output slice, weights, K); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvolution2D_ntap_tensor: expected 1 tensor result");

  Value A = launch.getOperand(0);
  Value C = launch.getOperand(1);
  Value W = launch.getOperand(2);
  Value K = launch.getOperand(3);

  auto aTy = dyn_cast<RankedTensorType>(A.getType());
  auto cTy = dyn_cast<RankedTensorType>(C.getType());
  auto wTy = dyn_cast<RankedTensorType>(W.getType());
  auto resTy = dyn_cast<RankedTensorType>(launch.getResult(0).getType());
  if (!aTy || !cTy || !wTy || !resTy)
    return launch.emitError(
        "cudnnConvolution2D_ntap_tensor: operands/result must be tensors");
  if (aTy.getRank() != 2 || cTy.getRank() != 2 || resTy.getRank() != 2 ||
      wTy.getRank() != 1)
    return launch.emitError(
        "cudnnConvolution2D_ntap_tensor: expected 2D input/output and 1D weights");
  Type elemTy = aTy.getElementType();
  if (cTy.getElementType() != elemTy || wTy.getElementType() != elemTy ||
      resTy.getElementType() != elemTy)
    return launch.emitError(
        "cudnnConvolution2D_ntap_tensor: input/output/weights dtypes must match");
  if (!(elemTy.isF64() || elemTy.isF32()))
    return launch.emitError(
        "cudnnConvolution2D_ntap_tensor: only f64/f32 packed weights are supported");
  if (!K.getType().isInteger(32))
    return launch.emitError("cudnnConvolution2D_ntap_tensor: K must be i32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();

  // Preserve tensor.extract_slice views as memref.subview so the runtime sees
  // the same top-left input window and output interior slice as the tensor IR.
  Value A_mr = valueToMemrefPreservingSlice(b, loc, A);
  Value C_mr = valueToOutputMemrefPreservingSlice(b, loc, C);
  Value W_mr = valueToMemrefPreservingSlice(b, loc, W);

  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);
  Value W_ptr = memrefBasePtr(b, loc, W_mr);

  Value oneI32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(1));
  Value border = b.create<arith::SubIOp>(loc, K, oneI32);
  Value outH = memrefDimAsI32(b, loc, C_mr, 0);
  Value outW = memrefDimAsI32(b, loc, C_mr, 1);
  Value M = b.create<arith::AddIOp>(loc, outH, border);
  Value N = b.create<arith::AddIOp>(loc, outW, border);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{M, N, K, W_ptr, A_ptr, C_ptr});

  Value updatedView =
      memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, C);
  rewireTensorSliceLaunchResult(launch, updatedView, updatedBase);
  launch.erase();
  return success();
}

// cuSten consumes a complete dense grid, unlike cuDNN's convolution shim,
// which can consume the materialized top-left/output subviews. Recover the
// underlying submap bases so tensor bufferization does not create compact
// (M-K+1)x(N-K+1) temporaries and then pass those as if they were MxN grids.
static LogicalResult lowerCustenStencil2D(LaunchOp launch, ModuleOp module,
                                          StringRef shimSymbol) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() > 1)
    return launch.emitError(
        "cuSten 2D: expected input view, output view, weights, K and at most "
        "one result");
  Value inputView = launch.getOperand(0);
  Value outputView = launch.getOperand(1);
  Value weights = launch.getOperand(2);
  Value K = launch.getOperand(3);
  if (!K.getType().isInteger(32))
    return launch.emitError("cuSten 2D: K must be i32");

  Value inputBase = resolveSubmapBase(inputView);
  Value outputBase = resolveSubmapBase(outputView);
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, inputBase);
  Value outputMr = valueToOutputMemrefPreservingSlice(b, loc, outputBase);
  Value weightsMr = valueToMemrefPreservingSlice(b, loc, weights);
  auto inputTy = dyn_cast<MemRefType>(inputMr.getType());
  auto outputTy = dyn_cast<MemRefType>(outputMr.getType());
  auto weightsTy = dyn_cast<MemRefType>(weightsMr.getType());
  if (!inputTy || !outputTy || !weightsTy || inputTy.getRank() != 2 ||
      outputTy.getRank() != 2 || weightsTy.getRank() != 1 ||
      !inputTy.getElementType().isF64() ||
      !outputTy.getElementType().isF64() ||
      !weightsTy.getElementType().isF64())
    return launch.emitError("cuSten 2D: requires dense rank-2 f64 grids and "
                            "rank-1 f64 weights");

  Value M = memrefDimAsI32(b, loc, inputMr, 0);
  Value N = memrefDimAsI32(b, loc, inputMr, 1);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{M, N, K, memrefDataPtr(b, loc, weightsMr),
                 memrefDataPtr(b, loc, inputMr),
                 memrefDataPtr(b, loc, outputMr)});

  if (launch.getNumResults() == 1) {
    auto baseTensorTy = dyn_cast<RankedTensorType>(outputBase.getType());
    if (!baseTensorTy)
      return launch.emitError("cuSten tensor result requires a tensor base");
    Value updatedBase = memrefToTensor(b, loc, outputMr, baseTensorTy);
    rewireLaunchResult(launch, updatedBase);
  }
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConv3DNtapTensor(LaunchOp launch,
                                                ModuleOp module,
                                                StringRef shimSymbol) {
  if (launch.getNumOperands() != 4)
    return launch.emitError("cudnnConvolution3D_ntap_tensor: expected 4 "
                            "operands (input, output, weights, K); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvolution3D_ntap_tensor: expected 1 tensor result");

  Value A = launch.getOperand(0);
  Value C = launch.getOperand(1);
  Value W = launch.getOperand(2);
  Value K = launch.getOperand(3);

  auto aTy = dyn_cast<RankedTensorType>(A.getType());
  auto cTy = dyn_cast<RankedTensorType>(C.getType());
  auto wTy = dyn_cast<RankedTensorType>(W.getType());
  auto resTy = dyn_cast<RankedTensorType>(launch.getResult(0).getType());
  if (!aTy || !cTy || !wTy || !resTy)
    return launch.emitError(
        "cudnnConvolution3D_ntap_tensor: operands/result must be tensors");
  if (aTy.getRank() != 3 || cTy.getRank() != 3 || wTy.getRank() != 3 ||
      resTy.getRank() != 3)
    return launch.emitError(
        "cudnnConvolution3D_ntap_tensor: expected 3D input/output/weights");
  Type elemTy = aTy.getElementType();
  if (cTy.getElementType() != elemTy || wTy.getElementType() != elemTy ||
      resTy.getElementType() != elemTy)
    return launch.emitError(
        "cudnnConvolution3D_ntap_tensor: input/output/weights dtypes must match");
  if (!(elemTy.isF64() || elemTy.isF32()))
    return launch.emitError(
        "cudnnConvolution3D_ntap_tensor: only f64/f32 weights are supported");
  if (!K.getType().isInteger(32))
    return launch.emitError("cudnnConvolution3D_ntap_tensor: K must be i32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();

  Value A_mr = valueToMemrefPreservingSlice(b, loc, A);
  Value C_mr = valueToMemrefPreservingSlice(b, loc, C);
  Value W_mr = valueToMemrefPreservingSlice(b, loc, W);

  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);
  Value W_ptr = memrefBasePtr(b, loc, W_mr);

  Value oneI32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(1));
  Value border = b.create<arith::SubIOp>(loc, K, oneI32);
  Value outD = memrefDimAsI32(b, loc, C_mr, 0);
  Value outH = memrefDimAsI32(b, loc, C_mr, 1);
  Value outW = memrefDimAsI32(b, loc, C_mr, 2);
  Value inD = b.create<arith::AddIOp>(loc, outD, border);
  Value inH = b.create<arith::AddIOp>(loc, outH, border);
  Value inW = b.create<arith::AddIOp>(loc, outW, border);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{inD, inH, inW, outD, outH, outW, K, W_ptr, A_ptr, C_ptr});

  Value updatedView =
      memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, C);
  rewireTensorSliceLaunchResult(launch, updatedView, updatedBase);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnStencil3D7ptFlat(LaunchOp launch,
                                                ModuleOp module,
                                                StringRef shimSymbol) {
  if (launch.getNumOperands() != 9 || launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnStencil3D7pt_f32_flat_tensor: expected A, C, two scales, "
        "two strides, three output sizes and one result");
  Value A = launch.getOperand(0), C = launch.getOperand(1);
  auto aTy = dyn_cast<RankedTensorType>(A.getType());
  auto cTy = dyn_cast<RankedTensorType>(C.getType());
  if (!aTy || !cTy || aTy.getRank() != 1 || cTy.getRank() != 1 ||
      !aTy.getElementType().isF32() || !cTy.getElementType().isF32() ||
      launch.getResult(0).getType() != C.getType())
    return launch.emitError(
        "cudnnStencil3D7pt_f32_flat_tensor: expected rank-1 f32 tensors");
  if (!launch.getOperand(2).getType().isF32() ||
      !launch.getOperand(3).getType().isF32())
    return launch.emitError("cudnnStencil3D7pt_f32_flat_tensor: scales must be f32");
  for (Value dim : launch.getOperands().drop_front(4))
    if (!dim.getType().isIndex())
      return launch.emitError(
          "cudnnStencil3D7pt_f32_flat_tensor: dimensions must be index");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value aMemref = valueToMemrefPreservingSlice(b, loc, A);
  Value cMemref = valueToMemrefPreservingSlice(b, loc, C);
  Value aPtr = memrefBasePtr(b, loc, aMemref);
  Value cPtr = memrefBasePtr(b, loc, cMemref);
  SmallVector<Value> args = {aPtr, cPtr, launch.getOperand(2),
                             launch.getOperand(3)};
  for (Value dim : launch.getOperands().drop_front(4))
    args.push_back(b.create<arith::IndexCastOp>(loc, b.getI32Type(), dim));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {ptrTy, ptrTy, b.getF32Type(), b.getF32Type(),
                                b.getI32Type(), b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  rewireLaunchResult(launch, C);
  launch.erase();
  return success();
}

// Multi-channel, single-batch valid Conv3D.  The matcher passes the logical
// rank-8 window so the operation remains self-describing; here we unwrap that
// view to its NCDHW storage tensor and call the ordinary cuDNN Conv3D ABI.
static LogicalResult lowerCudnnConv3DChannelsF32(LaunchOp launch,
                                                 ModuleOp module,
                                                 bool hasBias) {
  unsigned expectedOperands = hasBias ? 4 : 3;
  if (launch.getNumOperands() != expectedOperands ||
      launch.getNumResults() != 1)
    return launch.emitError("cudnnConvolution3D_f32: expected window, filter")
           << (hasBias ? ", bias" : "") << ", output and one result";

  Value window = launch.getOperand(0);
  Value filter = launch.getOperand(1);
  Value bias = hasBias ? launch.getOperand(2) : Value();
  Value output = launch.getOperand(expectedOperands - 1);
  Value input = resolveSubmapBase(window);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto filterTy = dyn_cast<RankedTensorType>(filter.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  auto windowTy = dyn_cast<RankedTensorType>(window.getType());
  if (!inputTy || !filterTy || !outputTy || !windowTy ||
      (inputTy.getRank() != 4 && inputTy.getRank() != 5) ||
      filterTy.getRank() != 5 || outputTy.getRank() != 4 ||
      windowTy.getRank() != 8 || !inputTy.getElementType().isF32() ||
      !filterTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32())
    return launch.emitError(
        "cudnnConvolution3D_f32: expected rank-4/5 input storage, rank-8 "
        "window, rank-5 filter, and rank-4 f32 output");
  if (hasBias) {
    auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasTy || biasTy.getRank() != 1 ||
        !biasTy.getElementType().isF32())
      return launch.emitError(
          "cudnnConvolution3D_f32_bias: bias must be rank-1 f32");
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value filterMr = valueToMemrefPreservingSlice(b, loc, filter);
  Value outputMr = valueToMemrefPreservingSlice(b, loc, output);
  int64_t channelAxis = inputTy.getRank() == 5 ? 1 : 0;
  Value IC = memrefDimAsI32(b, loc, inputMr, channelAxis);
  Value inD = memrefDimAsI32(b, loc, inputMr, channelAxis + 1);
  Value inH = memrefDimAsI32(b, loc, inputMr, channelAxis + 2);
  Value inW = memrefDimAsI32(b, loc, inputMr, channelAxis + 3);
  Value OC = memrefDimAsI32(b, loc, filterMr, 0);
  Value kD = memrefDimAsI32(b, loc, filterMr, 2);
  Value kH = memrefDimAsI32(b, loc, filterMr, 3);
  Value kW = memrefDimAsI32(b, loc, filterMr, 4);
  Value inputPtr = memrefBasePtr(b, loc, inputMr);
  Value filterPtr = memrefBasePtr(b, loc, filterMr);
  Value outputPtr = memrefBasePtr(b, loc, outputMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  Value biasPtr = b.create<LLVM::ZeroOp>(loc, ptrTy);
  if (hasBias) {
    Value biasMr = valueToMemrefPreservingSlice(b, loc, bias);
    biasPtr = memrefBasePtr(b, loc, biasMr);
  }

  SmallVector<Type> argTypes(8, b.getI32Type());
  argTypes.append(4, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv3d_channels_f32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{IC, inD, inH, inW, OC, kD, kH, kW,
                 inputPtr, filterPtr, biasPtr, outputPtr});

  Value updatedView =
      memrefToTensor(b, loc, outputMr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, output);
  rewireTensorSliceLaunchResult(launch, updatedView, updatedBase);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConv1DBiasF32(LaunchOp launch,
                                             ModuleOp module) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvolution1D_f32_bias expects window, filter, bias, output");
  Value input = resolveSubmapBase(launch.getOperand(0));
  Value filter = resolveSubmapBase(launch.getOperand(1));
  Value bias = launch.getOperand(2);
  Value output = launch.getOperand(3);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto filterTy = dyn_cast<RankedTensorType>(filter.getType());
  auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  if (!inputTy || !filterTy || !biasTy || !outputTy ||
      inputTy.getRank() != 3 || filterTy.getRank() != 3 ||
      biasTy.getRank() != 1 || outputTy.getRank() != 3 ||
      !inputTy.getElementType().isF32() ||
      filterTy.getElementType() != inputTy.getElementType() ||
      biasTy.getElementType() != inputTy.getElementType() ||
      outputTy.getElementType() != inputTy.getElementType())
    return launch.emitError(
        "cudnnConvolution1D_f32_bias requires rank-3 f32 NCL tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value filterMr = valueToMemrefPreservingSlice(b, loc, filter);
  Value biasMr = valueToMemrefPreservingSlice(b, loc, bias);
  Value outputMr = valueToOutputMemrefPreservingSlice(b, loc, output);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, inputMr, 0),
      memrefDimAsI32(b, loc, inputMr, 1),
      memrefDimAsI32(b, loc, filterMr, 0),
      memrefDimAsI32(b, loc, inputMr, 2),
      memrefDimAsI32(b, loc, filterMr, 2),
      memrefDataPtr(b, loc, inputMr), memrefDataPtr(b, loc, filterMr),
      memrefDataPtr(b, loc, biasMr), memrefDataPtr(b, loc, outputMr)};
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(5, b.getI32Type());
  argTypes.append(4, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv1d_bias_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  Value updated = memrefToTensor(b, loc, outputMr, output.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConv2DDilatedF32(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvolution2D_f32_dilated expects window, filter, output");
  Value input = resolveSubmapBase(launch.getOperand(0));
  Value filter = resolveSubmapBase(launch.getOperand(1));
  Value output = launch.getOperand(2);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto filterTy = dyn_cast<RankedTensorType>(filter.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  auto dh = launch->getAttrOfType<IntegerAttr>("dilation_h");
  auto dw = launch->getAttrOfType<IntegerAttr>("dilation_w");
  if (!inputTy || !filterTy || !outputTy || !dh || !dw ||
      inputTy.getRank() != 3 || filterTy.getRank() != 4 ||
      outputTy.getRank() != 3 || !inputTy.getElementType().isF32() ||
      filterTy.getElementType() != inputTy.getElementType() ||
      outputTy.getElementType() != inputTy.getElementType())
    return launch.emitError(
        "dilated convolution requires rank-3 CHW, rank-4 OIHW f32 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value filterMr = valueToMemrefPreservingSlice(b, loc, filter);
  Value outputMr = valueToOutputMemrefPreservingSlice(b, loc, output);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, inputMr, 0),
      memrefDimAsI32(b, loc, filterMr, 0),
      memrefDimAsI32(b, loc, inputMr, 1),
      memrefDimAsI32(b, loc, inputMr, 2),
      memrefDimAsI32(b, loc, filterMr, 2),
      memrefDimAsI32(b, loc, filterMr, 3),
      b.create<arith::ConstantIntOp>(loc, dh.getInt(), 32),
      b.create<arith::ConstantIntOp>(loc, dw.getInt(), 32),
      memrefDataPtr(b, loc, inputMr), memrefDataPtr(b, loc, filterMr),
      memrefDataPtr(b, loc, outputMr)};
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(8, b.getI32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv2d_dilated_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  Value updated = memrefToTensor(b, loc, outputMr, output.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

static LogicalResult lowerCublasGemmExI8I32(LaunchOp launch,
                                            ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError("i8 GemmEx expects A, B, C and one result");
  auto aTy = dyn_cast<RankedTensorType>(launch.getOperand(0).getType());
  auto bTy = dyn_cast<RankedTensorType>(launch.getOperand(1).getType());
  auto cTy = dyn_cast<RankedTensorType>(launch.getOperand(2).getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      cTy.getRank() != 2 || !aTy.getElementType().isInteger(8) ||
      !bTy.getElementType().isInteger(8) ||
      !cTy.getElementType().isInteger(32))
    return launch.emitError("i8 GemmEx requires rank-2 i8/i8/i32 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value aMr = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value bMr = valueToMemrefPreservingSlice(b, loc, launch.getOperand(1));
  Value cMr = valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(2));
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, aMr, 0), memrefDimAsI32(b, loc, bMr, 1),
      memrefDimAsI32(b, loc, aMr, 1), memrefDataPtr(b, loc, aMr),
      memrefDataPtr(b, loc, bMr), memrefDataPtr(b, loc, cMr)};
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(3, b.getI32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_gemmex_i8_i32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  Value updated = memrefToTensor(b, loc, cMr, launch.getOperand(2).getType());
  rewireTensorSliceLaunchResult(
      launch, updated,
      tensorForOutputSliceSource(b, loc, launch.getOperand(2)));
  launch.erase();
  return success();
}

static LogicalResult lowerCublasSnrm2F32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 0)
    return launch.emitError("Snrm2 expects input and scalar output memrefs");
  auto inputTy = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto outputTy = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  if (!inputTy || !outputTy || inputTy.getRank() != 1 ||
      outputTy.getRank() != 1 || !inputTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32())
    return launch.emitError("Snrm2 requires rank-1 f32 memrefs");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value n = memrefDimAsI32(b, loc, launch.getOperand(0), 0);
  Value input = memrefDataPtr(b, loc, launch.getOperand(0));
  Value output = memrefDataPtr(b, loc, launch.getOperand(1));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_snrm2_f32",
      TypeRange{b.getI32Type(), ptrTy, ptrTy}, b);
  b.create<func::CallOp>(loc, shim, ValueRange{n, input, output});
  launch.erase();
  return success();
}

static LogicalResult lowerCublasJointMaxAbsProductF32(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("joint max-abs product expects a, b, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError("joint max-abs product requires rank-1 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value n = memrefDimAsI32(b, loc, launch.getOperand(0), 0);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{n};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_joint_maxabs_product_f32",
      TypeRange{b.getI32Type(), ptrTy, ptrTy, ptrTy}, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnFeatureMaskScaleF32(LaunchOp launch,
                                                   ModuleOp module) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() != 1)
    return launch.emitError("feature mask scale expects input, mask, scale, output");
  auto inputTy = dyn_cast<RankedTensorType>(launch.getOperand(0).getType());
  auto maskTy = dyn_cast<RankedTensorType>(launch.getOperand(1).getType());
  auto outputTy = dyn_cast<RankedTensorType>(launch.getOperand(3).getType());
  if (!inputTy || !maskTy || !outputTy || inputTy.getRank() != 4 ||
      maskTy.getRank() != 2 || outputTy.getRank() != 4 ||
      !inputTy.getElementType().isF32() || !maskTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32() ||
      !launch.getOperand(2).getType().isF32())
    return launch.emitError("feature mask scale requires rank-4/rank-2 f32 tensors");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value mask = valueToMemrefPreservingSlice(b, loc, launch.getOperand(1));
  Value output = valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(3));
  SmallVector<Value> args;
  for (unsigned dim = 0; dim < 4; ++dim)
    args.push_back(memrefDimAsI32(b, loc, input, dim));
  args.push_back(launch.getOperand(2));
  args.push_back(memrefDataPtr(b, loc, input));
  args.push_back(memrefDataPtr(b, loc, mask));
  args.push_back(memrefDataPtr(b, loc, output));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(4, b.getI32Type());
  argTypes.push_back(b.getF32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_feature_mask_scale_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  Value updated = memrefToTensor(b, loc, output, launch.getOperand(3).getType());
  rewireTensorSliceLaunchResult(
      launch, updated,
      tensorForOutputSliceSource(b, loc, launch.getOperand(3)));
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConvTranspose2DF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("transposed convolution expects input, filter, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 4 || !type.getElementType().isF32())
      return launch.emitError("transposed convolution requires rank-4 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = launch.getOperand(0), filter = launch.getOperand(1);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, input, 0), memrefDimAsI32(b, loc, input, 1),
      memrefDimAsI32(b, loc, filter, 1), memrefDimAsI32(b, loc, input, 2),
      memrefDimAsI32(b, loc, input, 3), memrefDimAsI32(b, loc, filter, 2),
      memrefDimAsI32(b, loc, filter, 3)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(7, b.getI32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv_transpose2d_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConvTranspose3DF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("3D transposed convolution expects input, filter, output");
  int ranks[] = {4, 5, 4};
  for (auto [value, rank] : llvm::zip(launch.getOperands(), ranks)) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != rank || !type.getElementType().isF32())
      return launch.emitError("3D transposed convolution requires rank-4/5/4 f32 memrefs");
  }
  OpBuilder b(launch); Location loc = launch.getLoc();
  Value input = launch.getOperand(0), filter = launch.getOperand(1);
  SmallVector<Value> args{
      memrefDimAsI32(b, loc, input, 0),
      memrefDimAsI32(b, loc, filter, 1),
      memrefDimAsI32(b, loc, input, 1),
      memrefDimAsI32(b, loc, input, 2),
      memrefDimAsI32(b, loc, input, 3),
      memrefDimAsI32(b, loc, filter, 2),
      memrefDimAsI32(b, loc, filter, 3),
      memrefDimAsI32(b, loc, filter, 4)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types(8, b.getI32Type()); types.append(3, ptr);
  auto shim = ensureShimDecl(module, "polygeist_cudnn_conv_transpose3d_f32",
                             types, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConvBackwardFilter3DF32(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("3D backward-filter expects input, gradient, filter");
  int ranks[] = {4, 4, 5};
  for (auto [value, rank] : llvm::zip(launch.getOperands(), ranks)) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != rank || !type.getElementType().isF32())
      return launch.emitError("3D backward-filter requires rank-4/4/5 f32 memrefs");
  }
  OpBuilder b(launch); Location loc = launch.getLoc();
  Value input = launch.getOperand(0), grad = launch.getOperand(1);
  Value filter = launch.getOperand(2);
  SmallVector<Value> args{
      memrefDimAsI32(b, loc, input, 0),
      memrefDimAsI32(b, loc, grad, 0),
      memrefDimAsI32(b, loc, input, 1),
      memrefDimAsI32(b, loc, input, 2),
      memrefDimAsI32(b, loc, input, 3),
      memrefDimAsI32(b, loc, grad, 1),
      memrefDimAsI32(b, loc, grad, 2),
      memrefDimAsI32(b, loc, grad, 3),
      memrefDimAsI32(b, loc, filter, 2),
      memrefDimAsI32(b, loc, filter, 3),
      memrefDimAsI32(b, loc, filter, 4)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types(11, b.getI32Type()); types.append(3, ptr);
  auto shim = ensureShimDecl(
      module, "polygeist_cudnn_conv_backward_filter3d_f32", types, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnDepthwiseConv2DF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() != 0)
    return launch.emitError(
        "depthwise convolution expects input, filter, bias, output");
  int expectedRanks[] = {4, 3, 1, 4};
  for (auto [value, rank] : llvm::zip(launch.getOperands(), expectedRanks)) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != rank || !type.getElementType().isF32())
      return launch.emitError(
          "depthwise convolution requires rank-4/3/1/4 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = launch.getOperand(0), filter = launch.getOperand(1);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, input, 0), memrefDimAsI32(b, loc, input, 1),
      memrefDimAsI32(b, loc, input, 2), memrefDimAsI32(b, loc, input, 3),
      memrefDimAsI32(b, loc, filter, 1),
      memrefDimAsI32(b, loc, filter, 2)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(6, b.getI32Type());
  argTypes.append(4, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_depthwise_conv2d_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnKroneckerProduct2DF32(LaunchOp launch,
                                                     ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("Kronecker product expects x, y, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 2 || !type.getElementType().isF32())
      return launch.emitError("Kronecker product requires rank-2 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value x = launch.getOperand(0), y = launch.getOperand(1);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, x, 0), memrefDimAsI32(b, loc, x, 1),
      memrefDimAsI32(b, loc, y, 0), memrefDimAsI32(b, loc, y, 1)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(4, b.getI32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cutensor_kronecker_product2d_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnBinaryCrossEntropyMeanF32(LaunchOp launch,
                                                         ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("BCE mean expects input, target, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError("BCE mean requires rank-1 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, launch.getOperand(0), 0)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_binary_cross_entropy_mean_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConvTBCF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("TBC convolution expects input, filter, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 3 || !type.getElementType().isF32())
      return launch.emitError("TBC convolution requires rank-3 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = launch.getOperand(0), filter = launch.getOperand(1);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, input, 0),
      memrefDimAsI32(b, loc, input, 1),
      memrefDimAsI32(b, loc, input, 2),
      memrefDimAsI32(b, loc, filter, 2),
      memrefDimAsI32(b, loc, filter, 0)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(5, b.getI32Type());
  argTypes.append(3, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv_tbc_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnConvTBCBackwardF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("TBC backward expects gradient, filter, output");
  for (Value value : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 3 || !type.getElementType().isF32())
      return launch.emitError("TBC backward requires rank-3 f32 memrefs");
  }
  OpBuilder b(launch); Location loc = launch.getLoc();
  Value grad = launch.getOperand(0), filter = launch.getOperand(1);
  SmallVector<Value> args{
      memrefDimAsI32(b, loc, grad, 0),
      memrefDimAsI32(b, loc, grad, 1),
      memrefDimAsI32(b, loc, filter, 1),
      memrefDimAsI32(b, loc, grad, 2),
      memrefDimAsI32(b, loc, filter, 0)};
  for (Value value : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, value));
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types(5, b.getI32Type()); types.append(3, ptr);
  auto shim = ensureShimDecl(module, "polygeist_cudnn_conv_tbc_backward_f32",
                             types, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnTransformBiasRescaleQKVF32(
    LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 6 || launch.getNumResults() != 0)
    return launch.emitError("QKV transform expects qkv, bias, scale, q, k, v");
  int expectedRanks[] = {5, 3, -1, 4, 4, 4};
  for (unsigned i = 0; i < launch.getNumOperands(); ++i) {
    if (i == 2) {
      if (!launch.getOperand(i).getType().isF32())
        return launch.emitError("QKV scale must be f32");
      continue;
    }
    auto type = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!type || type.getRank() != expectedRanks[i] ||
        !type.getElementType().isF32())
      return launch.emitError("QKV transform has invalid memref operand");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value qkv = launch.getOperand(0);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, qkv, 0), memrefDimAsI32(b, loc, qkv, 1),
      memrefDimAsI32(b, loc, qkv, 3), memrefDimAsI32(b, loc, qkv, 4),
      launch.getOperand(2), memrefDataPtr(b, loc, launch.getOperand(0)),
      memrefDataPtr(b, loc, launch.getOperand(1))};
  for (unsigned i = 3; i < 6; ++i)
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(4, b.getI32Type());
  argTypes.push_back(b.getF32Type());
  argTypes.append(5, ptrTy);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_transform_bias_rescale_qkv_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnAddrElementwiseF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 6 || launch.getNumResults() != 0)
    return launch.emitError("addr expects self, x, y, beta, alpha, output");
  for (unsigned i : {0u, 1u, 2u, 5u}) {
    auto type = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError("addr requires rank-1 f32 memrefs");
  }
  if (!launch.getOperand(3).getType().isF32() ||
      !launch.getOperand(4).getType().isF32())
    return launch.emitError("addr alpha/beta must be f32");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, launch.getOperand(0), 0),
      launch.getOperand(3), launch.getOperand(4),
      memrefDataPtr(b, loc, launch.getOperand(0)),
      memrefDataPtr(b, loc, launch.getOperand(1)),
      memrefDataPtr(b, loc, launch.getOperand(2)),
      memrefDataPtr(b, loc, launch.getOperand(5))};
  SmallVector<Type> argTypes = {b.getI32Type(), b.getF32Type(), b.getF32Type(),
                                ptrTy, ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_addr_elementwise_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnLogSigmoidF32(LaunchOp launch,
                                             ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("log sigmoid expects input, output, and buffer");
  for (Value operand : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError("log sigmoid requires rank-1 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_log_sigmoid_f32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, launch.getOperand(0), 0),
                 memrefDataPtr(b, loc, launch.getOperand(0)),
                 memrefDataPtr(b, loc, launch.getOperand(1)),
                 memrefDataPtr(b, loc, launch.getOperand(2))});
  launch.erase();
  return success();
}

static LogicalResult lowerCufftC2C1DTensor(LaunchOp launch, ModuleOp module,
                                           StringRef shimSymbol) {
  if (launch.getNumOperands() != 3)
    return launch.emitError(
        "cufft 1D tensor: expected operands (input, output, inverse)");
  if (launch.getNumResults() != 1)
    return launch.emitError("cufft 1D tensor: expected 1 tensor result");

  Value A = launch.getOperand(0);
  Value C = launch.getOperand(1);
  Value inverse = launch.getOperand(2);
  auto aTy = dyn_cast<RankedTensorType>(A.getType());
  auto cTy = dyn_cast<RankedTensorType>(C.getType());
  auto rTy = dyn_cast<RankedTensorType>(launch.getResult(0).getType());
  if (!aTy || !cTy || !rTy || aTy.getRank() != 2 || cTy.getRank() != 2 ||
      rTy.getRank() != 2)
    return launch.emitError(
        "cufft 1D tensor: input/output/result must be rank-2 tensors");
  if (aTy.getDimSize(1) != 2 || cTy.getDimSize(1) != 2 ||
      rTy.getDimSize(1) != 2)
    return launch.emitError(
        "cufft 1D tensor: trailing dimension must be static size 2");
  Type elemTy = aTy.getElementType();
  if (cTy.getElementType() != elemTy || rTy.getElementType() != elemTy)
    return launch.emitError(
        "cufft 1D tensor: input/output/result element types must match");
  if (!(elemTy.isF64() || elemTy.isF32()))
    return launch.emitError("cufft 1D tensor: only f64/f32 supported");
  if (!inverse.getType().isInteger(32))
    return launch.emitError("cufft 1D tensor: inverse flag must be i32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = valueToMemrefPreservingSlice(b, loc, A);
  Value C_mr = valueToMemrefPreservingSlice(b, loc, C);
  Value N = memrefDimAsI32(b, loc, A_mr, 0);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, inverse, A_ptr, C_ptr});

  Value updated =
      memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, C);
  rewireTensorSliceLaunchResult(launch, updated, updatedBase);
  launch.erase();
  return success();
}

// Tensor-product contraction
//   out[a,b,c] = sum(i,j,k) psi[a,i] psi[b,j] psi[c,k] u[i,j,k]
// reaches the matcher as five rank-6 submap views over three flat buffers.
// The first three views share the psi base.  Unwrap all views and let the
// cuTensorNet shim express the full Einstein contraction directly.
static LogicalResult lowerCutensornetTensorProduct3D(LaunchOp launch,
                                                     ModuleOp module,
                                                     bool useF64) {
  if (launch.getNumOperands() != 5 || launch.getNumResults() != 1)
    return launch.emitError(
        "cuTensorNet tensor product: expected 5 operands and 1 result");

  for (Value operand : launch.getOperands()) {
    auto ty = dyn_cast<RankedTensorType>(operand.getType());
    if (!ty || ty.getRank() != 6 ||
        (useF64 ? !ty.getElementType().isF64()
                : !ty.getElementType().isF32()))
      return launch.emitError(
          "cuTensorNet tensor product: operands have wrong rank or type");
    auto submap = operand.getDefiningOp<polygeist::SubmapOp>();
    if (!submap || submap.getSizes().size() != 6)
      return launch.emitError(
          "cuTensorNet tensor product: operands must be rank-6 submaps");
  }

  Value psi0 = resolveSubmapBase(launch.getOperand(0));
  Value psi1 = resolveSubmapBase(launch.getOperand(1));
  Value psi2 = resolveSubmapBase(launch.getOperand(2));
  Value u = resolveSubmapBase(launch.getOperand(3));
  Value out = resolveSubmapBase(launch.getOperand(4));
  if (psi0 != psi1 || psi0 != psi2)
    return launch.emitError(
        "cuTensorNet tensor product: first three views must share psi base");

  auto psiTy = dyn_cast<RankedTensorType>(psi0.getType());
  auto uTy = dyn_cast<RankedTensorType>(u.getType());
  auto outTy = dyn_cast<RankedTensorType>(out.getType());
  if (!psiTy || !uTy || !outTy ||
      (useF64 ? (!psiTy.getElementType().isF64() ||
                 !uTy.getElementType().isF64() ||
                 !outTy.getElementType().isF64())
              : (!psiTy.getElementType().isF32() ||
                 !uTy.getElementType().isF32() ||
                 !outTy.getElementType().isF32())))
    return launch.emitError(
        "cuTensorNet tensor product: submap bases have wrong type");

  auto firstView = launch.getOperand(0).getDefiningOp<polygeist::SubmapOp>();
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value KQ = valueAsI32(b, loc, firstView.getSizes()[0]);
  Value KP = valueAsI32(b, loc, firstView.getSizes()[3]);
  Value psiMr = tensorToMemref(b, loc, psi0);
  Value uMr = tensorToMemref(b, loc, u);
  Value outMr = tensorToMemref(b, loc, out);
  Value psiPtr = memrefBasePtr(b, loc, psiMr);
  Value uPtr = memrefBasePtr(b, loc, uMr);
  Value outPtr = memrefBasePtr(b, loc, outMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy,
                                ptrTy, ptrTy};
  StringRef shimName = useF64
                           ? "polygeist_cutensornet_tensor_product_3d_f64"
                           : "polygeist_cutensornet_tensor_product_3d_f32";
  func::FuncOp shim = ensureShimDecl(module, shimName, argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{KQ, KP, psiPtr, uPtr, outPtr});

  Value updatedOut = memrefToTensor(b, loc, outMr, out.getType());
  rewireLaunchResult(launch, updatedOut);
  launch.erase();
  return success();
}

// Return the constant coefficient of affine dimension `dim` in a linear
// affine expression. MFEM's polygeist.submap views are flattened row-major
// maps such as d4 + d0 * 64 + d1 * 16 + d2 * 4. Their coefficients are the
// physical element strides needed by cuTensorNet. Reject non-linear
// floor/mod expressions rather than guessing a layout.
static std::optional<int64_t>
constantAffineDimCoefficient(AffineExpr expr, unsigned dim) {
  if (auto d = expr.dyn_cast<AffineDimExpr>())
    return d.getPosition() == dim ? 1 : 0;
  if (expr.isa<AffineConstantExpr>() || expr.isa<AffineSymbolExpr>())
    return 0;
  auto binary = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!binary)
    return std::nullopt;
  if (binary.getKind() == AffineExprKind::Add) {
    auto lhs = constantAffineDimCoefficient(binary.getLHS(), dim);
    auto rhs = constantAffineDimCoefficient(binary.getRHS(), dim);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  if (binary.getKind() == AffineExprKind::Mul) {
    if (auto c = binary.getLHS().dyn_cast<AffineConstantExpr>()) {
      auto rhs = constantAffineDimCoefficient(binary.getRHS(), dim);
      return rhs ? std::optional<int64_t>(c.getValue() * *rhs)
                 : std::nullopt;
    }
    if (auto c = binary.getRHS().dyn_cast<AffineConstantExpr>()) {
      auto lhs = constantAffineDimCoefficient(binary.getLHS(), dim);
      return lhs ? std::optional<int64_t>(c.getValue() * *lhs)
                 : std::nullopt;
    }
  }
  return std::nullopt;
}

// Evaluate a linear affine expression with every dimension set to zero.  The
// result is the constant base offset of a flattened submap.  This matters for
// MFEM component views such as `... + 36`: strides alone describe their
// layout, but the runtime pointer must also start 36 elements into the base.
static std::optional<int64_t> constantAffineOffset(AffineExpr expr) {
  if (expr.isa<AffineDimExpr>())
    return 0;
  if (auto constant = expr.dyn_cast<AffineConstantExpr>())
    return constant.getValue();
  if (expr.isa<AffineSymbolExpr>())
    return std::nullopt;
  auto binary = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!binary)
    return std::nullopt;
  auto lhs = constantAffineOffset(binary.getLHS());
  auto rhs = constantAffineOffset(binary.getRHS());
  if (!lhs || !rhs)
    return std::nullopt;
  if (binary.getKind() == AffineExprKind::Add)
    return *lhs + *rhs;
  if (binary.getKind() == AffineExprKind::Mul)
    return *lhs * *rhs;
  return std::nullopt;
}

struct ContractionViewMetadata {
  Value base;
  Value elementOffset;
  bool needsDenseInputCopy = false;
  SmallVector<Value, 8> extents;
  SmallVector<Value, 8> strides;
  SmallVector<int64_t, 8> modes;
};

static constexpr int64_t kContractionMaxModes = 64;

static Value shapedDimAsI64(OpBuilder &b, Location loc, Value value,
                            unsigned dim) {
  auto shaped = cast<ShapedType>(value.getType());
  if (!shaped.isDynamicDim(dim))
    return b.create<arith::ConstantOp>(
        loc, b.getI64Type(), b.getI64IntegerAttr(shaped.getDimSize(dim)));
  Value axis = b.create<arith::ConstantIndexOp>(loc, dim);
  Value extent;
  if (isa<RankedTensorType>(value.getType()))
    extent = b.create<tensor::DimOp>(loc, value, axis);
  else
    extent = b.create<memref::DimOp>(loc, value, axis);
  return integerLikeAsI64(b, loc, extent);
}

// Return the physical element strides of an ordinary dense tensor or an
// extract_slice view.  A slice keeps the parent tensor's strides;
// its result shape alone is not enough to recover the layout.  For example,
// a 2x3x3x5 slice of a 2x4x4x5 tensor has strides 80,20,5,1, not the dense
// 45,15,5,1 strides implied by the slice's sizes.
static FailureOr<SmallVector<Value, 8>>
physicalTensorStrides(OpBuilder &b, Location loc, Value tensor) {
  Value stripped = stripTensorCasts(tensor);
  auto type = dyn_cast<RankedTensorType>(stripped.getType());
  if (!type)
    return failure();

  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!sourceType ||
        slice.getMixedStrides().size() != (unsigned)sourceType.getRank())
      return failure();
    auto sourceStrides =
        physicalTensorStrides(b, loc, slice.getSource());
    if (failed(sourceStrides))
      return failure();
    // getDroppedDims maps a rank-reduced result back to the source dimensions.
    // A dropped singleton dimension still changes the slice's base offset, but
    // it is not a logical cuTENSOR mode and therefore has no result stride.
    llvm::SmallBitVector droppedDims = slice.getDroppedDims();
    SmallVector<Value, 8> result;
    for (unsigned dim = 0; dim < (unsigned)sourceType.getRank(); ++dim) {
      if (droppedDims.test(dim))
        continue;
      Value step = opFoldResultAsI64(b, loc, slice.getMixedStrides()[dim]);
      result.push_back(
          b.create<arith::MulIOp>(loc, (*sourceStrides)[dim], step));
    }
    if (result.size() != (unsigned)type.getRank())
      return failure();
    return result;
  }

  SmallVector<Value, 8> result(type.getRank());
  Value stride = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(1));
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    result[dim] = stride;
    stride = b.create<arith::MulIOp>(
        loc, stride, shapedDimAsI64(b, loc, stripped, dim));
  }
  return result;
}

static FailureOr<ContractionViewMetadata>
buildContractionViewMetadata(OpBuilder &b, Location loc, Value operand,
                             AffineMap accessMap,
                             bool preserveDirectSubmapBase = false) {
  Value stripped = stripTensorCasts(operand);
  auto operandType = dyn_cast<RankedTensorType>(stripped.getType());
  if (!operandType ||
      !(operandType.getElementType().isF64() ||
        operandType.getElementType().isF32()) ||
      operandType.getRank() > kContractionMaxModes ||
      accessMap.getNumResults() != (unsigned)operandType.getRank())
    return failure();

  SmallVector<int64_t, 8> logicalModes;
  for (AffineExpr result : accessMap.getResults()) {
    auto dim = result.dyn_cast<AffineDimExpr>();
    if (!dim || dim.getPosition() >= kContractionMaxModes)
      return failure();
    logicalModes.push_back(dim.getPosition());
  }

  SmallVector<Value, 8> logicalExtents;
  SmallVector<Value, 8> logicalStrides;
  Value base = resolveSubmapBase(stripped);
  Value elementOffset = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(0));
  if (auto submap = stripped.getDefiningOp<polygeist::SubmapOp>()) {
    // A composed network can follow ordinary tensor computation. Preserve the
    // direct SSA base in that case instead of walking through a preceding
    // submapInverse to its first/original base. Keep the established resolved
    // in-place behavior for the pairwise contraction ABI: its chained MFEM
    // launches rely on that representation for GPU host/device staging.
    if (preserveDirectSubmapBase)
      base = submap.getBase();
    auto baseType = dyn_cast<RankedTensorType>(base.getType());
    if (!baseType ||
        submap.getSizes().size() != (unsigned)operandType.getRank())
      return failure();

    if (baseType.getRank() == 1 &&
        submap.getMap().getNumResults() == 1) {
      // General flattened view: derive one physical stride per logical dim
      // from the affine address expression.  Zero strides are broadcasts.
      AffineExpr flatExpr = submap.getMap().getResult(0);
      auto constantOffset = constantAffineOffset(flatExpr);
      if (!constantOffset || *constantOffset < 0)
        return failure();
      elementOffset = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(*constantOffset));
      for (unsigned dim = 0; dim < (unsigned)operandType.getRank(); ++dim) {
        logicalExtents.push_back(
            integerLikeAsI64(b, loc, submap.getSizes()[dim]));
        auto coefficient = constantAffineDimCoefficient(flatExpr, dim);
        if (!coefficient || *coefficient < 0)
          return failure();
        logicalStrides.push_back(b.create<arith::ConstantOp>(
            loc, b.getI64Type(), b.getI64IntegerAttr(*coefficient)));
      }
    } else {
      // A submap over an ordinary ranked tensor can still express
      // permutation and broadcasting.  Compose its affine coefficients with
      // the direct base's dense row-major strides.  This covers both identity
      // accumulator views and maps such as
      //   (d0,d1,d2,d3,d4) -> (d0,d1,d4,d3)
      // without pretending that an arbitrary nonlinear map is supported.
      Value directBase = submap.getBase();
      auto directBaseType = dyn_cast<RankedTensorType>(directBase.getType());
      if (!directBaseType ||
          submap.getMap().getNumResults() !=
              static_cast<unsigned>(directBaseType.getRank()) ||
          submap.getMap().getNumSymbols() != 0)
        return failure();
      base = directBase;
      for (Value size : submap.getSizes())
        logicalExtents.push_back(integerLikeAsI64(b, loc, size));

      SmallVector<Value, 8> baseStrides(directBaseType.getRank());
      Value stride = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(1));
      for (int64_t dim = directBaseType.getRank() - 1; dim >= 0; --dim) {
        baseStrides[dim] = stride;
        stride = b.create<arith::MulIOp>(
            loc, stride, shapedDimAsI64(b, loc, directBase, dim));
      }
      for (unsigned baseDim = 0;
           baseDim < static_cast<unsigned>(directBaseType.getRank());
           ++baseDim) {
        auto constantOffset =
            constantAffineOffset(submap.getMap().getResult(baseDim));
        if (!constantOffset || *constantOffset < 0)
          return failure();
        if (*constantOffset == 0)
          continue;
        Value coefficient = b.create<arith::ConstantOp>(
            loc, b.getI64Type(), b.getI64IntegerAttr(*constantOffset));
        Value contribution =
            b.create<arith::MulIOp>(loc, baseStrides[baseDim], coefficient);
        elementOffset =
            b.create<arith::AddIOp>(loc, elementOffset, contribution);
      }
      for (unsigned logicalDim = 0;
           logicalDim < static_cast<unsigned>(operandType.getRank());
           ++logicalDim) {
        Value logicalStride = b.create<arith::ConstantOp>(
            loc, b.getI64Type(), b.getI64IntegerAttr(0));
        for (unsigned baseDim = 0;
             baseDim < static_cast<unsigned>(directBaseType.getRank());
             ++baseDim) {
          auto coefficient = constantAffineDimCoefficient(
              submap.getMap().getResult(baseDim), logicalDim);
          if (!coefficient || *coefficient < 0)
            return failure();
          if (*coefficient == 0)
            continue;
          Value coefficientValue = b.create<arith::ConstantOp>(
              loc, b.getI64Type(), b.getI64IntegerAttr(*coefficient));
          Value contribution = b.create<arith::MulIOp>(
              loc, baseStrides[baseDim], coefficientValue);
          logicalStride =
              b.create<arith::AddIOp>(loc, logicalStride, contribution);
        }
        logicalStrides.push_back(logicalStride);
      }
    }
  } else if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    base = stripped;
    // The slice size list is expressed in source-rank coordinates.  Query the
    // result instead so rank-reduced singleton dimensions do not appear as
    // logical cuTENSOR modes.
    for (unsigned dim = 0; dim < (unsigned)operandType.getRank(); ++dim)
      logicalExtents.push_back(shapedDimAsI64(b, loc, stripped, dim));
    auto physicalStrides = physicalTensorStrides(b, loc, stripped);
    if (failed(physicalStrides))
      return failure();
    logicalStrides.append(physicalStrides->begin(), physicalStrides->end());
  } else {
    base = stripped;
    for (unsigned dim = 0; dim < (unsigned)operandType.getRank(); ++dim)
      logicalExtents.push_back(shapedDimAsI64(b, loc, stripped, dim));
    Value stride = b.create<arith::ConstantOp>(
        loc, b.getI64Type(), b.getI64IntegerAttr(1));
    logicalStrides.resize(operandType.getRank());
    for (int64_t dim = operandType.getRank() - 1; dim >= 0; --dim) {
      logicalStrides[dim] = stride;
      stride = b.create<arith::MulIOp>(loc, stride, logicalExtents[dim]);
    }
  }

  ContractionViewMetadata metadata;
  metadata.base = base;
  metadata.elementOffset = elementOffset;
  // An ordinary tensor result can have been bufferized in-place into a
  // strided DPS init even though tensor types carry no layout.  The dense
  // strides derived below are therefore only safe after materializing an
  // explicit dense input copy.  Submaps and extract_slice views have layout
  // provenance and do not need this fallback.
  if (stripped.getDefiningOp<polygeist::SubmapOp>()) {
    // A submap of a computed tensor (for example, the stress tensor produced
    // by an MFEM quadrature stage) must stay live across the opaque call and
    // later component contractions.  Materialize its flat/ranked base unless
    // it is a direct function memref view.
    metadata.needsDenseInputCopy = sourceToTensorOp(base) == nullptr;
  } else {
    metadata.needsDenseInputCopy =
        !stripped.getDefiningOp<tensor::ExtractSliceOp>() &&
        stripped.getDefiningOp() != nullptr;
  }
  for (unsigned dim = 0; dim < logicalModes.size(); ++dim) {
    // A zero physical stride is a broadcasted logical mode. cuTensorNet and
    // cuTENSOR represent broadcasting by omitting that mode from the tensor,
    // rather than by passing an illegal zero stride.
    llvm::APInt staticStride;
    if (matchPattern(logicalStrides[dim], m_ConstantInt(&staticStride)) &&
        staticStride.isZero())
      continue;
    metadata.extents.push_back(logicalExtents[dim]);
    metadata.strides.push_back(logicalStrides[dim]);
    metadata.modes.push_back(logicalModes[dim]);
  }
  return metadata;
}

// The generic tensor-network op is intentionally bufferizable and is normally
// lowered after one-shot bufferization. Preserve the older tensor provenance
// path above, but also accept arbitrary ranked memref views by reading their
// strided metadata. memrefDataPtr accounts for the view offset, so the
// metadata below contains strides relative to logical element zero.
static FailureOr<ContractionViewMetadata>
buildNetworkViewMetadata(OpBuilder &b, Location loc, Value operand,
                         AffineMap accessMap) {
  if (isa<TensorType>(operand.getType()))
    return buildContractionViewMetadata(b, loc, operand, accessMap,
                                        /*preserveDirectSubmapBase=*/true);

  // One-shot bufferization is allowed to leave an unknown Polygeist tensor
  // view behind a to_memref boundary. Recover that tensor provenance here
  // instead of treating the materialized memref as the semantic operand. The
  // tensor metadata builder understands submap affine maps and returns the
  // original flat ABI base; after the launch is erased the obsolete
  // to_memref(submap(...)) chain canonicalizes away.
  if (auto toMemref = operand.getDefiningOp<bufferization::ToMemrefOp>()) {
    Value tensor = toMemref.getTensor();
    if (isa<RankedTensorType>(tensor.getType()))
      return buildContractionViewMetadata(
          b, loc, tensor, accessMap, /*preserveDirectSubmapBase=*/true);
  }

  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type || !(type.getElementType().isF32() ||
                 type.getElementType().isF64()) ||
      type.getRank() > kContractionMaxModes ||
      accessMap.getNumResults() != (unsigned)type.getRank())
    return failure();

  ContractionViewMetadata result;
  result.base = operand;
  result.elementOffset = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(0));
  auto strided = b.create<memref::ExtractStridedMetadataOp>(loc, operand);
  for (auto [dim, expr] : llvm::enumerate(accessMap.getResults())) {
    auto mode = expr.dyn_cast<AffineDimExpr>();
    if (!mode || mode.getPosition() >= kContractionMaxModes)
      return failure();
    Value stride = integerLikeAsI64(b, loc, strided.getStrides()[dim]);
    llvm::APInt staticStride;
    if (matchPattern(stride, m_ConstantInt(&staticStride)) &&
        staticStride.isZero())
      continue;
    result.extents.push_back(
        integerLikeAsI64(b, loc, strided.getSizes()[dim]));
    result.strides.push_back(stride);
    result.modes.push_back(mode.getPosition());
  }
  return result;
}

// Generic two-input FP64 Einstein contraction for MFEM's mode-wise
// sum-factorization stages. Metadata layout (all i64):
//   [rankA, rankB, rankC,
//    A.extent[64], A.stride[64], A.mode[64],
//    B.extent[64], B.stride[64], B.mode[64],
//    C.extent[64], C.stride[64], C.mode[64]]
// Unused slots are extent=1, stride=0, mode=-1.
static bool isDeclaredDeviceResidentTensor(Value value) {
  value = stripTensorCasts(value);
  if (isa<BlockArgument>(value))
    return true;
  if (auto submap = value.getDefiningOp<polygeist::SubmapOp>())
    return isDeclaredDeviceResidentTensor(submap.getBase());
  if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>())
    return isDeclaredDeviceResidentTensor(slice.getSource());
  if (auto launch = value.getDefiningOp<LaunchOp>())
    return launch->hasAttr("polygeist.device_resident");
  return false;
}

static LogicalResult verifyNoResidualHostDeviceConsumers(LaunchOp launch) {
  func::FuncOp function = launch->getParentOfType<func::FuncOp>();
  if (!function)
    return launch.emitError("device-resident launch must be inside a function");
  Operation *illegal = nullptr;
  function.walk([&](Operation *op) {
    if (illegal)
      return WalkResult::interrupt();
    StringRef name = op->getName().getStringRef();
    if ((name.startswith("linalg.") && name != "linalg.yield") ||
        name.startswith("affine.") ||
        name.startswith("scf.") || name == "memref.load" ||
        name == "memref.store" || name == "memref.copy" ||
        name == "tensor.extract" || name == "tensor.insert" ||
        name == "tensor.insert_slice" ||
        name == "polygeist.submapInverse") {
      illegal = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!illegal)
    return success();
  InFlightDiagnostic diagnostic = launch.emitError(
      "device-resident cuTensorNet ABI is illegal while residual host tensor "
      "computation remains");
  diagnostic.attachNote(illegal->getLoc())
      << "host operation is here: " << illegal->getName().getStringRef();
  return failure();
}

static LogicalResult lowerCutensornetContraction2F64(LaunchOp launch,
                                                     ModuleOp module) {
  bool bufferized = launch->hasAttr("polygeist.bufferized");
  if (launch.getNumOperands() != 3 ||
      (bufferized ? launch.getNumResults() != 0
                  : launch.getNumResults() != 1))
    return launch.emitError(
        "cuTensorNet contraction: expected A/B/C operands and either one "
        "tensor result or a result-free bufferized destination");
  auto mapsAttr = launch->getAttrOfType<ArrayAttr>("contraction_maps");
  if (!mapsAttr || mapsAttr.size() != 3)
    return launch.emitError(
        "cuTensorNet contraction: expected three contraction_maps");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  const bool deviceResident =
      launch->hasAttr("polygeist.device_resident");
  if (deviceResident && failed(verifyNoResidualHostDeviceConsumers(launch)))
    return failure();
  SmallVector<ContractionViewMetadata, 3> metadata;
  for (unsigned i = 0; i < 3; ++i) {
    auto mapAttr = dyn_cast<AffineMapAttr>(mapsAttr[i]);
    if (!mapAttr)
      return launch.emitError(
          "cuTensorNet contraction: contraction_maps must be affine maps");
    auto view = buildNetworkViewMetadata(b, loc, launch.getOperand(i),
                                         mapAttr.getValue());
    if (failed(view))
      return launch.emitError(
          "cuTensorNet contraction: unsupported operand view/map layout");
    metadata.push_back(*view);
    if (deviceResident && !isDeclaredDeviceResidentTensor(view->base))
      return launch.emitError(
          "device-resident cuTensorNet ABI requires every operand buffer to "
          "originate from a device-ABI function argument or another "
          "device-resident launch");
  }
  if (metadata[2].modes.empty())
    return launch.emitError(
        "cuTensorNet contraction: scalar/fully-broadcast outputs are not yet "
        "supported");

  llvm::SmallSet<int64_t, 16> inputModes;
  llvm::SmallSet<int64_t, 16> outputModes;
  for (int64_t mode : metadata[0].modes)
    inputModes.insert(mode);
  for (int64_t mode : metadata[1].modes)
    inputModes.insert(mode);
  for (int64_t mode : metadata[2].modes) {
    if (!inputModes.contains(mode))
      return launch.emitError(
          "cuTensorNet contraction: output mode is absent from both inputs");
    outputModes.insert(mode);
  }
  bool hasReduction = llvm::any_of(
      inputModes, [&](int64_t mode) { return !outputModes.contains(mode); });
  if (!hasReduction)
    return launch.emitError(
        "cuTensorNet contraction: expected at least one reduced mode");

  constexpr int64_t kMaxRank = kContractionMaxModes;
  constexpr int64_t kFieldsPerTensor = 3 * kMaxRank;
  constexpr int64_t kMetadataSize = 3 + 3 * kFieldsPerTensor;
  auto metadataType = MemRefType::get({kMetadataSize}, b.getI64Type());
  Value metadataBuffer = b.create<memref::AllocaOp>(loc, metadataType);
  auto storeMetadata = [&](int64_t index, Value value) {
    Value slot = b.create<arith::ConstantIndexOp>(loc, index);
    b.create<memref::StoreOp>(loc, value, metadataBuffer, slot);
  };
  auto constantI64 = [&](int64_t value) -> Value {
    return b.create<arith::ConstantOp>(
        loc, b.getI64Type(), b.getI64IntegerAttr(value));
  };
  for (unsigned tensor = 0; tensor < 3; ++tensor) {
    storeMetadata(tensor, constantI64(metadata[tensor].modes.size()));
    int64_t baseOffset = 3 + tensor * kFieldsPerTensor;
    for (int64_t dim = 0; dim < kMaxRank; ++dim) {
      bool present = dim < (int64_t)metadata[tensor].modes.size();
      storeMetadata(baseOffset + dim,
                    present ? metadata[tensor].extents[dim] : constantI64(1));
      storeMetadata(baseOffset + kMaxRank + dim,
                    present ? metadata[tensor].strides[dim] : constantI64(0));
      storeMetadata(baseOffset + 2 * kMaxRank + dim,
                    constantI64(present ? metadata[tensor].modes[dim] : -1));
    }
  }

  SmallVector<Value, 3> memrefs;
  SmallVector<Value, 3> pointers;
  for (unsigned tensor = 0; tensor < metadata.size(); ++tensor) {
    const ContractionViewMetadata &view = metadata[tensor];
    Value memref = valueToMemref(b, loc, view.base);
    if (deviceResident && tensor < 2 && view.needsDenseInputCopy)
      return launch.emitError(
          "device-resident cuTensorNet ABI cannot materialize a host-side "
          "dense input snapshot");
    if (tensor < 2 && view.needsDenseInputCopy)
      memref = snapshotOpaqueCallResult(b, loc, memref);
    memrefs.push_back(memref);
    Value pointer = memrefBasePtr(b, loc, memref);
    llvm::APInt staticOffset;
    if (!matchPattern(view.elementOffset, m_ConstantInt(&staticOffset)) ||
        !staticOffset.isZero()) {
      Value address = b.create<LLVM::PtrToIntOp>(
          loc, b.getI64Type(), pointer);
      unsigned bits = cast<MemRefType>(memref.getType())
                          .getElementType()
                          .getIntOrFloatBitWidth();
      Value elementBytes = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
      Value byteOffset =
          b.create<arith::MulIOp>(loc, view.elementOffset, elementBytes);
      address = b.create<arith::AddIOp>(loc, address, byteOffset);
      pointer = b.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(b.getContext()), address);
    }
    pointers.push_back(pointer);
  }
  Value metadataPtr = memrefBasePtr(b, loc, metadataBuffer);
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(4, ptrType);
  StringRef shimName = deviceResident
                           ? "polygeist_cutensornet_contraction2_f64_device"
                           : "polygeist_cutensornet_contraction2_f64";
  func::FuncOp shim = ensureShimDecl(
      module, shimName, argTypes, b);
  auto runtimeCall = b.create<func::CallOp>(
      loc, shim,
      ValueRange{pointers[0], pointers[1], pointers[2], metadataPtr});
  // The device-resident ABI performs no host/device registration or transfer
  // and its operand addresses are stable under the ABI contract, so it may be
  // enclosed by the optional CUDA Graph wrapper. Host-mapped shims are not
  // annotated until their complete host-side setup path is capture-safe.
  if (deviceResident)
    runtimeCall->setAttr("polygeist.cuda_graph_safe", b.getUnitAttr());

  // One-shot bufferization has already made the launch's destination write
  // explicit and replaced every tensor result with that destination buffer.
  // The runtime call above mutates the exact same memref, so manufacturing a
  // tensor snapshot here would both duplicate the output and discard the
  // alias/lifetime proof that bufferization just established.
  if (bufferized) {
    launch.erase();
    return success();
  }

  // The common destination-style form extracts the full (or a sliced) output
  // from a to_tensor of the public ABI memref, then inserts the launch result
  // back into that same tensor.  The pointer above already aliases that ABI
  // destination, so bypass both the compatibility snapshot and the terminal
  // write-back.  Besides avoiding a full tensor copy, this keeps large GEMMs
  // from manufacturing dynamic memref descriptors solely for a dead copy.
  if (Value destinationBase =
          tensorForOutputSliceSource(b, loc, metadata[2].base)) {
    // The opaque runtime mutates the output slice in place. In addition to
    // terminal insert_slice write-backs, a staged contraction may consume the
    // launch result directly as the input of its next stage. Forward such
    // users to the output view whose storage the call just updated.
    rewireTensorSliceLaunchResult(launch, launch.getOperand(2),
                                  destinationBase);
    if (!launch.getResult(0).use_empty())
      return launch.emitError(
          "cuTensorNet contraction: unsupported consumer of direct output");
    launch.erase();
    return success();
  }

  if (deviceResident) {
    // The device runtime writes the DPS output buffer in place. Avoid the
    // correctness-first host snapshot used by the compatibility ABI: copying
    // a CUDA device pointer with memref.copy would be invalid. The legality
    // pre-check permits this path only when no residual host tensor operation
    // can observe the SSA value independently of that buffer side effect.
    Value outputView = launch.getOperand(2);
    Value outputBase = metadata[2].base;
    if (failed(rewireSubmapLaunchResult(launch, outputView, outputBase)))
      return failure();
    launch.erase();
    return success();
  }

  // The runtime receives only an LLVM pointer, so its write is invisible to
  // tensor bufferization.  Preserve this result before a later scratch tensor
  // can reuse the output allocation.  This is correctness-first; a future
  // bufferizable library-call op can model the write directly and remove the
  // snapshot copy.
  Value outputSnapshot = snapshotOpaqueCallResult(b, loc, memrefs[2]);
  Value updatedOutput =
      memrefToTensor(b, loc, outputSnapshot, metadata[2].base.getType());
  Value updatedOutputView = updatedOutput;
  Value originalOutput = stripTensorCasts(launch.getOperand(2));
  if (auto submap = originalOutput.getDefiningOp<polygeist::SubmapOp>()) {
    SmallVector<Value> indicesAndSizes(submap.getOperands().drop_front());
    updatedOutputView = b.create<polygeist::SubmapOp>(
        loc, launch.getOperand(2).getType(), updatedOutput, indicesAndSizes,
        submap.getMap());
  } else if (updatedOutput.getType() != launch.getResult(0).getType()) {
    // Extract-slice and ordinary DPS outputs may be statically shaped below
    // the dynamic ABI cast used by kernel.launch.  Their snapshot already has
    // the right rank and layout; restore only the launch's exposed tensor
    // type before reconnecting its consumers.
    updatedOutputView = b.create<tensor::CastOp>(
        loc, launch.getResult(0).getType(), updatedOutput);
  }
  if (isa<UnrankedTensorType>(launch.getResult(0).getType())) {
    SmallVector<tensor::CastOp> resultCasts;
    for (Operation *user : launch.getResult(0).getUsers())
      if (auto cast = dyn_cast<tensor::CastOp>(user))
        resultCasts.push_back(cast);
    for (tensor::CastOp cast : resultCasts) {
      SmallVector<polygeist::SubmapInverseOp> inverses;
      for (Operation *user : cast.getResult().getUsers())
        if (auto inverse = dyn_cast<polygeist::SubmapInverseOp>(user))
          inverses.push_back(inverse);
      for (polygeist::SubmapInverseOp inverse : inverses) {
        inverse.getResult().replaceAllUsesWith(updatedOutput);
        inverse.erase();
      }
      if (!cast.getResult().use_empty() &&
          cast.getResult().getType() == updatedOutput.getType())
        cast.getResult().replaceAllUsesWith(updatedOutput);
      if (cast.getResult().use_empty())
        cast.erase();
    }
  }
  if (failed(rewireSubmapLaunchResult(launch, updatedOutputView,
                                      updatedOutput)))
    return failure();
  launch.erase();
  return success();
}

// Lower a variable-arity Einstein network. The launch carries one affine map
// per operand in `network_maps`; exactly one operand is the destination and
// all remaining operands are input tensor nodes. Every map has the same domain
// (the global network modes), so no MFEM-specific rank or contraction order is
// encoded in this lowering.
static LogicalResult lowerCutensornetNetwork(LaunchOp launch, ModuleOp module,
                                             bool useF64) {
  if (launch.getNumOperands() < 3)
    return launch.emitError(
        "cuTensorNet network requires at least two inputs and one output");
  auto mapsAttr = launch->getAttrOfType<ArrayAttr>("network_maps");
  if (!mapsAttr || mapsAttr.size() != launch.getNumOperands())
    return launch.emitError(
        "cuTensorNet network requires one network_maps entry per operand");

  unsigned outputOperand = launch.getNumOperands() - 1;
  if (auto destinations = launch->getAttrOfType<DenseI64ArrayAttr>(
          "polygeist.result_destinations")) {
    if (destinations.size() != 1 || destinations[0] < 0 ||
        destinations[0] >= (int64_t)launch.getNumOperands())
      return launch.emitError(
          "cuTensorNet network requires exactly one valid result destination");
    outputOperand = (unsigned)destinations[0];
  } else if (launch.getNumResults() != 1) {
    return launch.emitError(
        "unbufferized cuTensorNet network requires exactly one result");
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  bool deviceResident = launch->hasAttr("polygeist.device_resident");
  if (deviceResident && failed(verifyNoResidualHostDeviceConsumers(launch)))
    return failure();

  SmallVector<unsigned> tensorOrder;
  tensorOrder.reserve(launch.getNumOperands());
  for (unsigned i = 0; i < launch.getNumOperands(); ++i)
    if (i != outputOperand)
      tensorOrder.push_back(i);
  tensorOrder.push_back(outputOperand);

  auto fixedExtents = launch->getAttrOfType<DenseI64ArrayAttr>(
      "polygeist.fixed_operand_extents");
  if (fixedExtents) {
    unsigned expectedCount = 0;
    for (Value operand : launch.getOperands()) {
      auto shaped = getRankedShapedType(operand);
      if (!shaped)
        return launch.emitError(
            "fixed cuTensorNet extents require ranked operands");
      expectedCount += shaped.getRank();
    }
    if (fixedExtents.size() != expectedCount ||
        llvm::any_of(fixedExtents.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return launch.emitError(
          "fixed cuTensorNet extents must provide one positive value per dimension");
  }

  SmallVector<ContractionViewMetadata, 4> metadata;
  metadata.reserve(tensorOrder.size());
  Type expectedElementType = useF64 ? b.getF64Type() : b.getF32Type();
  unsigned globalModeCount = 0;
  for (unsigned operandNumber : tensorOrder) {
    auto shaped = getRankedShapedType(launch.getOperand(operandNumber));
    if (!shaped || shaped.getElementType() != expectedElementType)
      return launch.emitError(
          "cuTensorNet network operands must be ranked and have the symbol's "
          "element type");
    auto mapAttr = dyn_cast<AffineMapAttr>(mapsAttr[operandNumber]);
    if (!mapAttr)
      return launch.emitError(
          "cuTensorNet network maps must be affine map attributes");
    globalModeCount = std::max(globalModeCount,
                               mapAttr.getValue().getNumDims());
    auto view = buildNetworkViewMetadata(
        b, loc, launch.getOperand(operandNumber), mapAttr.getValue());
    if (failed(view))
      return launch.emitError(
          "unsupported cuTensorNet network operand view or access map");
    if (fixedExtents) {
      unsigned extentOffset = 0;
      for (unsigned prior = 0; prior < operandNumber; ++prior)
        extentOffset += getRankedShapedType(
            launch.getOperand(prior)).getRank();
      if (view->extents.size() != (unsigned)shaped.getRank())
        return launch.emitError(
            "fixed cuTensorNet extents cannot describe broadcast operands");
      for (unsigned dim = 0; dim < view->extents.size(); ++dim)
        view->extents[dim] = b.create<arith::ConstantOp>(
            loc, b.getI64Type(),
            b.getI64IntegerAttr(fixedExtents[extentOffset + dim]));
    }
    metadata.push_back(*view);
  }
  if (globalModeCount > kContractionMaxModes)
    return launch.emitError("cuTensorNet network exceeds the 64-mode ABI");
  if (launch.getNumResults() == 1 && !deviceResident &&
      !sourceToTensorOp(metadata.back().base))
    return launch.emitError(
        "host-ABI cuTensorNet network requires a direct ABI-backed output; "
        "computed tensor accumulators must remain uncomposed until the "
        "connected region is bufferized/device-resident");

  llvm::SmallSet<int64_t, 16> inputModes;
  llvm::SmallSet<int64_t, 16> outputModes;
  for (unsigned tensor = 0; tensor + 1 < metadata.size(); ++tensor)
    for (int64_t mode : metadata[tensor].modes)
      inputModes.insert(mode);
  for (int64_t mode : metadata.back().modes) {
    if (!inputModes.contains(mode))
      return launch.emitError(
          "cuTensorNet network output mode is absent from all inputs");
    outputModes.insert(mode);
  }
  if (!llvm::any_of(inputModes, [&](int64_t mode) {
        return !outputModes.contains(mode);
      }))
    return launch.emitError(
        "cuTensorNet network must contain at least one reduced mode");

  int64_t tensorCount = metadata.size();
  int64_t metadataSize = 3 + tensorCount;
  for (const ContractionViewMetadata &view : metadata)
    metadataSize += 3 * view.modes.size();
  auto metadataType = MemRefType::get({metadataSize}, b.getI64Type());
  Value metadataBuffer = b.create<memref::AllocaOp>(loc, metadataType);
  auto pointerArrayType = MemRefType::get({tensorCount}, b.getI64Type());
  Value pointerArray = b.create<memref::AllocaOp>(loc, pointerArrayType);
  auto constantI64 = [&](int64_t value) -> Value {
    return b.create<arith::ConstantOp>(
        loc, b.getI64Type(), b.getI64IntegerAttr(value));
  };
  auto storeI64 = [&](Value buffer, int64_t index, Value value) {
    Value slot = b.create<arith::ConstantIndexOp>(loc, index);
    b.create<memref::StoreOp>(loc, value, buffer, slot);
  };
  storeI64(metadataBuffer, 0, constantI64(1)); // ABI version.
  storeI64(metadataBuffer, 1, constantI64(tensorCount - 1));
  storeI64(metadataBuffer, 2,
           constantI64(launch->hasAttr("network_accumulate") ? 1 : 0));
  int64_t metadataCursor = 3 + tensorCount;
  for (int64_t tensor = 0; tensor < tensorCount; ++tensor) {
    storeI64(metadataBuffer, 3 + tensor,
             constantI64(metadata[tensor].modes.size()));
    for (int64_t dim = 0; dim < (int64_t)metadata[tensor].modes.size();
         ++dim) {
      storeI64(metadataBuffer, metadataCursor++,
               metadata[tensor].extents[dim]);
      storeI64(metadataBuffer, metadataCursor++,
               metadata[tensor].strides[dim]);
      storeI64(metadataBuffer, metadataCursor++,
               constantI64(metadata[tensor].modes[dim]));
    }

    Value operand = launch.getOperand(tensorOrder[tensor]);
    Value pointer;
    if (isa<MemRefType>(operand.getType())) {
      pointer = memrefDataPtr(b, loc, operand);
    } else {
      Value memref = valueToMemref(b, loc, metadata[tensor].base);
      if (deviceResident && tensor + 1 < tensorCount &&
          metadata[tensor].needsDenseInputCopy)
        return launch.emitError(
            "device-resident network cannot materialize a host snapshot");
      if (tensor + 1 < tensorCount && metadata[tensor].needsDenseInputCopy)
        memref = snapshotOpaqueCallResult(b, loc, memref);
      pointer = memrefBasePtr(b, loc, memref);
      llvm::APInt staticOffset;
      if (!matchPattern(metadata[tensor].elementOffset,
                        m_ConstantInt(&staticOffset)) ||
          !staticOffset.isZero()) {
        Value address = b.create<LLVM::PtrToIntOp>(
            loc, b.getI64Type(), pointer);
        unsigned bits = expectedElementType.getIntOrFloatBitWidth();
        Value elementBytes = constantI64(bits / 8);
        Value byteOffset = b.create<arith::MulIOp>(
            loc, metadata[tensor].elementOffset, elementBytes);
        address = b.create<arith::AddIOp>(loc, address, byteOffset);
        pointer = b.create<LLVM::IntToPtrOp>(
            loc, LLVM::LLVMPointerType::get(b.getContext()), address);
      }
    }
    Value address = b.create<LLVM::PtrToIntOp>(
        loc, b.getI64Type(), pointer);
    storeI64(pointerArray, tensor, address);
  }

  Value pointerArrayPtr = memrefBasePtr(b, loc, pointerArray);
  Value metadataPtr = memrefBasePtr(b, loc, metadataBuffer);
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  StringRef shimName = useF64
      ? (deviceResident ? "polygeist_cutensornet_network_f64_device"
                        : "polygeist_cutensornet_network_f64")
      : (deviceResident ? "polygeist_cutensornet_network_f32_device"
                        : "polygeist_cutensornet_network_f32");
  func::FuncOp shim = ensureShimDecl(module, shimName,
                                     TypeRange{ptrType, ptrType}, b);
  auto call = b.create<func::CallOp>(
      loc, shim, ValueRange{pointerArrayPtr, metadataPtr});
  if (deviceResident)
    call->setAttr("polygeist.cuda_graph_safe", b.getUnitAttr());

  if (launch.getNumResults() == 1) {
    // The network is one synchronized, in-place write to its terminal DPS
    // destination.  Reconnect tensor SSA directly to that destination/base.
    // Materializing a compatibility snapshot here is not only unnecessary:
    // for a submap output, LowerSubmapInverse would later copy the stale
    // pre-call tensor back over the data just produced by cuTensorNet.
    Value outputView = launch.getOperand(outputOperand);
    Value outputBase = metadata.back().base;
    if (failed(rewireSubmapLaunchResult(launch, outputView, outputBase))) {
      return failure();
    }
  }
  launch.erase();
  return success();
}

// Darknet im2col+GEMM reaches the matcher as rank-3 broadcasted submaps:
//   A(m, k, n) -> weights[m, k]
//   B(m, k, n) -> workspace[k, n]
//   C(m, k, n) -> output[m, n]
// The underlying buffers are still regular row-major 2D GEMM operands, so
// unwrap the submaps and call the FP32 cuBLAS shim with M/N/K from the view
// sizes. The middle C dimension is the reduction/broadcast dimension and is
// ignored by the base output map.
static LogicalResult lowerSgemmBroadcast3DSimple(LaunchOp launch,
                                                 ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError(
        "cublasSgemm_broadcast3d_simple: expected A/B/C operands");
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cublasSgemm_broadcast3d_simple: expected 1 result");

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 3 ||
      Ct.getRank() != 3 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError(
        "cublasSgemm_broadcast3d_simple: A/B/C must be 3D f32 tensors");

  auto aSubmap = A.getDefiningOp<polygeist::SubmapOp>();
  auto bSubmap = B.getDefiningOp<polygeist::SubmapOp>();
  auto cSubmap = C.getDefiningOp<polygeist::SubmapOp>();
  if (!aSubmap || !bSubmap || !cSubmap || aSubmap.getSizes().size() != 3 ||
      bSubmap.getSizes().size() != 3 || cSubmap.getSizes().size() != 3)
    return launch.emitError(
        "cublasSgemm_broadcast3d_simple: operands must be rank-3 submaps");

  OpBuilder b(launch);
  Location loc = launch.getLoc();

  Value M = valueAsI32(b, loc, aSubmap.getSizes()[0]);
  Value K = valueAsI32(b, loc, aSubmap.getSizes()[1]);
  Value N = valueAsI32(b, loc, aSubmap.getSizes()[2]);
  Value alpha = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                            b.getF32FloatAttr(1.0));
  Value beta = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(1.0));

  Value A_base = resolveSubmapBase(A);
  Value B_base = resolveSubmapBase(B);
  Value C_base = resolveSubmapBase(C);
  auto A_base_type = dyn_cast<RankedTensorType>(A_base.getType());
  auto B_base_type = dyn_cast<RankedTensorType>(B_base.getType());
  auto C_base_type = dyn_cast<RankedTensorType>(C_base.getType());
  if (!A_base_type || !B_base_type || !C_base_type ||
      !A_base_type.getElementType().isF32() ||
      !B_base_type.getElementType().isF32() ||
      !C_base_type.getElementType().isF32())
    return launch.emitError(
        "cublasSgemm_broadcast3d_simple: submap bases must be f32 tensors");

  // Llama's split Q/K projections have the same scalar contraction body as
  // the Darknet rank-3 GEMM fixture, but a different physical layout:
  //   A[h,p,j] * x[j] -> C[h,p].
  // Treat the two parallel dimensions as one GEMV row dimension.  Reusing
  // the logical (H,P,J) sizes as (M,K,N) would read past x and C and was a
  // shape-dependent false lowering that happened to evade the tiny fixture.
  Value lda = K;
  Value ldb = N;
  Value ldc = N;
  if (A_base_type.getRank() == 3 && B_base_type.getRank() == 1 &&
      C_base_type.getRank() == 2) {
    Value H = valueAsI32(b, loc, aSubmap.getSizes()[0]);
    Value P = valueAsI32(b, loc, aSubmap.getSizes()[1]);
    Value J = valueAsI32(b, loc, aSubmap.getSizes()[2]);
    M = b.create<arith::MulIOp>(loc, H, P);
    N = b.create<arith::ConstantIntOp>(loc, 1, 32);
    K = J;
    lda = J;
    ldb = N;
    ldc = N;
  }

  Value A_mr = tensorToMemref(b, loc, A_base);
  Value B_mr = tensorToMemref(b, loc, B_base);
  Value C_mr = tensorToMemref(b, loc, C_base);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value B_ptr = memrefBasePtr(b, loc, B_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getF32Type(),
      ptrTy, b.getI32Type(),
      ptrTy, b.getI32Type(),
      b.getF32Type(),
      ptrTy, b.getI32Type(),
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_sgemm",
                                     argTypes, b);
  SmallVector<Value> callOperands = {M, N, K, alpha, A_ptr, lda,
                                     B_ptr, ldb, beta, C_ptr, ldc};
  b.create<func::CallOp>(loc, shim, callOperands);

  Value updatedBaseTensor = memrefToTensor(b, loc, C_mr, C_base.getType());
  rewireLaunchResult(launch, updatedBaseTensor);
  launch.erase();
  return success();
}

// Parboil basic SGEMM reaches Linalg as identity-indexed (M,N,K) views over
// flat column-major buffers:
//   A[m,n,k] -> A_base[m + k*lda]
//   B[m,n,k] -> B_base[n + k*ldb]
//   C[m,n,k] -> C_base[m + n*ldc]
// Reinterpret the buffers as row-major transposes and compute
// C^T = B^T * A^T through the existing arbitrary-alpha/beta SGEMM shim.
static LogicalResult lowerSgemmBroadcast3DColMajorNTAlphaBeta(
    LaunchOp launch, ModuleOp module) {
  StringRef name = "cublasSgemm_broadcast3d_colmajor_nt_alpha_beta";
  if (launch.getNumOperands() != 5 || launch.getNumResults() != 1)
    return launch.emitError(name) << ": expected A, B, C, beta, alpha";

  Value A = launch.getOperand(0), B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 3 ||
      Ct.getRank() != 2 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError(name)
           << ": A/B must be rank-3 and C rank-2 f32 tensors";

  auto aSubmap = A.getDefiningOp<polygeist::SubmapOp>();
  auto bSubmap = B.getDefiningOp<polygeist::SubmapOp>();
  auto cSubmap = C.getDefiningOp<polygeist::SubmapOp>();
  if (!aSubmap || !bSubmap || !cSubmap ||
      aSubmap.getSizes().size() != 3 || bSubmap.getSizes().size() != 3 ||
      cSubmap.getSizes().size() != 2 ||
      aSubmap.getSymbols().size() != 1 ||
      bSubmap.getSymbols().size() != 1 ||
      cSubmap.getSymbols().size() != 1)
    return launch.emitError(name)
           << ": expected rank-3 submaps with one leading-dimension symbol";

  MLIRContext *ctx = launch.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);
  AffineExpr d2 = getAffineDimExpr(2, ctx);
  AffineExpr s0 = getAffineSymbolExpr(0, ctx);
  AffineMap expectedA = AffineMap::get(3, 1, d2 * s0 + d0);
  AffineMap expectedB = AffineMap::get(3, 1, d2 * s0 + d1);
  AffineMap expectedC = AffineMap::get(
      2, 1, getAffineDimExpr(1, ctx) * s0 + getAffineDimExpr(0, ctx));
  if (aSubmap.getMap() != expectedA || bSubmap.getMap() != expectedB ||
      cSubmap.getMap() != expectedC)
    return launch.emitError(name)
           << ": submap layout does not implement column-major NT SGEMM";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M = valueAsI32(b, loc, aSubmap.getSizes()[0]);
  Value N = valueAsI32(b, loc, aSubmap.getSizes()[1]);
  Value K = valueAsI32(b, loc, aSubmap.getSizes()[2]);
  Value lda = valueAsI32(b, loc, aSubmap.getSymbols()[0]);
  Value ldb = valueAsI32(b, loc, bSubmap.getSymbols()[0]);
  Value ldc = valueAsI32(b, loc, cSubmap.getSymbols()[0]);
  Value beta = launch.getOperand(3), alpha = launch.getOperand(4);
  Value trans = b.create<arith::ConstantIntOp>(loc, 1, 32);
  Value noTrans = b.create<arith::ConstantIntOp>(loc, 0, 32);

  Value ABase = resolveSubmapBase(A);
  Value BBase = resolveSubmapBase(B);
  Value CBase = resolveSubmapBase(C);
  Value AMemref = tensorToMemref(b, loc, ABase);
  Value BMemref = tensorToMemref(b, loc, BBase);
  Value CMemref = tensorToMemref(b, loc, CBase);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getF32Type(), ptrTy, b.getI32Type(), ptrTy,
      b.getI32Type(), b.getF32Type(), ptrTy, b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_sgemm_transpose", argTypes, b);
  // Row-major C^T has shape N x M.  B_base is K x N row-major and must be
  // transposed; A_base is K x M row-major and is used directly.
  b.create<func::CallOp>(loc, shim,
      ValueRange{N, M, K, trans, noTrans, alpha,
                 memrefBasePtr(b, loc, BMemref), ldb,
                 memrefBasePtr(b, loc, AMemref), lda, beta,
                 memrefBasePtr(b, loc, CMemref), ldc});

  Value updatedBase = memrefToTensor(b, loc, CMemref, CBase.getType());
  rewireLaunchResult(launch, updatedBase);
  launch.erase();
  return success();
}

// Source-faithful Parboil SGEMM keeps A/B/C as flat memrefs and exposes the
// logical sizes and leading dimensions as scalar operands. This is the same
// column-major NT operation as the rank-3 submap form above, without requiring
// the matcher to synthesize temporary tensor views solely for the ABI.
static LogicalResult lowerSgemmFlatColMajorNTAlphaBeta(LaunchOp launch,
                                                       ModuleOp module) {
  StringRef name = "cublasSgemm_flat_colmajor_nt_alpha_beta";
  if (launch.getNumOperands() != 11 || launch.getNumResults() != 0)
    return launch.emitError(name)
           << ": expected A, B, C, M, N, K, lda, ldb, ldc, beta, alpha";
  for (unsigned i = 0; i < 3; ++i) {
    auto type = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError(name)
             << ": A/B/C must be rank-1 f32 memrefs";
  }
  if (!launch.getOperand(9).getType().isF32() ||
      !launch.getOperand(10).getType().isF32())
    return launch.emitError(name) << ": beta and alpha must be f32";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M = valueAsI32(b, loc, launch.getOperand(3));
  Value N = valueAsI32(b, loc, launch.getOperand(4));
  Value K = valueAsI32(b, loc, launch.getOperand(5));
  Value lda = valueAsI32(b, loc, launch.getOperand(6));
  Value ldb = valueAsI32(b, loc, launch.getOperand(7));
  Value ldc = valueAsI32(b, loc, launch.getOperand(8));
  Value beta = launch.getOperand(9), alpha = launch.getOperand(10);
  Value trans = b.create<arith::ConstantIntOp>(loc, 1, 32);
  Value noTrans = b.create<arith::ConstantIntOp>(loc, 0, 32);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getF32Type(), ptrTy, b.getI32Type(), ptrTy,
      b.getI32Type(), b.getF32Type(), ptrTy, b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_sgemm_transpose", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{N, M, K, trans, noTrans, alpha,
                 memrefBasePtr(b, loc, launch.getOperand(1)), ldb,
                 memrefBasePtr(b, loc, launch.getOperand(0)), lda, beta,
                 memrefBasePtr(b, loc, launch.getOperand(2)), ldc});
  launch.erase();
  return success();
}

static LogicalResult lowerSgemmStridedBatchedBroadcastRhs(LaunchOp launch,
                                                          ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "cublasSgemm_strided_batched_broadcast_rhs: expected A/B/C and one "
        "result");

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto Bt = dyn_cast<RankedTensorType>(B.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 2 ||
      Ct.getRank() != 3 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError(
        "cublasSgemm_strided_batched_broadcast_rhs: expected rank-3/rank-2/"
        "rank-3 f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value batch = dimForTensorOrMemrefAsI32(b, loc, A, 0);
  Value M = dimForTensorOrMemrefAsI32(b, loc, A, 1);
  Value K = dimForTensorOrMemrefAsI32(b, loc, A, 2);
  Value N = dimForTensorOrMemrefAsI32(b, loc, B, 1);
  Value A_mr = valueToMemrefPreservingSlice(b, loc, A);
  Value B_mr = valueToMemrefPreservingSlice(b, loc, B);
  Value C_mr = valueToMemrefPreservingSlice(b, loc, C);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value B_ptr = memrefBasePtr(b, loc, B_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), b.getI32Type(),
                                ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_sgemm_strided_batched_broadcast_rhs",
      argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{batch, M, N, K, A_ptr, B_ptr, C_ptr});

  Value out = memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, out,
                                tensorForSliceSource(b, loc, C));
  launch.erase();
  return success();
}

static LogicalResult lowerCusparseCsrSpmv(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 6 || launch.getNumResults() != 0)
    return launch.emitError(
        "cuSPARSE CSR SpMV: expected rows, row offsets, column indices, "
        "values, x, y and no result");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{valueAsI32(b, loc, launch.getOperand(0))};
  for (unsigned i = 1; i < 6; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  SmallVector<Type> types{b.getI32Type()};
  for (unsigned i = 1; i < 6; ++i) {
    types.push_back(b.getI32Type());
    types.push_back(ptrTy);
  }
  auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
  StringRef shim = shimSymbolFor(sym.getLeafReference().getValue());
  func::FuncOp decl = ensureShimDecl(module, shim, types, b);
  b.create<func::CallOp>(loc, decl, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCusparseSpmm(LaunchOp launch, ModuleOp module,
                                      StringRef shim, bool hasExplicitNnz) {
  unsigned scalarOperands = hasExplicitNnz ? 2 : 1;
  if (launch.getNumOperands() != scalarOperands + 5 ||
      launch.getNumResults() != 0)
    return launch.emitError(
        "cuSPARSE SpMM: expected rows, optional nnz, sparse indices, "
        "column indices, values, B, C and no result");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args;
  for (unsigned i = 0; i < scalarOperands; ++i)
    args.push_back(valueAsI32(b, loc, launch.getOperand(i)));
  for (unsigned i = scalarOperands; i < scalarOperands + 3; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  for (unsigned i = scalarOperands + 3; i < scalarOperands + 5; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 1));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  SmallVector<Type> types(scalarOperands, b.getI32Type());
  for (unsigned i = scalarOperands; i < scalarOperands + 3; ++i) {
    types.push_back(b.getI32Type());
    types.push_back(ptrTy);
  }
  for (unsigned i = scalarOperands + 3; i < scalarOperands + 5; ++i) {
    types.push_back(b.getI32Type());
    types.push_back(b.getI32Type());
    types.push_back(ptrTy);
  }
  func::FuncOp decl = ensureShimDecl(module, shim, types, b);
  b.create<func::CallOp>(loc, decl, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCusparseBsrSpmm(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 7 || launch.getNumResults() != 0)
    return launch.emitError(
        "cuSPARSE BSR SpMM: expected block rows, block dimension, row "
        "offsets, column indices, values, x, y and no result");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{
      valueAsI32(b, loc, launch.getOperand(0)),
      valueAsI32(b, loc, launch.getOperand(1))};
  for (unsigned i = 2; i < 4; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  Value values = launch.getOperand(4);
  for (unsigned dimension = 0; dimension < 3; ++dimension)
    args.push_back(dimForTensorOrMemrefAsI32(b, loc, values, dimension));
  args.push_back(memrefDataPtr(b, loc, values));
  for (unsigned i = 5; i < 7; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  SmallVector<Type> types{
      b.getI32Type(), b.getI32Type(), b.getI32Type(), ptrTy,
      b.getI32Type(), ptrTy, b.getI32Type(), b.getI32Type(),
      b.getI32Type(), ptrTy, b.getI32Type(), ptrTy, b.getI32Type(), ptrTy};
  func::FuncOp decl = ensureShimDecl(
      module, "polygeist_cusparse_spmm_bsr_f32_sized", types, b);
  b.create<func::CallOp>(loc, decl, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCusparseJdsSpmv(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 9 || launch.getNumResults() != 0)
    return launch.emitError(
        "cuSPARSE JDS SpMV adapter: expected rows, repetitions, seven JDS/vector "
        "memrefs and no result");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{valueAsI32(b, loc, launch.getOperand(0)),
                          valueAsI32(b, loc, launch.getOperand(1))};
  for (unsigned i = 2; i < 9; ++i) {
    args.push_back(
        dimForTensorOrMemrefAsI32(b, loc, launch.getOperand(i), 0));
    args.push_back(memrefDataPtr(b, loc, launch.getOperand(i)));
  }
  SmallVector<Type> types{b.getI32Type(), b.getI32Type()};
  for (unsigned i = 2; i < 9; ++i) {
    types.push_back(b.getI32Type());
    types.push_back(ptrTy);
  }
  func::FuncOp decl = ensureShimDecl(
      module, "polygeist_cusparse_spmv_jds_f32_sized", types, b);
  b.create<func::CallOp>(loc, decl, args);
  launch.erase();
  return success();
}

static LogicalResult lowerSgemmBroadcast3DMemRef(LaunchOp launch,
                                                 ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError(
        "cublasSgemm_broadcast3d_memref: expected A/B/C operands");
  if (launch.getNumResults() != 0)
    return launch.emitError(
        "cublasSgemm_broadcast3d_memref: expected no results");

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto At = dyn_cast<MemRefType>(A.getType());
  auto Bt = dyn_cast<MemRefType>(B.getType());
  auto Ct = dyn_cast<MemRefType>(C.getType());
  if (!At || !Bt || !Ct || At.getRank() != 3 || Bt.getRank() != 3 ||
      Ct.getRank() != 3 || !At.getElementType().isF32() ||
      !Bt.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError(
        "cublasSgemm_broadcast3d_memref: A/B/C must be 3D f32 memrefs");

  auto aSubmap = A.getDefiningOp<polygeist::SubmapOp>();
  auto bSubmap = B.getDefiningOp<polygeist::SubmapOp>();
  auto cSubmap = C.getDefiningOp<polygeist::SubmapOp>();
  if (!aSubmap || !bSubmap || !cSubmap || aSubmap.getSizes().size() != 3 ||
      bSubmap.getSizes().size() != 3 || cSubmap.getSizes().size() != 3)
    return launch.emitError(
        "cublasSgemm_broadcast3d_memref: operands must be rank-3 submaps");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M = valueAsI32(b, loc, aSubmap.getSizes()[0]);
  Value K = valueAsI32(b, loc, aSubmap.getSizes()[1]);
  Value N = valueAsI32(b, loc, aSubmap.getSizes()[2]);
  Value alpha = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                            b.getF32FloatAttr(1.0));
  Value beta = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(1.0));

  Value A_base = aSubmap.getBase();
  Value B_base = bSubmap.getBase();
  Value C_base = cSubmap.getBase();
  auto ABaseType = dyn_cast<MemRefType>(A_base.getType());
  auto BBaseType = dyn_cast<MemRefType>(B_base.getType());
  auto CBaseType = dyn_cast<MemRefType>(C_base.getType());
  if (!ABaseType || !BBaseType || !CBaseType ||
      !ABaseType.getElementType().isF32() ||
      !BBaseType.getElementType().isF32() ||
      !CBaseType.getElementType().isF32())
    return launch.emitError(
        "cublasSgemm_broadcast3d_memref: submap bases must be f32 memrefs");

  Value A_ptr = memrefBasePtr(b, loc, A_base);
  Value B_ptr = memrefBasePtr(b, loc, B_base);
  Value C_ptr = memrefBasePtr(b, loc, C_base);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getF32Type(),
      ptrTy, b.getI32Type(),
      ptrTy, b.getI32Type(),
      b.getF32Type(),
      ptrTy, b.getI32Type(),
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_sgemm",
                                     argTypes, b);
  SmallVector<Value> callOperands = {M, N, K, alpha, A_ptr, K,
                                     B_ptr, N, beta, C_ptr, N};
  b.create<func::CallOp>(loc, shim, callOperands);
  launch.erase();
  return success();
}

// @cublasDgeam_scale2D(%M : tensor<?x?xf64>, %scale : f64) -> tensor<?x?xf64>
// Diagonal/scale-only geam: M = scale * M, in place.
static LogicalResult lowerDgeamScale2D(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("cublasDgeam_scale2D: expected 2 operands");
  Value M = launch.getOperand(0);
  Value scale = launch.getOperand(1);
  auto Mt = dyn_cast<RankedTensorType>(M.getType());
  if (!Mt || Mt.getRank() != 2 || !Mt.getElementType().isF64())
    return launch.emitError("cublasDgeam_scale2D: M must be 2D f64 tensor");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M_mr = tensorToMemref(b, loc, M);
  Value rows = memrefDimAsI32(b, loc, M_mr, 0);
  Value cols = memrefDimAsI32(b, loc, M_mr, 1);
  Value M_ptr = memrefBasePtr(b, loc, M_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                 b.getF64Type(), ptrTy, b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dscal_2d",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{rows, cols, scale, M_ptr, cols});

  Value out = memrefToTensor(b, loc, M_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// The actual @cudnnConvolution2D_9tap lowering body is shared with
// LowerKernelLaunchToPVA via KernelLaunchLoweringUtils.cpp. Bring it into
// this file's scope so the dispatch switch below can name it unqualified.
using mlir::polygeist::lowerCudnnConv2D9tap;
using mlir::polygeist::lowerCudnnConv2D25tap;
using mlir::polygeist::lowerCudnnConv2DNtapPacked;

// Shared lowering for tensor GEMV. D/S variants differ only in element type
// and runtime shim symbol; transpose picks A*x vs A^T*x.
static LogicalResult lowerDgemvImpl(LaunchOp launch, ModuleOp module,
                                       bool transpose, bool useF32,
                                       bool subtract = false,
                                       bool overwrite = false);

static LogicalResult lowerDgemv(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/false, /*useF32=*/false);
}

static LogicalResult lowerDgemvT(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true, /*useF32=*/false);
}

static LogicalResult lowerDgemvTZero(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true, /*useF32=*/false,
                        /*subtract=*/false, /*overwrite=*/true);
}

static LogicalResult lowerDgemvSubtract(LaunchOp launch, ModuleOp module,
                                        bool transpose) {
  return lowerDgemvImpl(launch, module, transpose, /*useF32=*/false,
                        /*subtract=*/true);
}

static LogicalResult lowerSgemv(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/false, /*useF32=*/true);
}

static LogicalResult lowerSgemvT(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true, /*useF32=*/true);
}

// @cublasDgemv(%A : tensor<MxNxf64>, %x : tensor<Nxf64>, %y : tensor<Mxf64>)
//   -> tensor<Mxf64>
// Computes y += A * x. The canonical kernel definition retains the output
// accumulator, so its BLAS beta is 1.
//
// cuBLAS gemv signature (in our row-major convention):
//   polygeist_cublas_dgemv(M, N, alpha, A*, lda, x*, beta, y*)
static LogicalResult lowerDgemvImpl(LaunchOp launch, ModuleOp module,
                                       bool transpose, bool useF32,
                                       bool subtract, bool overwrite) {
  StringRef libName = useF32 ? "cublasSgemv" : "cublasDgemv";
  StringRef elemName = useF32 ? "f32" : "f64";
  if (launch.getNumOperands() != 3)
    return launch.emitError(libName)
           << " lowering: expected 3 operands (A, x, y), got "
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(libName) << " lowering: expected 1 result";

  Value A = launch.getOperand(0);
  Value x = launch.getOperand(1);
  Value y = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  auto yt = dyn_cast<RankedTensorType>(y.getType());
  auto hasElem = [&](Type ty) { return useF32 ? ty.isF32() : ty.isF64(); };
  if (!At || At.getRank() != 2 || !hasElem(At.getElementType()))
    return launch.emitError(libName)
           << " lowering: A must be 2D " << elemName << " tensor";
  if (!xt || xt.getRank() != 1 || !hasElem(xt.getElementType()))
    return launch.emitError(libName)
           << " lowering: x must be 1D " << elemName << " tensor";
  if (!yt || yt.getRank() != 1 || !hasElem(yt.getElementType()))
    return launch.emitError(libName)
           << " lowering: y must be 1D " << elemName << " tensor";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Type scalarTy = useF32 ? b.getF32Type() : b.getF64Type();
  TypedAttr oneAttr = useF32 ? b.getF32FloatAttr(1.0f)
                             : b.getF64FloatAttr(1.0);
  Value one = b.create<arith::ConstantOp>(loc, scalarTy, oneAttr);
  TypedAttr zeroAttr = useF32 ? b.getF32FloatAttr(0.0f)
                              : b.getF64FloatAttr(0.0);
  Value beta = overwrite
                   ? b.create<arith::ConstantOp>(loc, scalarTy, zeroAttr)
                   : one;
  TypedAttr minusOneAttr = useF32 ? b.getF32FloatAttr(-1.0f)
                                  : b.getF64FloatAttr(-1.0);
  Value alpha = subtract
                    ? b.create<arith::ConstantOp>(loc, scalarTy, minusOneAttr)
                    : one;

  // Do not blindly materialize tensor operands with bufferization.to_memref.
  // Matched GEMVs commonly consume tensor.extract_slice views of the original
  // C ABI memrefs.  A to_memref here makes one-shot-bufferize allocate and
  // copy the complete matrix/vector before the cuBLAS call; it also makes
  // device-resident C ABI pointers unsafe because that copy executes on the
  // host.  Preserve those views and derive the runtime pointers directly from
  // their source buffers instead.
  Value M = dimForTensorOrMemrefAsI32(b, loc, A, 0);
  Value N = dimForTensorOrMemrefAsI32(b, loc, A, 1);
  Value lda = N;  // row-major
  if (auto slice = stripTensorCasts(A).getDefiningOp<tensor::ExtractSliceOp>()) {
    auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
    auto mixedStrides = slice.getMixedStrides();
    bool unitMinorStride = mixedStrides.size() == 2 &&
        getConstantIntValue(mixedStrides[1]).value_or(0) == 1;
    if (sourceType && sourceType.getRank() == 2 && unitMinorStride) {
      Value axis = b.create<arith::ConstantIndexOp>(loc, 1);
      Value sourceCols = b.create<tensor::DimOp>(loc, slice.getSource(), axis);
      lda = b.create<arith::IndexCastOp>(loc, b.getI32Type(), sourceCols);
    }
  }

  Value A_ptr = pointerForTensorOrMemref(b, loc, A);
  Value x_ptr = pointerForTensorOrMemref(b, loc, x);
  Value y_mr = valueToOutputMemrefPreservingSlice(b, loc, y);
  Value y_ptr = memrefDataPtr(b, loc, y_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(),   // M, N (A's row-major shape)
      scalarTy,                          // alpha
      ptrTy, b.getI32Type(),             // A*, lda
      ptrTy,                             // x*
      scalarTy,                          // beta
      ptrTy,                             // y*
  };
  StringRef shimSym =
      useF32 ? (transpose ? "polygeist_cublas_sgemv_T"
                          : "polygeist_cublas_sgemv")
             : (transpose ? "polygeist_cublas_dgemv_T"
                          : "polygeist_cublas_dgemv");
  func::FuncOp shim = ensureShimDecl(module, shimSym, argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{M, N, alpha, A_ptr, lda, x_ptr, beta, y_ptr});

  if (overwrite) {
    // The beta=0 form was redirected to its final destination by the
    // matcher. The opaque call mutates that storage directly; retaining a
    // functional insert_slice result would recreate a loop-carried tensor
    // allocation and later copy stale storage over the call's output.
    launch.getResult(0).replaceAllUsesWith(y);
    launch.erase();
    return success();
  }

  // Keep the output as a view of its original destination buffer and bypass
  // the canonical tensor.insert_slice write-back.  The opaque cuBLAS call has
  // already updated that storage in place.
  Value updatedView =
      memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, y);
  rewireTensorSliceLaunchResult(launch, updatedView, updatedBase);
  launch.erase();
  return success();
}

// @cublasDaxpby(%x : tensor<Nxf64>, %y : tensor<Nxf64>, %alpha : f64, %beta : f64)
//   -> tensor<Nxf64>
// Computes y = α*x + β*y. Output (the second tensor) is updated in place.
static LogicalResult lowerDaxpby(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 4)
    return launch.emitError("cublasDaxpby: expected 4 operands (x, y, α, β)");
  Value x = launch.getOperand(0);
  Value y = launch.getOperand(1);
  Value alpha = launch.getOperand(2);
  Value beta = launch.getOperand(3);
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  auto yt = dyn_cast<RankedTensorType>(y.getType());
  if (!xt || xt.getRank() != 1 || !xt.getElementType().isF64() ||
      !yt || yt.getRank() != 1 || !yt.getElementType().isF64())
    return launch.emitError("cublasDaxpby: x,y must be 1D f64 tensors");
  if (!alpha.getType().isF64() || !beta.getType().isF64())
    return launch.emitError("cublasDaxpby: α,β must be f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value x_mr = tensorToMemref(b, loc, x);
  Value y_mr = tensorToMemref(b, loc, y);
  Value N = memrefDimAsI32(b, loc, y_mr, 0);
  Value x_ptr = memrefBasePtr(b, loc, x_mr);
  Value y_ptr = memrefBasePtr(b, loc, y_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getF64Type(), ptrTy,
                                 b.getF64Type(), ptrTy};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_daxpby",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{N, alpha, x_ptr, beta, y_ptr});
  Value out = memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerSaxpby(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() > 1)
    return launch.emitError("cublasSaxpby: expected x, y, alpha, beta and one result");
  Value x = launch.getOperand(0), y = launch.getOperand(1);
  Value alpha = launch.getOperand(2), beta = launch.getOperand(3);
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  auto yt = dyn_cast<RankedTensorType>(y.getType());
  if (!xt || !yt || xt.getRank() != 1 || yt.getRank() != 1 ||
      !xt.getElementType().isF32() || !yt.getElementType().isF32() ||
      !alpha.getType().isF32() || !beta.getType().isF32())
    return launch.emitError("cublasSaxpby: requires rank-1 f32 vectors and f32 coefficients");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value x_mr = tensorToMemref(b, loc, x);
  Value y_mr = tensorToMemref(b, loc, y);
  Value N = memrefDimAsI32(b, loc, y_mr, 0);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type(), b.getF32Type(), ptrTy,
                             b.getF32Type(), ptrTy};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_saxpby", types, b);
  b.create<func::CallOp>(loc, shim, ValueRange{
      N, alpha, memrefBasePtr(b, loc, x_mr), beta, memrefBasePtr(b, loc, y_mr)});
  Value out = memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

static LogicalResult lowerSscal(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 1)
    return launch.emitError("cublasSscal: expected x, scale and one result");
  Value x = launch.getOperand(0), scale = launch.getOperand(1);
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  if (!xt || xt.getRank() != 1 || !xt.getElementType().isF32() ||
      !scale.getType().isF32())
    return launch.emitError("cublasSscal: requires a rank-1 f32 vector and f32 scale");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value x_mr = tensorToMemref(b, loc, x);
  Value N = memrefDimAsI32(b, loc, x_mr, 0);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type(), b.getF32Type(), ptrTy};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_sscal", types, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{N, scale, memrefBasePtr(b, loc, x_mr)});
  Value out = memrefToTensor(b, loc, x_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// @cublasDaxpy_unit(%x : tensor<Nxf64>, %y : tensor<Nxf64>) -> tensor<Nxf64>
// Computes y += x. α=1, no β scale.
static LogicalResult lowerDaxpyUnit(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("cublasDaxpy_unit: expected 2 operands (x, y)");
  Value x = launch.getOperand(0);
  Value y = launch.getOperand(1);
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  auto yt = dyn_cast<RankedTensorType>(y.getType());
  if (!xt || xt.getRank() != 1 || !xt.getElementType().isF64() ||
      !yt || yt.getRank() != 1 || !yt.getElementType().isF64())
    return launch.emitError("cublasDaxpy_unit: x,y must be 1D f64 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value x_mr = tensorToMemref(b, loc, x);
  Value y_mr = tensorToMemref(b, loc, y);
  Value N = memrefDimAsI32(b, loc, y_mr, 0);
  Value x_ptr = memrefBasePtr(b, loc, x_mr);
  Value y_ptr = memrefBasePtr(b, loc, y_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_daxpy_unit",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, x_ptr, y_ptr});
  Value out = memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// @cublasDgemv_alpha(%A, %x, %y, %alpha) → tensor (y += α·A·x)
static LogicalResult lowerDgemvAlpha(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 4)
    return launch.emitError(
        "cublasDgemv_alpha: expected 4 operands (A, x, y, α)");
  Value A = launch.getOperand(0);
  Value x = launch.getOperand(1);
  Value y = launch.getOperand(2);
  Value alpha = launch.getOperand(3);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  if (!At || At.getRank() != 2 || !At.getElementType().isF64())
    return launch.emitError("cublasDgemv_alpha: A must be 2D f64");
  if (!alpha.getType().isF64())
    return launch.emitError("cublasDgemv_alpha: α must be f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value one = b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                          b.getF64FloatAttr(1.0));
  Value A_mr = tensorToMemref(b, loc, A);
  Value x_mr = tensorToMemref(b, loc, x);
  Value y_mr = tensorToMemref(b, loc, y);
  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value N = memrefDimAsI32(b, loc, A_mr, 1);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value x_ptr = memrefBasePtr(b, loc, x_mr);
  Value y_ptr = memrefBasePtr(b, loc, y_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  // Use the same dgemv shim but with α from launch and β=1 (accumulate).
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getF64Type(),
      ptrTy, b.getI32Type(), ptrTy, b.getF64Type(), ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dgemv",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{M, N, alpha, A_ptr, N, x_ptr, one, y_ptr});
  Value out = memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// @cublasDger_rank2(%u1, %v1, %u2, %v2, %A) → tensor<NxN>
// Rank-2 update: A = A + u1·v1ᵀ + u2·v2ᵀ.
static LogicalResult lowerDgerRank2(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 5)
    return launch.emitError(
        "cublasDger_rank2: expected 5 operands (u1, v1, u2, v2, A)");
  Value A = launch.getOperand(4);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  if (!At || At.getRank() != 2 || !At.getElementType().isF64())
    return launch.emitError("cublasDger_rank2: A must be 2D f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, A);
  SmallVector<Value> vec_mrs;
  for (unsigned i = 0; i < 4; ++i)
    vec_mrs.push_back(tensorToMemref(b, loc, launch.getOperand(i)));
  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value N = memrefDimAsI32(b, loc, A_mr, 1);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  SmallVector<Value> vec_ptrs;
  for (Value v : vec_mrs) vec_ptrs.push_back(memrefBasePtr(b, loc, v));

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  // (M, N, u1, v1, u2, v2, A, lda)
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, b.getI32Type(),
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dger_rank2",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{M, N,
                 vec_ptrs[0], vec_ptrs[1], vec_ptrs[2], vec_ptrs[3],
                 A_ptr, N});
  Value out = memrefToTensor(b, loc, A_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// @cublasDgemm_outer_product(%u, %v, %C) -> tensor<MxNxf64>
// Computes C = u*v^T.  The runtime deliberately overwrites C, rather than
// exposing BLAS GER's accumulator semantics, so a preceding zero-fill stage
// can be removed as part of the matched composition.
static LogicalResult lowerDgemmOuterProduct(LaunchOp launch,
                                            ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "cublasDgemm_outer_product: expected u/v/C and one result");
  Value u = launch.getOperand(0);
  Value v = launch.getOperand(1);
  Value C = launch.getOperand(2);
  auto ut = dyn_cast<RankedTensorType>(u.getType());
  auto vt = dyn_cast<RankedTensorType>(v.getType());
  auto Ct = dyn_cast<RankedTensorType>(C.getType());
  if (!ut || !vt || !Ct || ut.getRank() != 1 || vt.getRank() != 1 ||
      Ct.getRank() != 2 || !ut.getElementType().isF64() ||
      !vt.getElementType().isF64() || !Ct.getElementType().isF64())
    return launch.emitError(
        "cublasDgemm_outer_product: expected rank-1/rank-1/rank-2 f64 "
        "tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M = dimForTensorOrMemrefAsI32(b, loc, C, 0);
  Value N = dimForTensorOrMemrefAsI32(b, loc, C, 1);
  Value u_mr = valueToMemrefPreservingSlice(b, loc, u);
  Value v_mr = valueToMemrefPreservingSlice(b, loc, v);
  Value C_mr = valueToOutputMemrefPreservingSlice(b, loc, C);
  Value u_ptr = memrefBasePtr(b, loc, u_mr);
  Value v_ptr = memrefBasePtr(b, loc, v_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy,
                                ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_dgemm_outer_product", argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{M, N, u_ptr, v_ptr, C_ptr});

  Value out = memrefToTensor(b, loc, C_mr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, out, tensorForOutputSliceSource(b, loc, C));
  launch.erase();
  return success();
}

// @memset_zero_1D(%v : tensor<Nxf64>) -> tensor<Nxf64>
// @memset_zero_1D_f32(%v : tensor<Nxf32>) -> tensor<Nxf32>
static LogicalResult lowerMemsetZero1D(LaunchOp launch, ModuleOp module,
                                       StringRef variant) {
  if (launch.getNumOperands() != 1)
    return launch.emitError(variant) << ": expected 1 operand";
  Value V = launch.getOperand(0);
  auto Vt = dyn_cast<RankedTensorType>(V.getType());
  bool isF32Variant = variant == "memset_zero_1D_f32";
  if (!Vt || Vt.getRank() != 1 ||
      (isF32Variant ? !Vt.getElementType().isF32()
                    : !Vt.getElementType().isF64()))
    return launch.emitError(variant)
           << ": V must be a 1D "
           << (isF32Variant ? "f32" : "f64") << " tensor";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value len = dimForTensorOrMemrefAsI32(b, loc, V, 0);
  Value V_ptr = pointerForTensorOrMemref(b, loc, V);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy};
  StringRef shimName = isF32Variant ? "polygeist_cublas_memset_zero_1d_f32"
                                    : "polygeist_cublas_memset_zero_1d";
  func::FuncOp shim = ensureShimDecl(module, shimName, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{len, V_ptr});

  Value V_mr = valueToMemrefPreservingSlice(b, loc, V);
  Value updatedView =
      memrefToTensor(b, loc, V_mr, launch.getResult(0).getType());
  Value updatedBase = tensorForSliceSource(b, loc, V);
  rewireTensorSliceLaunchResult(launch, updatedView, updatedBase);
  launch.erase();
  return success();
}

// @memset_zero_2D(%M : tensor<?x?xf{32,64}>) -> tensor<?x?xf{32,64}>
// Dtype-agnostic: zero is the same bit pattern at any width, so we
// dispatch to a single host-side memset that takes a byte count.
static LogicalResult lowerMemsetZero2D(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 1)
    return launch.emitError("memset_zero_2D: expected 1 operand");
  Value M = launch.getOperand(0);
  auto Mt = dyn_cast<RankedTensorType>(M.getType());
  if (!Mt || Mt.getRank() != 2 ||
      !(Mt.getElementType().isF32() || Mt.getElementType().isF64()))
    return launch.emitError(
        "memset_zero_2D: M must be 2D f32 or f64 tensor");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M_mr = tensorToMemref(b, loc, M);
  Value rows = memrefDimAsI32(b, loc, M_mr, 0);
  Value cols = memrefDimAsI32(b, loc, M_mr, 1);
  Value M_ptr = memrefBasePtr(b, loc, M_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy,
                                 b.getI32Type()};
  // Pick the dtype-suffixed memset shim. The cuBLAS memset is just
  // a host-side `memset(ptr, 0, M*N*sizeof(elem))` — but it has to
  // know which sizeof to use, so we emit a different symbol per dtype.
  StringRef memsetSym = Mt.getElementType().isF64()
      ? "polygeist_cublas_memset_zero_2d"
      : "polygeist_cublas_memset_zero_2d_f32";
  func::FuncOp shim = ensureShimDecl(module, memsetSym, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{rows, cols, M_ptr, cols});

  Value out = memrefToTensor(b, loc, M_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
  launch.erase();
  return success();
}

// @cudnnConvolutionFwd_batched(%input_view, %filter, %output_view)
//
// The matcher fires this two-step composition (init-to-zero + the
// 7-iter par×4+red×3 contraction) when the IR matches a batched
// multi-channel 2D conv (NCHW). The launch operands are:
//   - input_view: 7D `polygeist.submap` view of the underlying
//     `tensor<?xICxHxWxf32>` (the strided window — implicit im2col).
//   - filter: plain `tensor<?xICxKxKxf32>` (no submap).
//   - output_view: 4D submap view of the underlying `tensor<?xOCxOHxOWxf32>`.
//
// Lowers to:
//   polygeist_cudnn_conv2d_batched(B, IC, OC, H, W, K, A*, F*, Out*)
//
// where the shape ints are recovered from the base 4D shapes (the
// output 4D submap has the same shape as the underlying Bout tensor).
static LogicalResult lowerCudnnConv2dBatched(LaunchOp launch,
                                             ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cudnnConvolutionFwd_batched: expected 3 "
                            "operands (input_view, filter, output_view); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cudnnConvolutionFwd_batched: expected 1 result");

  Value inputView  = launch.getOperand(0);
  Value filterView = launch.getOperand(1);
  Value outputView = launch.getOperand(2);

  // linalg-debufferize wraps every tensor operand of the contraction
  // generic in a polygeist.submap — even the filter (conceptually a
  // plain 4D tensor). Resolve all three back to their underlying base.
  Value inputBase  = resolveSubmapBase(inputView);
  Value filterBase = resolveSubmapBase(filterView);
  Value outputBase = resolveSubmapBase(outputView);

  auto inT = dyn_cast<RankedTensorType>(inputBase.getType());
  auto fT  = dyn_cast<RankedTensorType>(filterBase.getType());
  auto oT  = dyn_cast<RankedTensorType>(outputBase.getType());
  if (!inT || !fT || !oT || inT.getRank() != 4 || fT.getRank() != 4 ||
      oT.getRank() != 4)
    return launch.emitError(
        "cudnnConvolutionFwd_batched: input/filter/output must each be "
        "4D after resolving submap (NCHW)");
  Type elemTy = inT.getElementType();
  if (!elemTy.isF32() || fT.getElementType() != elemTy ||
      oT.getElementType() != elemTy)
    return launch.emitError(
        "cudnnConvolutionFwd_batched: only f32 supported for now; got ")
           << elemTy;

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, inputBase);
  Value F_mr = tensorToMemref(b, loc, filterBase);
  Value O_mr = valueToOutputMemrefPreservingSlice(b, loc, outputBase);

  // Shape recovery: B = dim(in, 0), IC = dim(in, 1) = dim(filter, 1),
  // OC = dim(filter, 0), H = dim(in, 2), W = dim(in, 3),
  // K = dim(filter, 2) (assume square 3D filter K==dim(filter,3)).
  Value B  = memrefDimAsI32(b, loc, A_mr, 0);
  Value IC = memrefDimAsI32(b, loc, A_mr, 1);
  Value OC = memrefDimAsI32(b, loc, F_mr, 0);
  Value H  = memrefDimAsI32(b, loc, A_mr, 2);
  Value W  = memrefDimAsI32(b, loc, A_mr, 3);
  Value K  = memrefDimAsI32(b, loc, F_mr, 2);

  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value F_ptr = memrefBasePtr(b, loc, F_mr);
  Value O_ptr = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cudnn_conv2d_batched",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{B, IC, OC, H, W, K, A_ptr, F_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outputBase.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, outputBase));
  launch.erase();
  return success();
}

// @cudnnConvolutionFwd_im2col_gemm(%input, %weights_view, %output,
//                                  channels, height, width, out_channels,
//                                  ksize, stride, pad)
//
// This is the explicit Darknet im2col + GEMM composition:
//   zero(output); workspace = im2col(input); output += weights * workspace
// The matcher has already proven the guarded im2col body and GEMM body are
// adjacent. Lower the whole composition to one cuDNN convolution call, avoiding
// materialization of the workspace.
static LogicalResult lowerCudnnConv2dIm2colGemm(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 10)
    return launch.emitError("cudnnConvolutionFwd_im2col_gemm: expected 10 "
                            "operands (input, weights, output, 7 shape ints); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 0)
    return launch.emitError(
        "cudnnConvolutionFwd_im2col_gemm: expected no results");

  Value input = launch.getOperand(0);
  Value weightsView = launch.getOperand(1);
  Value output = launch.getOperand(2);

  auto inputTy = dyn_cast<MemRefType>(input.getType());
  auto weightsTy = dyn_cast<MemRefType>(weightsView.getType());
  auto outputTy = dyn_cast<MemRefType>(output.getType());
  if (!inputTy || !weightsTy || !outputTy || inputTy.getRank() != 1 ||
      weightsTy.getRank() != 3 || outputTy.getRank() != 1 ||
      !inputTy.getElementType().isF32() ||
      !weightsTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32())
    return launch.emitError(
        "cudnnConvolutionFwd_im2col_gemm: expected f32 input/output flat "
        "memrefs and a rank-3 f32 weights submap");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value IC = valueAsI32(b, loc, launch.getOperand(3));
  Value H = valueAsI32(b, loc, launch.getOperand(4));
  Value W = valueAsI32(b, loc, launch.getOperand(5));
  Value OC = valueAsI32(b, loc, launch.getOperand(6));
  Value K = valueAsI32(b, loc, launch.getOperand(7));
  Value S = valueAsI32(b, loc, launch.getOperand(8));
  Value P = valueAsI32(b, loc, launch.getOperand(9));

  Value weightsBase = resolveSubmapBase(weightsView);
  Value A_ptr = memrefBasePtr(b, loc, input);
  Value F_ptr = memrefBasePtr(b, loc, weightsBase);
  Value O_ptr = memrefBasePtr(b, loc, output);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv2d_im2col_gemm_f32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim, ValueRange{IC, H, W, OC, K, S, P, A_ptr, F_ptr, O_ptr});

  launch.erase();
  return success();
}

// Channel-preserving fixed-window reduction lowered as grouped/depthwise
// cuDNN convolution. The matcher has already proved the affine access:
//   input[n,c,oh*SH+kh*DH-PH,ow*SW+kw*DW-PW]
// and the uniform multiply-add reduction body.
static LogicalResult lowerCudnnUniformWindowConv2DF32(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 11 || launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvolution2DWindow_f32: expected input, output, weight, "
        "KH, KW, SH, SW, DH, DW, PH, PW and one result");
  Value input = launch.getOperand(0);
  Value output = launch.getOperand(1);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != 4 ||
      outputType.getRank() != 4 || !inputType.getElementType().isF32() ||
      !outputType.getElementType().isF32() ||
      !launch.getOperand(2).getType().isF32())
    return launch.emitError(
        "cudnnConvolution2DWindow_f32: input/output must be rank-4 f32 "
        "tensors and weight must be f32");
  for (unsigned i = 3; i < 11; ++i)
    if (!launch.getOperand(i).getType().isInteger(32))
      return launch.emitError(
          "cudnnConvolution2DWindow_f32: window parameters must be i32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMemref = valueToMemrefPreservingSlice(b, loc, input);
  Value outputMemref = valueToOutputMemrefPreservingSlice(b, loc, output);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, inputMemref, 0),
      memrefDimAsI32(b, loc, inputMemref, 1),
      memrefDimAsI32(b, loc, inputMemref, 2),
      memrefDimAsI32(b, loc, inputMemref, 3),
      memrefDimAsI32(b, loc, outputMemref, 2),
      memrefDimAsI32(b, loc, outputMemref, 3),
      launch.getOperand(2),
  };
  args.append(launch.getOperands().begin() + 3,
              launch.getOperands().begin() + 11);
  args.push_back(memrefDataPtr(b, loc, inputMemref));
  args.push_back(memrefDataPtr(b, loc, outputMemref));

  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(6, b.getI32Type());
  argTypes.push_back(b.getF32Type());
  argTypes.append(8, b.getI32Type());
  argTypes.append(2, ptrType);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_conv2d_uniform_window_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);

  Value updated = memrefToTensor(
      b, loc, outputMemref, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

// Tensor-form average pooling. Same operand layout as
// cudnnConvolution2DWindow_f32 (input, output, weight, KH,KW,SH,SW,DH,DW,PH,PW
// -> outputTensor), but the matcher has proved the weight is exactly 1/(KH*KW)
// over a non-overlapping valid window, so this is a box average. Marshal it to
// the existing cuDNN pooling shim (op-tag 4 = fixed-window average forward)
// instead of a grouped/depthwise convolution.
static LogicalResult lowerCudnnAvgPoolWindowF32(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 11 || launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnAvgPoolWindow_f32: expected input, output, weight, "
        "KH, KW, SH, SW, DH, DW, PH, PW and one result");
  Value input = launch.getOperand(0);
  Value output = launch.getOperand(1);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != 4 ||
      outputType.getRank() != 4 || !inputType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return launch.emitError(
        "cudnnAvgPoolWindow_f32: input/output must be rank-4 f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMemref = valueToMemrefPreservingSlice(b, loc, input);
  Value outputMemref = valueToOutputMemrefPreservingSlice(b, loc, output);
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  auto ci32 = [&](int32_t v) {
    return b.create<arith::ConstantIntOp>(loc, v, 32).getResult();
  };
  // polygeist_cudnn_adaptive_pool_f32(op, rank, N, C, i0,i1,i2, o0,o1,o2,
  //                                   inPtr, outPtr, idxPtr)
  SmallVector<Value> args = {
      ci32(4),                                  // op: fixed-window avg forward
      ci32(2),                                  // spatial rank
      memrefDimAsI32(b, loc, inputMemref, 0),   // N
      memrefDimAsI32(b, loc, inputMemref, 1),   // C
      memrefDimAsI32(b, loc, inputMemref, 2),   // i0 = H
      memrefDimAsI32(b, loc, inputMemref, 3),   // i1 = W
      ci32(1),                                  // i2 (unused for 2D)
      memrefDimAsI32(b, loc, outputMemref, 2),  // o0 = OH
      memrefDimAsI32(b, loc, outputMemref, 3),  // o1 = OW
      ci32(1),                                  // o2 (unused for 2D)
  };
  args.push_back(memrefDataPtr(b, loc, inputMemref));
  args.push_back(memrefDataPtr(b, loc, outputMemref));
  args.push_back(b.create<LLVM::ZeroOp>(loc, ptrType));  // no index tensor

  SmallVector<Type> argTypes(10, b.getI32Type());
  argTypes.append(3, ptrType);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_adaptive_pool_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, args);

  Value updated = memrefToTensor(
      b, loc, outputMemref, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

// Rank-generic adaptive average/max pooling. The matcher has recovered the
// exact ATen floor/ceil partition and emits raw pointers because the extracted
// CPU fixtures use both flattened and shaped memrefs for the same semantics.
static LogicalResult lowerCudnnAdaptivePoolF32(LaunchOp launch,
                                               ModuleOp module) {
  if ((launch.getNumOperands() != 12 && launch.getNumOperands() != 13) ||
      launch.getNumResults() != 0)
    return launch.emitError(
        "cudnnAdaptivePool_f32: expected 10 i32 parameters, 2 or 3 "
        "buffers, and no results");
  for (unsigned i = 0; i < 10; ++i)
    if (!launch.getOperand(i).getType().isInteger(32))
      return launch.emitError(
          "cudnnAdaptivePool_f32: parameters 0..9 must be i32");
  for (unsigned i = 10; i < launch.getNumOperands(); ++i)
    if (!isa<BaseMemRefType, RankedTensorType>(launch.getOperand(i).getType()))
      return launch.emitError(
          "cudnnAdaptivePool_f32: trailing operands must be shaped buffers");

  OpBuilder b(launch);
  auto ptrType = LLVM::LLVMPointerType::get(launch.getContext());
  SmallVector<Type> argTypes(10, b.getI32Type());
  argTypes.append(3, ptrType);
  SmallVector<Value> args(launch.getOperands().begin(),
                          launch.getOperands().begin() + 10);
  for (unsigned i = 10; i < launch.getNumOperands(); ++i)
    args.push_back(pointerForTensorOrMemref(
        b, launch.getLoc(), launch.getOperand(i)));
  if (args.size() == 12)
    args.push_back(b.create<LLVM::ZeroOp>(launch.getLoc(), ptrType));
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_adaptive_pool_f32", argTypes, b);
  b.create<func::CallOp>(launch.getLoc(), shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnBatchNormBackwardF32(LaunchOp launch,
                                                    ModuleOp module,
                                                    bool fullOutputs) {
  unsigned expected = fullOutputs ? 11 : 8;
  if (launch.getNumOperands() != expected || launch.getNumResults() != 0)
    return launch.emitError("cudnnBatchNormBackward_f32: unexpected ABI");
  for (unsigned i = 0; i < 3; ++i)
    if (!launch.getOperand(i).getType().isInteger(32))
      return launch.emitError("cudnnBatchNormBackward_f32: N/C/S must be i32");

  OpBuilder b(launch);
  auto ptrType = LLVM::LLVMPointerType::get(launch.getContext());
  SmallVector<Type> argTypes(3, b.getI32Type());
  argTypes.push_back(b.getI32Type());
  argTypes.append(8, ptrType);
  SmallVector<Value> args(launch.getOperands().begin(),
                          launch.getOperands().begin() + 3);
  args.push_back(b.create<arith::ConstantIntOp>(launch.getLoc(),
                                                fullOutputs ? 1 : 0, 32));
  if (fullOutputs) {
    for (unsigned i = 3; i < launch.getNumOperands(); ++i)
      args.push_back(pointerForTensorOrMemref(
          b, launch.getLoc(), launch.getOperand(i)));
  } else {
    // grad, x, mean, invstd, [implicit unit weight], dx,
    // [discarded dweight], [discarded dbias]
    for (unsigned i = 3; i < 7; ++i)
      args.push_back(pointerForTensorOrMemref(
          b, launch.getLoc(), launch.getOperand(i)));
    args.push_back(b.create<LLVM::ZeroOp>(launch.getLoc(), ptrType));
    args.push_back(pointerForTensorOrMemref(
        b, launch.getLoc(), launch.getOperand(7)));
    args.push_back(b.create<LLVM::ZeroOp>(launch.getLoc(), ptrType));
    args.push_back(b.create<LLVM::ZeroOp>(launch.getLoc(), ptrType));
  }
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_batchnorm_backward_f32", argTypes, b);
  b.create<func::CallOp>(launch.getLoc(), shim, args);
  launch.erase();
  return success();
}

// @cudnnMaxPoolFwd_batched(%input_view, %output_view)
//   Inputs: input (6D submap of 4D base), output (4D submap of 4D base).
// Lowers to polygeist_cudnn_maxpool_batched(B, C, H, W, K, S, A*, Out*).
//
// The window size K and stride S are encoded in the submap's affine map
// constants (we hard-code 2 + S from typical maxpool, but recover them
// at runtime from the base / output dim ratio: K = ((H - (OH-1)*S) → we
// pass the *output* dims separately and let the shim's pooling descriptor
// derive K = H - (OH-1)*S, treating stride and window as equal to
// (H/OH) — works for typical 2x2 stride-2 maxpool).
//
// To keep the shim simple, we *also* pass K + S as ints. Recovering them
// from the submap's affine map would need C++ introspection of an
// AffineMap; instead, the harness passes the matched window/stride in
// via the wrapper. For the polybench-style extracted kernels here we
// know K, S at compile time (MINI: K=S=2). We embed those as compile-
// time constants in the kernel C source and read them at runtime via
// the harness — see the maxpool_batched.c harness for the convention.
//
// Simpler approach: just pass H, W, OH, OW. The shim derives
//   S = (H - K) / (OH - 1) once K is fixed; or for the common stride==K
//   case, S = H / OH and K = S.
// Since both extracted shapes (MINI: K=S=2; LARGE: K=3, S=2) have known
// values, we pass them as separate ints from the harness via the
// wrapper, NOT from MLIR (the matcher doesn't preserve them).
//
// The MLIR-level call therefore passes B, C, H, W (from base/output
// dims) and the runtime shim looks up K, S from per-call thread-locals
// set by the wrapper. This is documented in polygeist_cublas_rt.h.
static LogicalResult lowerCudnnMaxpoolBatched(LaunchOp launch,
                                              ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("cudnnMaxPoolFwd_batched: expected 2 operands "
                            "(input_view, output_view); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cudnnMaxPoolFwd_batched: expected 1 result");

  Value inView  = launch.getOperand(0);
  Value outView = launch.getOperand(1);
  Value inBase  = resolveSubmapBase(inView);
  Value outBase = resolveSubmapBase(outView);

  auto inT  = dyn_cast<RankedTensorType>(inBase.getType());
  auto outT = dyn_cast<RankedTensorType>(outBase.getType());
  if (!inT || !outT || inT.getRank() != 4 || outT.getRank() != 4)
    return launch.emitError("cudnnMaxPoolFwd_batched: both operands must "
                            "be 4D after resolving submap");
  Type elemTy = inT.getElementType();
  if (!elemTy.isF32() || outT.getElementType() != elemTy)
    return launch.emitError("cudnnMaxPoolFwd_batched: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, inBase);
  // Keep the selected output slice: the physical parent can have a larger
  // dynamic batch extent than the fixed region recognized by the matcher.
  // H/W come from the input base because the rank-6 window submap encodes
  // them indirectly, while N/C/OH/OW are explicit on the output slice.
  Value O_mr = valueToOutputMemrefPreservingSlice(b, loc, outView);
  Value B  = memrefDimAsI32(b, loc, O_mr, 0);
  Value C  = memrefDimAsI32(b, loc, O_mr, 1);
  Value H  = memrefDimAsI32(b, loc, A_mr, 2);
  Value W  = memrefDimAsI32(b, loc, A_mr, 3);
  Value OH = memrefDimAsI32(b, loc, O_mr, 2);
  Value OW = memrefDimAsI32(b, loc, O_mr, 3);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value O_ptr = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cudnn_maxpool_batched",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{B, C, H, W, OH, OW, A_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outBase.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, outBase));
  launch.erase();
  return success();
}

// @cudnnBatchNormalizationForwardInference(
//     %A_view, %scale_view, %mean_view, %inv_std_view, %bias_view,
//     %output_view)
//
// All 6 operands are submap views. The raise pass orders them
// (A, scale, mean, inv_std, bias) — see the matcher template
// (_cudnn_batchnorm_inference) for the order. After walking through
// submaps:
//   - scale, mean, inv_std, bias are 1D tensors (per-channel)
//   - A and output are 4D tensors (NCHW)
//
// Lowers to:
//   polygeist_cudnn_batchnorm_inference(B, C, H, W,
//                                          A*, scale*, mean*, inv_std*, bias*,
//                                          Out*)
static LogicalResult lowerCudnnBatchnormInference(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 6)
    return launch.emitError(
        "cudnnBatchNormalizationForwardInference: expected 6 operands; got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnBatchNormalizationForwardInference: expected 1 result");

  Value aBase       = resolveSubmapBase(launch.getOperand(0));
  Value scaleBase   = resolveSubmapBase(launch.getOperand(1));
  Value meanBase    = resolveSubmapBase(launch.getOperand(2));
  Value invStdBase  = resolveSubmapBase(launch.getOperand(3));
  Value biasBase    = resolveSubmapBase(launch.getOperand(4));
  Value outBase     = resolveSubmapBase(launch.getOperand(5));

  auto aT = dyn_cast<RankedTensorType>(aBase.getType());
  auto oT = dyn_cast<RankedTensorType>(outBase.getType());
  if (!aT || !oT || aT.getRank() != 4 || oT.getRank() != 4)
    return launch.emitError(
        "batchnorm: A and Out must be 4D after resolving submap");
  Type elemTy = aT.getElementType();
  if (!elemTy.isF32() || oT.getElementType() != elemTy)
    return launch.emitError("batchnorm: only f32 supported");
  for (Value v : {scaleBase, meanBase, invStdBase, biasBase}) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != 1 || t.getElementType() != elemTy)
      return launch.emitError(
          "batchnorm: scale/mean/inv_std/bias must be 1D f32 per-channel "
          "after resolving submap");
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr  = tensorToMemref(b, loc, aBase);
  Value S_mr  = tensorToMemref(b, loc, scaleBase);
  Value M_mr  = tensorToMemref(b, loc, meanBase);
  Value I_mr  = tensorToMemref(b, loc, invStdBase);
  Value Bi_mr = tensorToMemref(b, loc, biasBase);
  Value O_mr  = valueToOutputMemrefPreservingSlice(b, loc, outBase);

  Value B = memrefDimAsI32(b, loc, A_mr, 0);
  Value C = memrefDimAsI32(b, loc, A_mr, 1);
  Value H = memrefDimAsI32(b, loc, A_mr, 2);
  Value W = memrefDimAsI32(b, loc, A_mr, 3);

  Value A_ptr  = memrefBasePtr(b, loc, A_mr);
  Value S_ptr  = memrefBasePtr(b, loc, S_mr);
  Value M_ptr  = memrefBasePtr(b, loc, M_mr);
  Value I_ptr  = memrefBasePtr(b, loc, I_mr);
  Value Bi_ptr = memrefBasePtr(b, loc, Bi_mr);
  Value O_ptr  = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cudnn_batchnorm_inference", argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{B, C, H, W, A_ptr, S_ptr, M_ptr, I_ptr, Bi_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outBase.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, outBase));
  launch.erase();
  return success();
}

// @cudnnAddTensor_batched(%input_view, %output_view)
//   out[b,c,h,w] += in[b,c,h,w]  — ResNet residual add.
// Lowers to polygeist_cudnn_add_tensor_batched(B, C, H, W, A*, Out*).
static LogicalResult lowerCudnnAddTensorBatched(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("cudnnAddTensor_batched: expected 2 operands");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudnnAddTensor_batched: expected 1 result");

  Value inBase  = resolveSubmapBase(launch.getOperand(0));
  Value outBase = resolveSubmapBase(launch.getOperand(1));
  auto inT  = dyn_cast<RankedTensorType>(inBase.getType());
  auto outT = dyn_cast<RankedTensorType>(outBase.getType());
  if (!inT || !outT || inT.getRank() != 4 || outT.getRank() != 4)
    return launch.emitError(
        "cudnnAddTensor_batched: both operands must be 4D after submap");
  Type elemTy = inT.getElementType();
  if (!elemTy.isF32() || outT.getElementType() != elemTy)
    return launch.emitError("cudnnAddTensor_batched: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, inBase);
  Value O_mr = tensorToMemref(b, loc, outBase);
  Value B = memrefDimAsI32(b, loc, A_mr, 0);
  Value C = memrefDimAsI32(b, loc, A_mr, 1);
  Value H = memrefDimAsI32(b, loc, A_mr, 2);
  Value W = memrefDimAsI32(b, loc, A_mr, 3);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value O_ptr = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cudnn_add_tensor_batched", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{B, C, H, W, A_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outBase.getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, outBase));
  launch.erase();
  return success();
}

// @cudnnConvBnReluFwdFused(%input_view, %filter_view, %scale_view, %mean_view,
//                          %inv_std_view, %bias_view, %output_view)
//
// 7 operands. The matcher emits this for the canonical ResNet inner
// pattern conv + bn-inference + relu. After resolving submaps:
//   - input  (4D NCHW): from the conv's input submap
//   - filter (4D OCxICxKxK): from the conv's filter submap
//   - scale, mean, inv_std, bias (1D length OC): the BN per-channel vectors
//   - output (4D NCHW): the in-place destination
//
// Lowers to one call:
//   polygeist_cudnn_conv_bn_relu_fused(
//       B, IC, OC, H, W, K, A*, F*, scale*, mean*, inv_std*, bias*, Out*)
//
// The runtime shim folds the BN params into a scaled filter + bias and
// uses cudnnConvolutionBiasActivationForward (which natively does
// conv+bias+activation in one call) with CUDNN_ACTIVATION_RELU.
static LogicalResult lowerCudnnConvBnReluFused(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 7)
    return launch.emitError("cudnnConvBnReluFwdFused: expected 7 operands, got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cudnnConvBnReluFwdFused: expected 1 result");

  Value inputBase   = resolveSubmapBase(launch.getOperand(0));
  Value filterBase  = resolveSubmapBase(launch.getOperand(1));
  Value scaleBase   = resolveSubmapBase(launch.getOperand(2));
  Value meanBase    = resolveSubmapBase(launch.getOperand(3));
  Value invStdBase  = resolveSubmapBase(launch.getOperand(4));
  Value biasBase    = resolveSubmapBase(launch.getOperand(5));
  Value outBase     = resolveSubmapBase(launch.getOperand(6));

  auto inT  = dyn_cast<RankedTensorType>(inputBase.getType());
  auto fT   = dyn_cast<RankedTensorType>(filterBase.getType());
  auto outT = dyn_cast<RankedTensorType>(outBase.getType());
  if (!inT || !fT || !outT ||
      inT.getRank() != 4 || fT.getRank() != 4 || outT.getRank() != 4)
    return launch.emitError(
        "cudnnConvBnReluFwdFused: input/filter/output must each be 4D "
        "after resolving submap");
  Type elemTy = inT.getElementType();
  if (!elemTy.isF32() || fT.getElementType() != elemTy ||
      outT.getElementType() != elemTy)
    return launch.emitError("cudnnConvBnReluFwdFused: only f32 supported");
  for (Value v : {scaleBase, meanBase, invStdBase, biasBase}) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != 1 || t.getElementType() != elemTy)
      return launch.emitError(
          "cudnnConvBnReluFwdFused: scale/mean/inv_std/bias must be 1D f32");
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr  = tensorToMemref(b, loc, inputBase);
  Value F_mr  = tensorToMemref(b, loc, filterBase);
  Value S_mr  = tensorToMemref(b, loc, scaleBase);
  Value M_mr  = tensorToMemref(b, loc, meanBase);
  Value I_mr  = tensorToMemref(b, loc, invStdBase);
  Value Bi_mr = tensorToMemref(b, loc, biasBase);
  Value O_mr  = tensorToMemref(b, loc, outBase);

  Value B  = memrefDimAsI32(b, loc, A_mr, 0);
  Value IC = memrefDimAsI32(b, loc, A_mr, 1);
  Value OC = memrefDimAsI32(b, loc, F_mr, 0);
  Value H  = memrefDimAsI32(b, loc, A_mr, 2);
  Value W  = memrefDimAsI32(b, loc, A_mr, 3);
  Value K  = memrefDimAsI32(b, loc, F_mr, 2);

  Value A_ptr  = memrefBasePtr(b, loc, A_mr);
  Value F_ptr  = memrefBasePtr(b, loc, F_mr);
  Value S_ptr  = memrefBasePtr(b, loc, S_mr);
  Value M_ptr  = memrefBasePtr(b, loc, M_mr);
  Value I_ptr  = memrefBasePtr(b, loc, I_mr);
  Value Bi_ptr = memrefBasePtr(b, loc, Bi_mr);
  Value O_ptr  = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),  // B, IC, OC
      b.getI32Type(), b.getI32Type(), b.getI32Type(),  // H, W, K
      ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, // A, F, scale, mean, inv_std, bias, Out
  };
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cudnn_conv_bn_relu_fused", argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{B, IC, OC, H, W, K,
                 A_ptr, F_ptr, S_ptr, M_ptr, I_ptr, Bi_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outBase.getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

// @cudnnConvBiasReluAddFwdFused(%input, %filter, %op0, %op1, %output)
//
// Five linalg.generic ops folded into one launch by the matcher. The
// last two pre-relu ins (steps 2 + 3, both `Out + In(0)` body shape)
// are NOT distinguishable at the matcher level — both are
// "Out + In". The lowering disambiguates by operand rank after
// resolving submap:
//   • 1D operand → bias (per-output-channel, broadcast)
//   • 4D operand → residual (same shape as output, the Z addend)
//
// Routes to:
//   polygeist_cudnn_conv_bias_relu_add_fused(B, IC, OC, H, W, K,
//       A*, F*, bias*, Z*, Out*)
//
// The shim then issues one cudnnConvolutionBiasActivationForward with
// α₁=1, α₂=1 and CUDNN_ACTIVATION_RELU.
static LogicalResult lowerCudnnConvBiasReluAdd(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 5)
    return launch.emitError(
        "cudnnConvBiasReluAddFwdFused: expected 5 operands, got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cudnnConvBiasReluAddFwdFused: expected 1 result");

  Value inputBase  = resolveSubmapBase(launch.getOperand(0));
  Value filterBase = resolveSubmapBase(launch.getOperand(1));
  Value addOp0     = resolveSubmapBase(launch.getOperand(2));
  Value addOp1     = resolveSubmapBase(launch.getOperand(3));
  Value outBase    = resolveSubmapBase(launch.getOperand(4));

  // Disambiguate bias vs residual by rank of the underlying base.
  auto rankOf = [](Value v) -> int {
    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
      return t.getRank();
    return -1;
  };
  Value biasBase, residualBase;
  if (rankOf(addOp0) == 1 && rankOf(addOp1) == 4) {
    biasBase = addOp0; residualBase = addOp1;
  } else if (rankOf(addOp0) == 4 && rankOf(addOp1) == 1) {
    biasBase = addOp1; residualBase = addOp0;
  } else {
    return launch.emitError(
        "cudnnConvBiasReluAddFwdFused: addend operands must be one 1D "
        "(bias) and one 4D (residual), got ranks ")
           << rankOf(addOp0) << " and " << rankOf(addOp1);
  }

  auto inT  = dyn_cast<RankedTensorType>(inputBase.getType());
  auto fT   = dyn_cast<RankedTensorType>(filterBase.getType());
  auto outT = dyn_cast<RankedTensorType>(outBase.getType());
  auto bT   = dyn_cast<RankedTensorType>(biasBase.getType());
  auto rT   = dyn_cast<RankedTensorType>(residualBase.getType());
  if (!inT || !fT || !outT || !bT || !rT)
    return launch.emitError("cudnnConvBiasReluAddFwdFused: non-tensor operand");
  Type elemTy = inT.getElementType();
  if (!elemTy.isF32() || fT.getElementType() != elemTy ||
      outT.getElementType() != elemTy || bT.getElementType() != elemTy ||
      rT.getElementType() != elemTy)
    return launch.emitError("cudnnConvBiasReluAddFwdFused: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr  = tensorToMemref(b, loc, inputBase);
  Value F_mr  = tensorToMemref(b, loc, filterBase);
  Value Bi_mr = tensorToMemref(b, loc, biasBase);
  Value Z_mr  = tensorToMemref(b, loc, residualBase);
  Value O_mr  = tensorToMemref(b, loc, outBase);

  Value B  = memrefDimAsI32(b, loc, A_mr, 0);
  Value IC = memrefDimAsI32(b, loc, A_mr, 1);
  Value OC = memrefDimAsI32(b, loc, F_mr, 0);
  Value H  = memrefDimAsI32(b, loc, A_mr, 2);
  Value W  = memrefDimAsI32(b, loc, A_mr, 3);
  Value K  = memrefDimAsI32(b, loc, F_mr, 2);

  Value A_ptr  = memrefBasePtr(b, loc, A_mr);
  Value F_ptr  = memrefBasePtr(b, loc, F_mr);
  Value Bi_ptr = memrefBasePtr(b, loc, Bi_mr);
  Value Z_ptr  = memrefBasePtr(b, loc, Z_mr);
  Value O_ptr  = memrefBasePtr(b, loc, O_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cudnn_conv_bias_relu_add_fused", argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{B, IC, OC, H, W, K,
                 A_ptr, F_ptr, Bi_ptr, Z_ptr, O_ptr});

  Value updated = memrefToTensor(b, loc, O_mr, outBase.getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

// Runtime computes:
//   out[i] = weight[i] * x[i] * rsqrt(sum_j x[j]^2 / N + 1e-5)
// @cudnnPointwiseAffineRelu_f32(%x, %bias, %out, %alpha), FP32 1D.
// Runtime executes the two-node cuDNN graph:
//   tmp = alpha * x + bias; out = relu(tmp)
static LogicalResult lowerCudnnPointwiseAffineReluF32(LaunchOp launch,
                                                       ModuleOp module) {
  if (launch.getNumOperands() != 4)
    return launch.emitError(
        "cudnn pointwise affine+relu: expected (x, bias, out, alpha)");
  if (launch.getNumResults() > 1)
    return launch.emitError(
        "cudnn pointwise affine+relu: expected zero or one result");

  Value x = resolveSubmapBase(launch.getOperand(0));
  Value bias = resolveSubmapBase(launch.getOperand(1));
  Value out = resolveSubmapBase(launch.getOperand(2));
  Value alpha = launch.getOperand(3);
  ShapedType xTy = getRankedShapedType(x);
  ShapedType bTy = getRankedShapedType(bias);
  ShapedType oTy = getRankedShapedType(out);
  if (!xTy || !bTy || !oTy || xTy.getRank() != 1 || bTy.getRank() != 1 ||
      oTy.getRank() != 1)
    return launch.emitError(
        "cudnn pointwise affine+relu: x/bias/out must be ranked 1D");
  if (!xTy.getElementType().isF32() ||
      bTy.getElementType() != xTy.getElementType() ||
      oTy.getElementType() != xTy.getElementType() ||
      !alpha.getType().isF32())
    return launch.emitError(
        "cudnn pointwise affine+relu: only f32 is supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value biasMr = valueToMemrefPreservingSlice(b, loc, bias);
  Value outMr = valueToOutputMemrefPreservingSlice(b, loc, out);
  Value n = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value biasPtr = memrefBasePtr(b, loc, biasMr);
  Value outPtr = memrefBasePtr(b, loc, outMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getF32Type(), ptrTy, ptrTy,
                                ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_pointwise_affine_relu_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{n, alpha, xPtr, biasPtr, outPtr});

  if (launch.getNumResults() == 1) {
    Value updated =
        memrefToTensor(b, loc, outMr, launch.getResult(0).getType());
    rewireTensorSliceLaunchResult(
        launch, updated, tensorForOutputSliceSource(b, loc, out));
  }
  launch.erase();
  return success();
}

// Generic bounded f32 pointwise DAG. Static graph bytecode is carried on the
// launch as twelve i64 words; ordinary operands carry four tensor inputs, one
// output, and eight by-value broadcast scalars.
static LogicalResult lowerCudnnPointwiseGraphF32(LaunchOp launch,
                                                  ModuleOp module) {
  bool isBufferized = launch->hasAttr("polygeist.bufferized");
  unsigned expectedResults = isBufferized ? 0 : 1;
  if (launch.getNumOperands() != 13 ||
      launch.getNumResults() != expectedResults)
    return launch.emitError(
        "cudnn pointwise graph: expected 4 inputs, out, 8 scalars, and "
        "one tensor result (or no result after bufferization)");
  if (isBufferized) {
    auto destinations = launch->getAttrOfType<DenseI64ArrayAttr>(
        "polygeist.result_destinations");
    if (!destinations || destinations.size() != 1 || destinations[0] != 4)
      return launch.emitError(
          "cudnn pointwise graph: bufferized result must alias operand 4");
  }
  auto graph = launch->getAttrOfType<DenseI64ArrayAttr>("pointwise_graph");
  auto nodeCount =
      launch->getAttrOfType<IntegerAttr>("pointwise_num_nodes");
  if (!graph || (graph.size() != 8 && graph.size() != 12) || !nodeCount ||
      nodeCount.getInt() <= 0 || nodeCount.getInt() > 24)
    return launch.emitError("cudnn pointwise graph: invalid graph bytecode");

  SmallVector<Value> tensors;
  tensors.reserve(5);
  for (unsigned i = 0; i < 5; ++i) {
    Value value = resolveSubmapBase(launch.getOperand(i));
    ShapedType ty = getRankedShapedType(value);
    if (!ty || ty.getRank() != 1 || !ty.getElementType().isF32())
      return launch.emitError(
          "cudnn pointwise graph: tensors must be rank-1 f32");
    tensors.push_back(value);
  }
  for (unsigned i = 5; i < 13; ++i)
    if (!launch.getOperand(i).getType().isF32())
      return launch.emitError("cudnn pointwise graph: scalars must be f32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  SmallVector<Value> memrefs;
  SmallVector<Value> ptrs;
  SmallVector<Value> strides;
  for (unsigned i = 0; i < 5; ++i) {
    Value mr = i == 4
        ? valueToOutputMemrefPreservingSlice(b, loc, tensors[i])
        : valueToMemrefPreservingSlice(b, loc, tensors[i]);
    memrefs.push_back(mr);
    // Preserve both the extract_slice offset and its physical element stride.
    // A rank-1 pointwise view is not necessarily contiguous (cross-product
    // components, for example, are every third element of an Nx3 tensor).
    ptrs.push_back(memrefDataPtr(b, loc, mr));
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, mr);
    strides.push_back(valueAsI32(b, loc, metadata.getStrides()[0]));
  }
  Value n = memrefDimAsI32(b, loc, memrefs[0], 0);
  SmallVector<Value> graphWords;
  graphWords.reserve(12);
  for (int64_t word : graph.asArrayRef())
    graphWords.push_back(
        b.create<arith::ConstantIntOp>(loc, word, 64));
  while (graphWords.size() < 12)
    graphWords.push_back(b.create<arith::ConstantIntOp>(loc, 0, 64));
  Value nodes =
      b.create<arith::ConstantIntOp>(loc, nodeCount.getInt(), 32);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type()};
  argTypes.append(12, b.getI64Type());
  argTypes.append({
      b.getI32Type(),
      b.getF32Type(), b.getF32Type(), b.getF32Type(), b.getF32Type(),
      b.getF32Type(), b.getF32Type(), b.getF32Type(), b.getF32Type(),
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      b.getI32Type(),
      ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_pointwise_graph_f32", argTypes, b);
  SmallVector<Value> args = {n};
  args.append(graphWords);
  args.push_back(nodes);
  args.append(launch.getOperands().begin() + 5,
              launch.getOperands().begin() + 13);
  args.append(strides);
  args.append(ptrs);
  b.create<func::CallOp>(loc, shim, args);

  if (!isBufferized) {
    Value updated =
        memrefToTensor(b, loc, memrefs[4], launch.getResult(0).getType());
    rewireTensorSliceLaunchResult(
        launch, updated, tensorForOutputSliceSource(b, loc, tensors[4]));
  }
  launch.erase();
  return success();
}

static LogicalResult lowerCubInclusiveSum1DF32(LaunchOp launch,
                                                ModuleOp module) {
  bool bufferized = launch->hasAttr("polygeist.bufferized");
  if (launch.getNumOperands() != 3 ||
      launch.getNumResults() != (bufferized ? 0u : 2u))
    return launch.emitError(
        "CUB inclusive sum expects input, final scalar, and output");
  if (bufferized) {
    auto destinations = launch->getAttrOfType<DenseI64ArrayAttr>(
        "polygeist.result_destinations");
    if (!destinations || destinations.size() != 2 ||
        destinations[0] != 1 || destinations[1] != 2)
      return launch.emitError("CUB inclusive sum has invalid destinations");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value finalValue =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(1));
  Value output =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(2));
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto finalType = dyn_cast<MemRefType>(finalValue.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!inputType || !finalType || !outputType || inputType.getRank() != 1 ||
      finalType.getRank() != 0 || outputType.getRank() != 1 ||
      !inputType.getElementType().isF32() ||
      !finalType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return launch.emitError("CUB inclusive sum requires f32 [N], scalar, [N]");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cub_inclusive_sum1d_f32",
      {b.getI32Type(), ptr, ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, input, 0),
                 memrefDataPtr(b, loc, input),
                 memrefDataPtr(b, loc, finalValue),
                 memrefDataPtr(b, loc, output)});
  if (!bufferized) {
    Value finalTensor =
        memrefToTensor(b, loc, finalValue, launch.getResult(0).getType());
    Value outputTensor =
        memrefToTensor(b, loc, output, launch.getResult(1).getType());
    launch.getResult(0).replaceAllUsesWith(finalTensor);
    rewireTensorSliceLaunchResult(
        launch, outputTensor,
        tensorForOutputSliceSource(b, loc, launch.getOperand(2)), 1);
  }
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedInclusiveProduct2DF32(
    LaunchOp launch, ModuleOp module) {
  bool bufferized = launch->hasAttr("polygeist.bufferized");
  if (launch.getNumOperands() != 3 ||
      launch.getNumResults() != (bufferized ? 0u : 2u))
    return launch.emitError(
        "CUB segmented product expects input, output, and final values");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value output =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(1));
  Value finalValues =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(2));
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  auto finalType = dyn_cast<MemRefType>(finalValues.getType());
  if (!inputType || !outputType || !finalType || inputType.getRank() != 2 ||
      outputType.getRank() != 2 || finalType.getRank() != 1 ||
      !inputType.getElementType().isF32() ||
      !outputType.getElementType().isF32() ||
      !finalType.getElementType().isF32())
    return launch.emitError(
        "CUB segmented product requires f32 [R,K], [R,K], [R]");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cub_segmented_inclusive_product2d_f32",
      {b.getI32Type(), b.getI32Type(), ptr, ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, input, 0),
                 memrefDimAsI32(b, loc, input, 1),
                 memrefDataPtr(b, loc, input),
                 memrefDataPtr(b, loc, finalValues),
                 memrefDataPtr(b, loc, output)});
  if (!bufferized) {
    Value outputTensor =
        memrefToTensor(b, loc, output, launch.getResult(0).getType());
    Value finalTensor =
        memrefToTensor(b, loc, finalValues, launch.getResult(1).getType());
    rewireTensorSliceLaunchResult(
        launch, outputTensor,
        tensorForOutputSliceSource(b, loc, launch.getOperand(1)), 0);
    if (!launch.getResult(1).use_empty())
      launch.getResult(1).replaceAllUsesWith(finalTensor);
  }
  launch.erase();
  return success();
}

static LogicalResult lowerCubExclusiveSum1DI32(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 0)
    return launch.emitError("CUB exclusive sum expects input and output");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value input = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value output =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(1));
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != 1 ||
      outputType.getRank() != 1 || !inputType.getElementType().isInteger(32) ||
      !outputType.getElementType().isInteger(32))
    return launch.emitError("CUB exclusive sum requires i32 [N] buffers");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(module, "polygeist_cub_exclusive_sum1d_i32",
                             {b.getI32Type(), ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, input, 0),
                 memrefDataPtr(b, loc, input),
                 memrefDataPtr(b, loc, output)});
  launch.erase();
  return success();
}

static LogicalResult lowerCubHistogramEvenI32ShiftZero(LaunchOp launch,
                                                        ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError(
        "CUB integer histogram expects samples and histogram buffers");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value samples = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value histogram =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(1));
  auto samplesType = dyn_cast<MemRefType>(samples.getType());
  auto histogramType = dyn_cast<MemRefType>(histogram.getType());
  if (!samplesType || !histogramType || samplesType.getRank() != 1 ||
      histogramType.getRank() != 1 ||
      !samplesType.getElementType().isInteger(32) ||
      !histogramType.getElementType().isInteger(32))
    return launch.emitError("CUB integer histogram requires i32 [N] buffers");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cub_histogram_even_i32_shift_zero",
      {b.getI32Type(), b.getI32Type(), ptr, ptr, b.getI32Type()}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, samples, 0),
                 memrefDimAsI32(b, loc, histogram, 0),
                 memrefDataPtr(b, loc, samples),
                 memrefDataPtr(b, loc, histogram), launch.getOperand(2)});
  launch.erase();
  return success();
}

static LogicalResult lowerDtrsvLowerRowMajor(LaunchOp launch,
                                              ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("row-major DTRSV expects A, b, and x buffers");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value matrix = valueToMemrefPreservingSlice(b, loc, launch.getOperand(0));
  Value rhs = valueToMemrefPreservingSlice(b, loc, launch.getOperand(1));
  Value output =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(2));
  auto matrixType = dyn_cast<MemRefType>(matrix.getType());
  auto rhsType = dyn_cast<MemRefType>(rhs.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!matrixType || !rhsType || !outputType || matrixType.getRank() != 2 ||
      rhsType.getRank() != 1 || outputType.getRank() != 1 ||
      !matrixType.getElementType().isF64() ||
      !rhsType.getElementType().isF64() ||
      !outputType.getElementType().isF64())
    return launch.emitError("row-major DTRSV requires f64 A[N,N], b[N], x[N]");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cublas_dtrsv_lower_row_major",
      {b.getI32Type(), ptr, ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, output, 0),
                 memrefDataPtr(b, loc, matrix), memrefDataPtr(b, loc, rhs),
                 memrefDataPtr(b, loc, output)});
  launch.erase();
  return success();
}

static LogicalResult lowerDpotrfLowerRowMajor(LaunchOp launch,
                                               ModuleOp module) {
  if (launch.getNumOperands() != 1 || launch.getNumResults() != 0)
    return launch.emitError("row-major DPOTRF expects one matrix buffer");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value matrix =
      valueToOutputMemrefPreservingSlice(b, loc, launch.getOperand(0));
  auto matrixType = dyn_cast<MemRefType>(matrix.getType());
  if (!matrixType || matrixType.getRank() != 2 ||
      !matrixType.getElementType().isF64())
    return launch.emitError("row-major DPOTRF requires f64 A[N,N]");
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(module,
                             "polygeist_cusolver_dpotrf_lower_row_major",
                             {b.getI32Type(), ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, matrix, 0),
                 memrefDataPtr(b, loc, matrix)});
  launch.erase();
  return success();
}

static LogicalResult lowerCubPredicateReduction(
    LaunchOp launch, ModuleOp module, StringRef libSym) {
  bool bufferized = launch->hasAttr("polygeist.bufferized");
  unsigned inputCount = libSym == "cubEqualAll1D_f32_tensor" ? 2 : 1;
  if (launch.getNumOperands() != inputCount + 1 ||
      launch.getNumResults() != (bufferized ? 0u : 1u))
    return launch.emitError("CUB predicate reduction operand/result mismatch");
  if (bufferized) {
    auto destinations = launch->getAttrOfType<DenseI64ArrayAttr>(
        "polygeist.result_destinations");
    if (!destinations || destinations.size() != 1 ||
        destinations[0] != static_cast<int64_t>(inputCount))
      return launch.emitError("CUB predicate reduction destination mismatch");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  SmallVector<Value> inputs;
  for (unsigned i = 0; i < inputCount; ++i)
    inputs.push_back(valueToMemrefPreservingSlice(
        b, loc, launch.getOperand(i)));
  Value output = valueToOutputMemrefPreservingSlice(
      b, loc, launch.getOperand(inputCount));
  auto outputType = dyn_cast<MemRefType>(output.getType());
  unsigned inputRank =
      libSym == "cubSegmentedCountNonzero2D_f32_tensor" ? 2 : 1;
  unsigned outputRank = inputRank == 2 ? 1 : 0;
  if (!outputType || outputType.getRank() != outputRank ||
      !outputType.getElementType().isInteger(32))
    return launch.emitError("CUB predicate reduction requires i32 output");
  for (Value input : inputs) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || type.getRank() != inputRank ||
        !type.getElementType().isF32())
      return launch.emitError("CUB predicate reduction requires f32 inputs");
  }
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args;
  SmallVector<Type> types;
  if (inputRank == 2) {
    args.push_back(memrefDimAsI32(b, loc, inputs[0], 0));
    args.push_back(memrefDimAsI32(b, loc, inputs[0], 1));
    types.append(2, b.getI32Type());
  } else {
    args.push_back(memrefDimAsI32(b, loc, inputs[0], 0));
    types.push_back(b.getI32Type());
  }
  for (Value input : inputs) {
    args.push_back(memrefDataPtr(b, loc, input));
    types.push_back(ptr);
  }
  args.push_back(memrefDataPtr(b, loc, output));
  types.push_back(ptr);
  StringRef shim = libSym == "cubCountNonzero1D_f32_tensor"
      ? "polygeist_cub_count_nonzero1d_f32"
      : libSym == "cubSegmentedCountNonzero2D_f32_tensor"
          ? "polygeist_cub_segmented_count_nonzero2d_f32"
          : "polygeist_cub_equal_all1d_f32";
  auto declaration = ensureShimDecl(module, shim, types, b);
  b.create<func::CallOp>(loc, declaration, args);
  if (!bufferized) {
    Value updated = memrefToTensor(
        b, loc, output, launch.getResult(0).getType());
    rewireTensorSliceLaunchResult(
        launch, updated,
        tensorForOutputSliceSource(b, loc, launch.getOperand(inputCount)));
  }
  launch.erase();
  return success();
}

// Runtime computes:
//   out[i] = x[i] * rsqrt(sum_j x[j]^2 / N + 1e-5)
static LogicalResult lowerCublasDot(LaunchOp launch, ModuleOp module,
                                    bool singlePrecision) {
  StringRef abi = singlePrecision ? "cublasSdot" : "cublasDdot";
  if (launch.getNumOperands() != 3)
    return launch.emitError() << abi << ": expected 3 operands (x, y, out)";
  if (launch.getNumResults() != 1)
    return launch.emitError() << abi << ": expected one result";

  Value x = resolveSubmapBase(launch.getOperand(0));
  Value y = resolveSubmapBase(launch.getOperand(1));
  Value out = resolveSubmapBase(launch.getOperand(2));

  ShapedType xTy = getRankedShapedType(x);
  ShapedType yTy = getRankedShapedType(y);
  ShapedType oTy = getRankedShapedType(out);
  if (!xTy || !yTy || !oTy || xTy.getRank() != 1 || yTy.getRank() != 1 ||
      oTy.getRank() != 0)
    return launch.emitError() << abi << ": x/y must be 1D and out rank-0";
  Type expectedType;
  if (singlePrecision)
    expectedType = Float32Type::get(launch.getContext());
  else
    expectedType = Float64Type::get(launch.getContext());
  if (xTy.getElementType() != expectedType ||
      yTy.getElementType() != xTy.getElementType() ||
      oTy.getElementType() != xTy.getElementType())
    return launch.emitError() << abi << ": operands must be "
                              << (singlePrecision ? "f32" : "f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value yMr = valueToMemrefPreservingSlice(b, loc, y);
  Value oMr;
  Value strippedOut = stripTensorCasts(out);
  if (auto slice = strippedOut.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (auto toTensor = destinationToTensorOp(slice.getSource())) {
      auto sourceType = cast<MemRefType>(toTensor.getMemref().getType());
      auto resultType = cast<MemRefType>(
          memref::SubViewOp::inferRankReducedResultType(
              slice.getType().getShape(), sourceType,
              slice.getMixedOffsets(), slice.getMixedSizes(),
              slice.getMixedStrides()));
      oMr = b.create<memref::SubViewOp>(
          loc, resultType, toTensor.getMemref(), slice.getMixedOffsets(),
          slice.getMixedSizes(), slice.getMixedStrides());
    }
  }
  if (!oMr)
    oMr = valueToOutputMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value yPtr = memrefBasePtr(b, loc, yMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, singlePrecision ? "polygeist_cublas_dot_f32"
                              : "polygeist_cublas_dot_f64",
      argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, yPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCublasDotMemref(LaunchOp launch, ModuleOp module,
                                          bool singlePrecision) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("bufferized dot expects x, y, and output");
  for (Value operand : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != 1 ||
        (singlePrecision ? !type.getElementType().isF32()
                         : !type.getElementType().isF64()))
      return launch.emitError(
          "bufferized dot requires rank-1 memrefs of matching precision");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(module, singlePrecision
                                         ? "polygeist_cublas_dot_f32"
                                         : "polygeist_cublas_dot_f64",
                             {b.getI32Type(), ptr, ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{memrefDimAsI32(b, loc, launch.getOperand(0), 0),
                 memrefDataPtr(b, loc, launch.getOperand(0)),
                 memrefDataPtr(b, loc, launch.getOperand(1)),
                 memrefDataPtr(b, loc, launch.getOperand(2))});
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedArgReduceF32(
    LaunchOp launch, ModuleOp module, bool isMin) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 0)
    return launch.emitError("segmented arg-reduction expects input and output");
  auto inputType = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto outputType = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  if (!inputType || !outputType || inputType.getRank() != 2 ||
      outputType.getRank() != 1 || !inputType.getElementType().isF32() ||
      !outputType.getElementType().isInteger(32))
    return launch.emitError("segmented arg-reduction requires 2D f32 to 1D i32");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cub_segmented_argreduce_f32",
      {b.getI32Type(), b.getI32Type(), b.getI32Type(), ptr, ptr}, b);
  Value op = b.create<arith::ConstantIntOp>(loc, isMin ? 1 : 0, 32);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{op, memrefDimAsI32(b, loc, launch.getOperand(0), 0),
                 memrefDimAsI32(b, loc, launch.getOperand(0), 1),
                 memrefDataPtr(b, loc, launch.getOperand(0)),
                 memrefDataPtr(b, loc, launch.getOperand(1))});
  launch.erase();
  return success();
}

static LogicalResult lowerCublasSgemvTZeroMemref(
    LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("bufferized SgemvT expects matrix, vector, output");
  auto matrixType = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto vectorType = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  auto outputType = dyn_cast<MemRefType>(launch.getOperand(2).getType());
  if (!matrixType || !vectorType || !outputType || matrixType.getRank() != 2 ||
      vectorType.getRank() != 1 || outputType.getRank() != 1 ||
      !matrixType.getElementType().isF32() ||
      !vectorType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return launch.emitError("bufferized SgemvT requires f32 matrix/vector/output");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value rows = memrefDimAsI32(b, loc, launch.getOperand(0), 0);
  Value cols = memrefDimAsI32(b, loc, launch.getOperand(0), 1);
  if (auto fixed = launch->getAttrOfType<DenseI64ArrayAttr>(
          "polygeist.fixed_extents")) {
    if (fixed.size() != 2 || fixed[0] <= 0 || fixed[1] <= 0)
      return launch.emitError(
          "SgemvT fixed extents must be two positive values");
    rows = b.create<arith::ConstantIntOp>(loc, fixed[0], 32);
    cols = b.create<arith::ConstantIntOp>(loc, fixed[1], 32);
  }
  Value one = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                          b.getF32FloatAttr(1.0f));
  Value zero = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(0.0f));
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cublas_sgemv_T",
      {b.getI32Type(), b.getI32Type(), b.getF32Type(), ptr, b.getI32Type(),
       ptr, b.getF32Type(), ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{rows, cols, one,
                 memrefDataPtr(b, loc, launch.getOperand(0)), cols,
                 memrefDataPtr(b, loc, launch.getOperand(1)), zero,
                 memrefDataPtr(b, loc, launch.getOperand(2))});
  launch.erase();
  return success();
}

static LogicalResult lowerCudnnSpecialGraph(LaunchOp launch, ModuleOp module) {
  constexpr unsigned expected = 2;
  if (launch.getNumOperands() != expected || launch.getNumResults() != 0)
    return launch.emitError("special graph operand mismatch");
  for (Value operand : launch.getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32())
      return launch.emitError("special graph requires rank-1 f32 memrefs");
  }
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type()};
  types.append(expected, ptr);
  auto shim = ensureShimDecl(module, "polygeist_cudnn_sinc_f32", types, b);
  SmallVector<Value> args = {
      memrefDimAsI32(b, loc, launch.getOperand(0), 0)};
  for (Value operand : launch.getOperands())
    args.push_back(memrefDataPtr(b, loc, operand));
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedSort(LaunchOp launch,ModuleOp module,
                                           bool topk){
  if(launch.getNumOperands()!=3||launch.getNumResults()!=0)return launch.emitError("segmented sort expects input, values, indices");
  for(unsigned i=0;i<3;++i){auto type=dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if(!type||type.getRank()!=2||
       (i<2?!type.getElementType().isF32():!type.getElementType().isInteger(32)))
      return launch.emitError("segmented sort has invalid operand types");}
  OpBuilder b(launch);Location loc=launch.getLoc();auto ptr=LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{memrefDimAsI32(b,loc,launch.getOperand(0),0),memrefDimAsI32(b,loc,launch.getOperand(0),1),
    topk?memrefDimAsI32(b,loc,launch.getOperand(1),1):memrefDimAsI32(b,loc,launch.getOperand(0),1)};
  for(Value value:launch.getOperands())args.push_back(memrefDataPtr(b,loc,value));
  auto shim=ensureShimDecl(module,"polygeist_cub_segmented_sort_descending_f32_i32",{b.getI32Type(),b.getI32Type(),b.getI32Type(),ptr,ptr,ptr},b);
  b.create<func::CallOp>(loc,shim,args);launch.erase();return success();
}

static LogicalResult lowerSegmentReduceLengths(LaunchOp launch,
                                                ModuleOp module) {
  constexpr unsigned expected = 4;
  if(launch.getNumOperands()!=expected||launch.getNumResults()!=0)return launch.emitError("invalid length-segmented reduction operands");
  constexpr unsigned reduceIndex = 2;
  if(!launch.getOperand(reduceIndex).getType().isInteger(32))return launch.emitError("reduction mode must be i32");
  OpBuilder b(launch);Location loc=launch.getLoc();auto ptr=LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{memrefDimAsI32(b,loc,launch.getOperand(0),0),memrefDimAsI32(b,loc,launch.getOperand(1),0),launch.getOperand(reduceIndex)};
  for(unsigned i=0;i<expected;++i)if(i!=reduceIndex)args.push_back(memrefDataPtr(b,loc,launch.getOperand(i)));
  SmallVector<Type> types{b.getI32Type(),b.getI32Type(),b.getI32Type()};types.append(expected-1,ptr);
  auto shim=ensureShimDecl(module,"polygeist_cub_segment_reduce_lengths_f32",types,b);
  b.create<func::CallOp>(loc,shim,args);launch.erase();return success();
}

static std::optional<int32_t> cudnnReductionOpId(StringRef libSym) {
  if (libSym == "cudnnReduceSum_f32" || libSym == "cudnnReduceSum_f64")
    return 0;
  if (libSym == "cudnnReduceProduct_f32") return 1;
  if (libSym == "cudnnReduceMin_f32") return 2;
  if (libSym == "cudnnReduceMax_f32") return 3;
  return std::nullopt;
}

static LogicalResult lowerCudnnReduction(LaunchOp launch, ModuleOp module,
                                         StringRef libSym) {
  bool minmax = libSym == "cudnnReduceMinMax_f32";
  bool diagonal = libSym == "cudnnReduceTrace_f32";
  unsigned expectedOutputs = minmax ? 2 : 1;
  if (launch.getNumOperands() != 1 + expectedOutputs ||
      launch.getNumResults() != expectedOutputs)
    return launch.emitError(
        "cudnn reduction: expected one input plus one destination per result");

  Value input = launch.getOperand(0);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy || inputTy.getRank() != (diagonal ? 2 : 1))
    return launch.emitError(
        "cudnn reduction: input rank does not match the selected route");
  bool isF32 = inputTy.getElementType().isF32();
  bool isF64 = inputTy.getElementType().isF64();
  if (!isF32 && !isF64)
    return launch.emitError("cudnn reduction: only f32/f64 are supported");
  if (minmax && !isF32)
    return launch.emitError("cudnn minmax reduction requires f32");
  for (unsigned i = 0; i < expectedOutputs; ++i) {
    auto outTy = dyn_cast<RankedTensorType>(
        launch.getOperand(1 + i).getType());
    if (!outTy || outTy.getRank() != 0 ||
        outTy.getElementType() != inputTy.getElementType())
      return launch.emitError(
          "cudnn reduction: destinations must be scalar tensors of the input type");
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value inputPtr = memrefDataPtr(b, loc, inputMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes;
  StringRef shimName;
  Value n;
  SmallVector<Value> diagonalArgs;
  if (diagonal) {
    Value rows = memrefDimAsI32(b, loc, inputMr, 0);
    Value cols = memrefDimAsI32(b, loc, inputMr, 1);
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, inputMr);
    Value rowStride = valueAsI32(b, loc, metadata.getStrides()[0]);
    Value colStride = valueAsI32(b, loc, metadata.getStrides()[1]);
    argTypes = {b.getI32Type(), b.getI32Type(), b.getI32Type(),
                b.getI32Type(), ptrTy, ptrTy};
    shimName = "polygeist_cudnn_reduce_diagonal_f32";
    diagonalArgs = {rows, cols, rowStride, colStride, inputPtr};
  } else {
    n = memrefNumElementsAsI32(b, loc, inputMr);
    argTypes = {b.getI32Type(), b.getI32Type(), ptrTy, ptrTy};
    shimName = isF32 ? "polygeist_cudnn_reduce_f32"
                     : "polygeist_cudnn_reduce_f64";
  }
  func::FuncOp shim = ensureShimDecl(module, shimName, argTypes, b);

  SmallVector<Value> updated;
  for (unsigned i = 0; i < expectedOutputs; ++i) {
    int32_t opId = diagonal ? 0 : (minmax ? (i == 0 ? 3 : 2)
                                          : *cudnnReductionOpId(libSym));
    Value op = b.create<arith::ConstantOp>(
        loc, b.getI32Type(), b.getI32IntegerAttr(opId));
    Value out = launch.getOperand(1 + i);
    Value outMr = valueToOutputMemrefPreservingSlice(b, loc, out);
    Value outPtr = memrefDataPtr(b, loc, outMr);
    if (diagonal) {
      SmallVector<Value> args(diagonalArgs);
      args.push_back(outPtr);
      b.create<func::CallOp>(loc, shim, args);
    } else {
      b.create<func::CallOp>(loc, shim, ValueRange{op, n, inputPtr, outPtr});
    }
    updated.push_back(
        memrefToTensor(b, loc, outMr, launch.getResult(i).getType()));
  }

  if (!minmax) {
    rewireTensorSliceLaunchResult(
        launch, updated[0],
        tensorForOutputSliceSource(b, loc, launch.getOperand(1)));
  } else {
    for (unsigned i = 0; i < expectedOutputs; ++i)
      launch.getResult(i).replaceAllUsesWith(updated[i]);
  }
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedI32(LaunchOp launch, ModuleOp module,
                                           StringRef libSym) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 1)
    return launch.emitError(
        "CUB segmented reduction expects (input, output) and one result");
  Value input = launch.getOperand(0);
  Value output = launch.getOperand(1);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  if (!inputTy || !outputTy || inputTy.getRank() != 2 ||
      outputTy.getRank() != 1 || !inputTy.getElementType().isInteger(32) ||
      !outputTy.getElementType().isInteger(32))
    return launch.emitError(
        "CUB segmented reduction requires rank-2 i32 input and rank-1 i32 output");

  int32_t opId = libSym == "cubSegmentedLogicalAnd_i32" ? 0
                 : libSym == "cubSegmentedLogicalOr_i32" ? 1 : 2;
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value outputMr = valueToOutputMemrefPreservingSlice(b, loc, output);
  Value rows = memrefDimAsI32(b, loc, inputMr, 0);
  Value cols = memrefDimAsI32(b, loc, inputMr, 1);
  Value op = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(opId));
  Value inputPtr = memrefDataPtr(b, loc, inputMr);
  Value outputPtr = memrefDataPtr(b, loc, outputMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cub_segmented_reduce_i32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim, ValueRange{op, rows, cols, inputPtr, outputPtr});
  Value updated =
      memrefToTensor(b, loc, outputMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedLogicalSelectI32(
    LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 4 || launch.getNumResults() != 1)
    return launch.emitError(
        "dynamic CUB logical reduction expects two inputs, flag, and output");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value allInput = valueToMemrefPreservingSlice(
      b, loc, launch.getOperand(0));
  Value anyInput = valueToMemrefPreservingSlice(
      b, loc, launch.getOperand(1));
  Value output = valueToOutputMemrefPreservingSlice(
      b, loc, launch.getOperand(3));
  auto allType = dyn_cast<MemRefType>(allInput.getType());
  auto anyType = dyn_cast<MemRefType>(anyInput.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!allType || !anyType || !outputType || allType.getRank() != 2 ||
      anyType.getRank() != 2 || outputType.getRank() != 1 ||
      !allType.getElementType().isInteger(32) ||
      !anyType.getElementType().isInteger(32) ||
      !outputType.getElementType().isInteger(32) ||
      !launch.getOperand(2).getType().isInteger(1))
    return launch.emitError("dynamic CUB logical reduction type mismatch");
  Value flag = launch.getOperand(2);
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  Value op = b.create<arith::SelectOp>(loc, flag, zero, one);
  Value allPtr = memrefDataPtr(b, loc, allInput);
  Value anyPtr = memrefDataPtr(b, loc, anyInput);
  Value inputPtr = b.create<arith::SelectOp>(loc, flag, allPtr, anyPtr);
  Value outputPtr = memrefDataPtr(b, loc, output);
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), ptr, ptr};
  auto shim = ensureShimDecl(
      module, "polygeist_cub_segmented_reduce_i32", types, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{op, memrefDimAsI32(b, loc, allInput, 0),
                 memrefDimAsI32(b, loc, allInput, 1), inputPtr, outputPtr});
  Value updated = memrefToTensor(
      b, loc, output, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated,
      tensorForOutputSliceSource(b, loc, launch.getOperand(3)));
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedLogicalMemrefI32(
    LaunchOp launch, ModuleOp module, bool dynamicSelect) {
  unsigned expected = dynamicSelect ? 4 : 2;
  if (launch.getNumOperands() != expected || launch.getNumResults() != 0)
    return launch.emitError("bufferized segmented logical operand mismatch");
  unsigned outputIndex = dynamicSelect ? 3 : 1;
  Value input = launch.getOperand(0);
  Value output = launch.getOperand(outputIndex);
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != 2 ||
      outputType.getRank() != 1 ||
      !inputType.getElementType().isInteger(32) ||
      !outputType.getElementType().isInteger(32))
    return launch.emitError("bufferized segmented logical type mismatch");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value selectedInput = input;
  Value op = b.create<arith::ConstantIntOp>(loc, 0, 32);
  if (dynamicSelect) {
    auto secondType = dyn_cast<MemRefType>(launch.getOperand(1).getType());
    if (!secondType || secondType.getRank() != 2 ||
        !secondType.getElementType().isInteger(32) ||
        !launch.getOperand(2).getType().isInteger(32))
      return launch.emitError("bufferized dynamic logical type mismatch");
    Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value all = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                       launch.getOperand(2), zero);
    Value anyOp = b.create<arith::ConstantIntOp>(loc, 1, 32);
    op = b.create<arith::SelectOp>(loc, all, op, anyOp);
    selectedInput = b.create<arith::SelectOp>(
        loc, all, input, launch.getOperand(1));
  }
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto shim = ensureShimDecl(
      module, "polygeist_cub_segmented_reduce_i32",
      {b.getI32Type(), b.getI32Type(), b.getI32Type(), ptr, ptr}, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{op, memrefDimAsI32(b, loc, selectedInput, 0),
                 memrefDimAsI32(b, loc, selectedInput, 1),
                 memrefDataPtr(b, loc, selectedInput),
                 memrefDataPtr(b, loc, output)});
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedPrefix(LaunchOp launch, ModuleOp module,
                                              StringRef libSym) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "CUB prefix reduction expects (input, lengths, output) and one result");
  Value input = launch.getOperand(0);
  Value lengths = launch.getOperand(1);
  Value output = launch.getOperand(2);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto lengthsTy = dyn_cast<RankedTensorType>(lengths.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  bool isSum = libSym == "cubSegmentedPrefixSum_f32";
  Type elementType = isSum ? Type(Float32Type::get(launch.getContext()))
                           : Type(IntegerType::get(launch.getContext(), 32));
  if (!inputTy || !lengthsTy || !outputTy || inputTy.getRank() != 2 ||
      lengthsTy.getRank() != 1 || outputTy.getRank() != 1 ||
      inputTy.getElementType() != elementType ||
      !lengthsTy.getElementType().isInteger(32) ||
      outputTy.getElementType() != elementType)
    return launch.emitError(
        "CUB prefix reduction requires rank-2 data, rank-1 i32 lengths, "
        "and a rank-1 output of the data element type");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value lengthsMr = valueToMemrefPreservingSlice(b, loc, lengths);
  Value outputMr = valueToOutputMemrefPreservingSlice(b, loc, output);
  Value rows = memrefDimAsI32(b, loc, inputMr, 0);
  Value cols = memrefDimAsI32(b, loc, inputMr, 1);
  Value inputPtr = memrefDataPtr(b, loc, inputMr);
  Value lengthsPtr = memrefDataPtr(b, loc, lengthsMr);
  Value outputPtr = memrefDataPtr(b, loc, outputMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), ptrTy, ptrTy, ptrTy};
  StringRef shimName = isSum
      ? "polygeist_cub_segmented_prefix_sum_f32"
      : "polygeist_cub_segmented_prefix_logical_and_i32";
  func::FuncOp shim = ensureShimDecl(module, shimName, argTypes, b);
  b.create<func::CallOp>(
      loc, shim, ValueRange{rows, cols, inputPtr, lengthsPtr, outputPtr});
  Value updated =
      memrefToTensor(b, loc, outputMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, output));
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedPrefixMemref(LaunchOp launch,
                                                    ModuleOp module,
                                                    bool isSum) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 0)
    return launch.emitError("bufferized prefix reduction expects data, lengths, output");
  auto inputTy = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto lengthsTy = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  auto outputTy = dyn_cast<MemRefType>(launch.getOperand(2).getType());
  Type element = isSum ? Type(Float32Type::get(launch.getContext()))
                       : Type(IntegerType::get(launch.getContext(), 32));
  if (!inputTy || inputTy.getRank() != 2 ||
      inputTy.getElementType() != element || !lengthsTy ||
      lengthsTy.getRank() != 1 ||
      !lengthsTy.getElementType().isInteger(32) || !outputTy ||
      outputTy.getRank() != 1 || outputTy.getElementType() != element)
    return launch.emitError("invalid bufferized prefix reduction operand types");
  OpBuilder b(launch); Location loc = launch.getLoc();
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Value> args{
      memrefDimAsI32(b, loc, launch.getOperand(0), 0),
      memrefDimAsI32(b, loc, launch.getOperand(0), 1),
      memrefDataPtr(b, loc, launch.getOperand(0)),
      memrefDataPtr(b, loc, launch.getOperand(1)),
      memrefDataPtr(b, loc, launch.getOperand(2))};
  SmallVector<Type> types{b.getI32Type(), b.getI32Type(), ptr, ptr, ptr};
  auto shim = ensureShimDecl(
      module, isSum ? "polygeist_cub_segmented_prefix_sum_f32"
                    : "polygeist_cub_segmented_prefix_logical_and_i32",
      types, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCubSegmentedFullMemref(LaunchOp launch,
                                                 ModuleOp module,
                                                 StringRef libSym) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 0)
    return launch.emitError("bufferized segmented reduction expects input, output");
  bool isI32 = libSym == "cubSegmentedBitXor_i32_memref";
  bool isF64 = libSym == "cubSegmentedSum_f64_memref";
  auto inputTy = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto outputTy = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  if (!inputTy || inputTy.getRank() != 2 || !outputTy ||
      outputTy.getRank() != 1 ||
      (isI32 ? (!inputTy.getElementType().isInteger(32) ||
                !outputTy.getElementType().isInteger(32))
       : isF64 ? (!inputTy.getElementType().isF64() ||
                  !outputTy.getElementType().isF64())
               : (!inputTy.getElementType().isF32() ||
                  !outputTy.getElementType().isF32())))
    return launch.emitError("invalid bufferized segmented reduction types");
  int32_t opId = isI32 ? 2
      : (libSym == "cubSegmentedSum_f32_memref" || isF64) ? 0
      : libSym == "cubSegmentedMin_f32_memref" ? 1 : 2;
  OpBuilder b(launch); Location loc = launch.getLoc();
  Value rows = memrefDimAsI32(b, loc, launch.getOperand(0), 0);
  Value cols = memrefDimAsI32(b, loc, launch.getOperand(0), 1);
  if (auto fixed = launch->getAttrOfType<DenseI64ArrayAttr>(
          "polygeist.fixed_extents")) {
    if (fixed.size() != 2 || fixed[0] <= 0 || fixed[1] <= 0)
      return launch.emitError(
          "segmented reduction fixed extents must be two positive values");
    rows = b.create<arith::ConstantIntOp>(loc, fixed[0], 32);
    cols = b.create<arith::ConstantIntOp>(loc, fixed[1], 32);
  }
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  Value op = b.create<arith::ConstantIntOp>(loc, opId, 32);
  SmallVector<Value> args{
      op, rows, cols,
      memrefDataPtr(b, loc, launch.getOperand(0)),
      memrefDataPtr(b, loc, launch.getOperand(1))};
  SmallVector<Type> types{b.getI32Type(), b.getI32Type(), b.getI32Type(),
                          ptr, ptr};
  auto shim = ensureShimDecl(
      module, isI32 ? "polygeist_cub_segmented_reduce_i32"
             : isF64 ? "polygeist_cub_segmented_reduce_f64"
                     : "polygeist_cub_segmented_reduce_f32",
      types, b);
  b.create<func::CallOp>(loc, shim, args);
  launch.erase();
  return success();
}

static LogicalResult lowerCutensorPermuteF32(
    LaunchOp launch, ModuleOp module, StringRef libSym) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 1)
    return launch.emitError("cuTENSOR permutation expects input, destination, result");
  Value input = launch.getOperand(0), output = launch.getOperand(1);
  Operation *inputView = stripTensorCasts(input).getDefiningOp();
  Operation *outputView = stripTensorCasts(output).getDefiningOp();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != outputType.getRank() ||
      inputType.getRank() < 2 || inputType.getRank() > 6 ||
      !inputType.getElementType().isF32() || !outputType.getElementType().isF32())
    return launch.emitError("cuTENSOR permutation requires equal-rank 2D-6D f32 tensors");
  unsigned rank = inputType.getRank();
  auto inputModes = launch->getAttrOfType<DenseI64ArrayAttr>(
      "cutensor_input_modes");
  auto outputModes = launch->getAttrOfType<DenseI64ArrayAttr>(
      "cutensor_output_modes");
  if (!inputModes || !outputModes || inputModes.size() != rank ||
      outputModes.size() != rank)
    return launch.emitError("cuTENSOR permutation mode arrays are missing");
  SmallVector<int64_t> sortedInput(inputModes.asArrayRef());
  SmallVector<int64_t> sortedOutput(outputModes.asArrayRef());
  llvm::sort(sortedInput); llvm::sort(sortedOutput);
  if (sortedInput != sortedOutput)
    return launch.emitError("cuTENSOR input/output modes do not match");

  OpBuilder b(launch); Location loc = launch.getLoc();
  SmallVector<AffineExpr> inputModeExprs, outputModeExprs;
  for (int64_t mode : inputModes.asArrayRef())
    inputModeExprs.push_back(getAffineDimExpr(mode, b.getContext()));
  for (int64_t mode : outputModes.asArrayRef())
    outputModeExprs.push_back(getAffineDimExpr(mode, b.getContext()));
  auto inputMap = AffineMap::get(rank, 0, inputModeExprs, b.getContext());
  auto outputMap = AffineMap::get(rank, 0, outputModeExprs, b.getContext());
  auto inputMetadata = buildContractionViewMetadata(b, loc, input, inputMap);
  auto outputMetadata = buildContractionViewMetadata(b, loc, output, outputMap);
  if (failed(inputMetadata) || failed(outputMetadata) ||
      inputMetadata->extents.size() != rank ||
      outputMetadata->extents.size() != rank)
    return launch.emitError("cuTENSOR permutation cannot recover view strides");
  Value inputMemref = valueToMemref(b, loc, inputMetadata->base);
  Value outputMemref = valueToMemref(b, loc, outputMetadata->base);
  auto i64ArrayType = MemRefType::get({static_cast<int64_t>(rank)}, b.getI64Type());
  auto i32ArrayType = MemRefType::get({static_cast<int64_t>(rank)}, b.getI32Type());
  Value inputExtents = b.create<memref::AllocaOp>(loc, i64ArrayType);
  Value inputStrides = b.create<memref::AllocaOp>(loc, i64ArrayType);
  Value inputModeArray = b.create<memref::AllocaOp>(loc, i32ArrayType);
  Value outputExtents = b.create<memref::AllocaOp>(loc, i64ArrayType);
  Value outputStrides = b.create<memref::AllocaOp>(loc, i64ArrayType);
  Value outputModeArray = b.create<memref::AllocaOp>(loc, i32ArrayType);
  for (unsigned d = 0; d < rank; ++d) {
    Value index = b.create<arith::ConstantIndexOp>(loc, d);
    Value inputExtent = inputMetadata->extents[d];
    Value outputExtent = outputMetadata->extents[d];
    Value inputStride = inputMetadata->strides[d];
    Value outputStride = outputMetadata->strides[d];
    Value inputMode = b.create<arith::ConstantIntOp>(loc, inputModes[d], 32);
    Value outputMode = b.create<arith::ConstantIntOp>(loc, outputModes[d], 32);
    b.create<memref::StoreOp>(loc, inputExtent, inputExtents, index);
    b.create<memref::StoreOp>(loc, inputStride, inputStrides, index);
    b.create<memref::StoreOp>(loc, inputMode, inputModeArray, index);
    b.create<memref::StoreOp>(loc, outputExtent, outputExtents, index);
    b.create<memref::StoreOp>(loc, outputStride, outputStrides, index);
    b.create<memref::StoreOp>(loc, outputMode, outputModeArray, index);
  }
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> types = {b.getI32Type()}; types.append(8, ptr);
  auto shim = ensureShimDecl(module, "polygeist_cutensor_permute_f32", types, b);
  Value rankValue = b.create<arith::ConstantIntOp>(loc, rank, 32);
  auto pointerWithOffset = [&](Value memref, Value elementOffset) {
    Value pointer = memrefBasePtr(b, loc, memref);
    llvm::APInt staticOffset;
    if (!matchPattern(elementOffset, m_ConstantInt(&staticOffset)) ||
        !staticOffset.isZero()) {
      Value address = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), pointer);
      Value bytes = b.create<arith::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(sizeof(float)));
      address = b.create<arith::AddIOp>(
          loc, address, b.create<arith::MulIOp>(loc, elementOffset, bytes));
      pointer = b.create<LLVM::IntToPtrOp>(loc, ptr, address);
    }
    return pointer;
  };
  Value inputPointer = pointerWithOffset(inputMemref,
                                         inputMetadata->elementOffset);
  Value outputPointer = pointerWithOffset(outputMemref,
                                          outputMetadata->elementOffset);
  b.create<func::CallOp>(loc, shim, ValueRange{
      rankValue, memrefDataPtr(b, loc, inputExtents),
      memrefDataPtr(b, loc, inputStrides), memrefDataPtr(b, loc, inputModeArray),
      memrefDataPtr(b, loc, outputExtents), memrefDataPtr(b, loc, outputStrides),
      memrefDataPtr(b, loc, outputModeArray), inputPointer, outputPointer});
  Value updatedBase = memrefToTensor(
      b, loc, outputMemref, outputMetadata->base.getType());
  Value strippedOutput = stripTensorCasts(output);
  if (auto submap = strippedOutput.getDefiningOp<polygeist::SubmapOp>()) {
    SmallVector<Value> indicesAndSizes(submap.getOperands().drop_front());
    Value updatedView = b.create<polygeist::SubmapOp>(
        loc, output.getType(), updatedBase, indicesAndSizes, submap.getMap());
    if (failed(rewireSubmapLaunchResult(launch, updatedView, updatedBase)))
      return failure();
  } else {
    Value updatedView = updatedBase;
    if (updatedView.getType() != launch.getResult(0).getType())
      updatedView = b.create<tensor::CastOp>(
          loc, launch.getResult(0).getType(), updatedView);
    rewireTensorSliceLaunchResult(
        launch, updatedView, tensorForOutputSliceSource(b, loc, output));
    if (!launch.getResult(0).use_empty())
      launch.getResult(0).replaceAllUsesWith(updatedView);
  }
  launch.erase();
  SmallVector<Operation *> worklist;
  if (inputView) worklist.push_back(inputView);
  if (outputView && outputView != inputView) worklist.push_back(outputView);
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    if (!candidate->getBlock() || !isOpTriviallyDead(candidate)) continue;
    for (Value operand : candidate->getOperands())
      if (Operation *def = operand.getDefiningOp()) worklist.push_back(def);
    candidate->erase();
  }
  return success();
}

static LogicalResult lowerCutensorUnaryF32(LaunchOp launch, ModuleOp module,
                                           StringRef libSym) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 1)
    return launch.emitError(
        "cutensorUnary: expected (input, output) and one tensor result");
  auto opId = cutensorUnaryOpId(libSym);
  if (!opId)
    return launch.emitError("cutensorUnary: unknown operation symbol ")
           << libSym;
  Value input = launch.getOperand(0);
  Value output = launch.getOperand(1);
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto outputTy = dyn_cast<RankedTensorType>(output.getType());
  if (!inputTy || !outputTy || inputTy.getRank() < 1 ||
      inputTy.getRank() != outputTy.getRank() ||
      !inputTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32() ||
      inputTy.getShape() != outputTy.getShape())
    return launch.emitError(
        "cutensorUnary: input/output must be same-shaped ranked f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value inputMr = valueToMemrefPreservingSlice(b, loc, input);
  Value outputMr = valueToMemrefPreservingSlice(b, loc, output);
  Value op = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(*opId));
  Value n = memrefNumElementsAsI32(b, loc, inputMr);
  Value inputPtr = memrefBasePtr(b, loc, inputMr);
  Value outputPtr = memrefBasePtr(b, loc, outputMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cutensor_unary_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{op, n, inputPtr, outputPtr});

  Value updated =
      memrefToTensor(b, loc, outputMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForSliceSource(b, loc, output));
  launch.erase();
  return success();
}

static LogicalResult lowerWhisperExpShiftSumF32(LaunchOp launch,
                                                ModuleOp module) {
  if (launch.getNumOperands() != 4)
    return launch.emitError(
        "whisperExpShiftSum: expected 4 operands (x, out, sum, max)");
  if (launch.getNumResults() != 2)
    return launch.emitError("whisperExpShiftSum: expected two results");

  Value x = launch.getOperand(0);
  Value out = launch.getOperand(1);
  Value sum = launch.getOperand(2);
  Value maxVal = launch.getOperand(3);

  ShapedType xTy = getRankedShapedType(x);
  ShapedType oTy = getRankedShapedType(out);
  ShapedType sTy = getRankedShapedType(sum);
  if (!xTy || !oTy || !sTy || xTy.getRank() != 1 || oTy.getRank() != 1 ||
      sTy.getRank() != 0)
    return launch.emitError(
        "whisperExpShiftSum: x/out must be 1D and sum must be rank-0");
  if (!xTy.getElementType().isF32() ||
      oTy.getElementType() != xTy.getElementType() ||
      sTy.getElementType() != xTy.getElementType() ||
      !maxVal.getType().isF32())
    return launch.emitError("whisperExpShiftSum: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value sMr = valueToMemrefPreservingSlice(b, loc, sum);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);
  Value sPtr = memrefBasePtr(b, loc, sMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), ptrTy, b.getF32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_whisper_exp_shift_sum_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim,
                         ValueRange{N, xPtr, maxVal, oPtr, sPtr});

  Value updatedOut = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updatedOut,
                                tensorForSliceSource(b, loc, out));
  Value updatedSum = memrefToTensor(b, loc, sMr, launch.getResult(1).getType());
  launch.getResult(1).replaceAllUsesWith(updatedSum);
  launch.erase();
  return success();
}

// @cudnnSoftmaxForward(%x), FP32 1D in-place row softmax.
// Tensor form returns the updated tensor after the same in-place shim call.
static LogicalResult lowerCudnnSoftmaxForwardF32(LaunchOp launch,
                                                  ModuleOp module) {
  if (launch.getNumOperands() != 1)
    return launch.emitError("cudnnSoftmaxForward: expected 1 operand");
  if (launch.getNumResults() > 1)
    return launch.emitError(
        "cudnnSoftmaxForward: expected void or one tensor result");

  Value x = resolveSubmapBase(launch.getOperand(0));
  ShapedType xTy = getRankedShapedType(x);
  if (!xTy || xTy.getRank() != 1)
    return launch.emitError("cudnnSoftmaxForward: x must be ranked 1D");
  if (!xTy.getElementType().isF32())
    return launch.emitError("cudnnSoftmaxForward: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemref(b, loc, x);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_softmax_forward_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr});

  if (launch.getNumResults() == 1) {
    Value updated = memrefToTensor(b, loc, xMr, launch.getResult(0).getType());
    rewireLaunchResult(launch, updated);
  }

  launch.erase();
  return success();
}

static LogicalResult lowerCudnnSoftmaxForwardOutF32(LaunchOp launch,
                                                     ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError(
        "cudnnSoftmaxForwardOut: expected 2 operands (scores, out)");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudnnSoftmaxForwardOut: expected one result");

  Value scores = launch.getOperand(0);
  Value out = launch.getOperand(1);
  auto sTy = dyn_cast<RankedTensorType>(scores.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!sTy || !oTy || sTy.getRank() != 1 || oTy.getRank() != 1 ||
      !sTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return launch.emitError(
        "cudnnSoftmaxForwardOut: scores/out must be 1D f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value sMr = valueToMemrefPreservingSlice(b, loc, scores);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, sMr, 0);
  Value sPtr = memrefBasePtr(b, loc, sMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cudnn_softmax_forward_out_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, sPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCudaCopyF32(LaunchOp launch, ModuleOp module,
                                      int expectedRank) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("cudaCopy_f32: expected 2 operands");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudaCopy_f32: expected one result");

  Value src = launch.getOperand(0);
  Value out = launch.getOperand(1);
  auto sTy = dyn_cast<RankedTensorType>(src.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!sTy || !oTy || sTy.getRank() != expectedRank ||
      oTy.getRank() != expectedRank || !sTy.getElementType().isF32() ||
      !oTy.getElementType().isF32())
    return launch.emitError("cudaCopy_f32: operands must be rank-")
           << expectedRank << " f32 tensors";

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value sMr = valueToMemrefPreservingSlice(b, loc, src);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  auto sMd = b.create<memref::ExtractStridedMetadataOp>(loc, sMr);
  auto oMd = b.create<memref::ExtractStridedMetadataOp>(loc, oMr);
  Value rows = memrefDimAsI32(b, loc, sMr, 0);
  Value cols = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(1));
  Value sRowStride = valueAsI32(b, loc, sMd.getStrides()[0]);
  Value oRowStride = valueAsI32(b, loc, oMd.getStrides()[0]);
  Value sColStride = cols;
  Value oColStride = cols;
  if (expectedRank == 2) {
    cols = memrefDimAsI32(b, loc, sMr, 1);
    sColStride = valueAsI32(b, loc, sMd.getStrides()[1]);
    oColStride = valueAsI32(b, loc, oMd.getStrides()[1]);
  }
  Value sPtr = memrefBasePtr(b, loc, sMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(6, b.getI32Type());
  argTypes.append({ptrTy, ptrTy});
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cuda_copy_strided_2d_f32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{rows, cols, sRowStride, sColStride,
                 oRowStride, oColStride, sPtr, oPtr});

  Value updatedBase = tensorForSliceSource(b, loc, out);
  // Preserve an updated slice for direct consumers as well as the base tensor
  // used to bypass a terminal insert_slice.
  Value updated = memrefToTensor(
      b, loc, valueToMemrefPreservingSlice(b, loc, out),
      launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated, updatedBase);
  if (!launch.getResult(0).use_empty())
    launch.getResult(0).replaceAllUsesWith(updated);
  launch.erase();
  return success();
}

static LogicalResult lowerCublasBroadcastF32(LaunchOp launch, ModuleOp module,
                                            int32_t axis) {
  if (launch.getNumOperands() != 2 || launch.getNumResults() != 1)
    return launch.emitError("cuDNN broadcast expects (source, output)");
  Value src = launch.getOperand(0);
  Value out = launch.getOperand(1);
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto outTy = dyn_cast<RankedTensorType>(out.getType());
  if (!srcTy || !outTy || srcTy.getRank() != 1 || outTy.getRank() != 2 ||
      !srcTy.getElementType().isF32() || !outTy.getElementType().isF32())
    return launch.emitError(
        "cuDNN broadcast requires rank-1 f32 source and rank-2 f32 output");
  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value srcMr = valueToMemrefPreservingSlice(b, loc, src);
  Value outMr = valueToOutputMemrefPreservingSlice(b, loc, out);
  Value rows = memrefDimAsI32(b, loc, outMr, 0);
  Value cols = memrefDimAsI32(b, loc, outMr, 1);
  Value axisValue = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(axis));
  Value srcPtr = memrefDataPtr(b, loc, srcMr);
  Value outPtr = memrefDataPtr(b, loc, outMr);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cublas_broadcast_1d_to_2d_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{axisValue, rows, cols, srcPtr, outPtr});
  Value updated = memrefToTensor(b, loc, outMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(
      launch, updated, tensorForOutputSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCudaAddF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cudaAdd_f32: expected 3 operands");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudaAdd_f32: expected one result");

  Value x = launch.getOperand(0);
  Value y = launch.getOperand(1);
  Value out = launch.getOperand(2);
  auto xTy = dyn_cast<RankedTensorType>(x.getType());
  auto yTy = dyn_cast<RankedTensorType>(y.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!xTy || !yTy || !oTy || xTy.getRank() != 1 || yTy.getRank() != 1 ||
      oTy.getRank() != 1 || !xTy.getElementType().isF32() ||
      !yTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return launch.emitError("cudaAdd_f32: operands must be 1D f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value yMr = valueToMemrefPreservingSlice(b, loc, y);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, oMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value yPtr = memrefBasePtr(b, loc, yMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_add_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, yPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCudaMaskSelectF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError(
        "cudaMaskSelect_f32: expected 3 operands (scores, out, pos)");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudaMaskSelect_f32: expected one result");

  Value scores = launch.getOperand(0);
  Value out = launch.getOperand(1);
  Value pos = launch.getOperand(2);
  auto sTy = dyn_cast<RankedTensorType>(scores.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!sTy || !oTy || sTy.getRank() != 1 || oTy.getRank() != 1 ||
      !sTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return launch.emitError(
        "cudaMaskSelect_f32: scores/out must be 1D f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value sMr = valueToMemrefPreservingSlice(b, loc, scores);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, sMr, 0);
  Value posI32 = valueAsI32(b, loc, pos);
  Value sPtr = memrefBasePtr(b, loc, sMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_mask_select_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, posI32, sPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCudaSwiGLUF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cudaSwiGLU_f32: expected 3 operands");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudaSwiGLU_f32: expected one result");

  Value gate = launch.getOperand(0);
  Value up = launch.getOperand(1);
  Value out = launch.getOperand(2);
  auto gTy = dyn_cast<RankedTensorType>(gate.getType());
  auto uTy = dyn_cast<RankedTensorType>(up.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!gTy || !uTy || !oTy || gTy.getRank() != 1 || uTy.getRank() != 1 ||
      oTy.getRank() != 1 || !gTy.getElementType().isF32() ||
      !uTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return launch.emitError("cudaSwiGLU_f32: operands must be 1D f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value gMr = valueToMemrefPreservingSlice(b, loc, gate);
  Value uMr = valueToMemrefPreservingSlice(b, loc, up);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, oMr, 0);
  Value gPtr = memrefBasePtr(b, loc, gMr);
  Value uPtr = memrefBasePtr(b, loc, uMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_swiglu_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, gPtr, uPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCudaRopeMulMulF32(LaunchOp launch, ModuleOp module,
                                            bool add) {
  if (launch.getNumOperands() != 5)
    return launch.emitError("cudaRopeMulMul_f32: expected 5 operands");
  if (launch.getNumResults() != 1)
    return launch.emitError("cudaRopeMulMul_f32: expected one result");

  Value A = launch.getOperand(0);
  Value B = launch.getOperand(1);
  Value C = launch.getOperand(2);
  Value D = launch.getOperand(3);
  Value Out = launch.getOperand(4);
  auto ATy = dyn_cast<RankedTensorType>(A.getType());
  auto BTy = dyn_cast<RankedTensorType>(B.getType());
  auto CTy = dyn_cast<RankedTensorType>(C.getType());
  auto DTy = dyn_cast<RankedTensorType>(D.getType());
  auto OTy = dyn_cast<RankedTensorType>(Out.getType());
  if (!ATy || !BTy || !CTy || !DTy || !OTy || ATy.getRank() != 2 ||
      BTy.getRank() != 1 || CTy.getRank() != 2 || DTy.getRank() != 1 ||
      OTy.getRank() != 2 || !ATy.getElementType().isF32() ||
      !BTy.getElementType().isF32() || !CTy.getElementType().isF32() ||
      !DTy.getElementType().isF32() || !OTy.getElementType().isF32())
    return launch.emitError(
        "cudaRopeMulMul_f32: expected [2D,1D,2D,1D,2D] f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M = dimForTensorOrMemrefAsI32(b, loc, Out, 0);
  Value N = dimForTensorOrMemrefAsI32(b, loc, Out, 1);
  Value addI32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(add ? 1 : 0));

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
                                b.getI32Type()};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_rope_mulmul_f32", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{M, N, pointerForTensorOrMemref(b, loc, A),
                 pointerForTensorOrMemref(b, loc, B),
                 pointerForTensorOrMemref(b, loc, C),
                 pointerForTensorOrMemref(b, loc, D),
                 pointerForTensorOrMemref(b, loc, Out),
                 addI32});

  Value updatedBase = tensorForSliceSource(b, loc, Out);
  Value updated = updatedBase ? Value()
      : memrefToTensor(b, loc, valueToMemrefPreservingSlice(b, loc, Out),
                       launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated, updatedBase);
  launch.erase();
  return success();
}

// @cublasLtMatmulBiasReluFused(%A_view, %B_view, %bias_view, %C_view)
//
// 4 operands. After resolving submap → 4 base tensors:
//   - A:    2D (M, K)
//   - B:    2D (K, N)
//   - bias: 1D (N)  — per-column, broadcast over rows
//   - C:    2D (M, N)
//
// Routes to polygeist_cublaslt_matmul_bias_relu(M, N, K, A*, B*, bias*, C*).
// Runtime issues a single cublasLtMatmul with CUBLASLT_EPILOGUE_RELU_BIAS.
static LogicalResult lowerCublasLtMatmulBiasRelu(LaunchOp launch,
                                                   ModuleOp module) {
  if (launch.getNumOperands() != 4)
    return launch.emitError(
        "cublasLtMatmulBiasReluFused: expected 4 operands, got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError(
        "cublasLtMatmulBiasReluFused: expected 1 result");

  Value Abase   = resolveSubmapBase(launch.getOperand(0));
  Value Bbase   = resolveSubmapBase(launch.getOperand(1));
  Value biasB   = resolveSubmapBase(launch.getOperand(2));
  Value Cbase   = resolveSubmapBase(launch.getOperand(3));

  auto At = dyn_cast<RankedTensorType>(Abase.getType());
  auto Bt = dyn_cast<RankedTensorType>(Bbase.getType());
  auto bT = dyn_cast<RankedTensorType>(biasB.getType());
  auto Ct = dyn_cast<RankedTensorType>(Cbase.getType());
  if (!At || !Bt || !bT || !Ct ||
      At.getRank() != 2 || Bt.getRank() != 2 ||
      bT.getRank() != 1 || Ct.getRank() != 2)
    return launch.emitError(
        "cublasLtMatmulBiasReluFused: expected (A:2D, B:2D, bias:1D, C:2D) "
        "after resolving submap");
  Type elemTy = At.getElementType();
  if (!elemTy.isF32() || Bt.getElementType() != elemTy ||
      bT.getElementType() != elemTy || Ct.getElementType() != elemTy)
    return launch.emitError(
        "cublasLtMatmulBiasReluFused: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr  = tensorToMemref(b, loc, Abase);
  Value B_mr  = tensorToMemref(b, loc, Bbase);
  Value Bi_mr = tensorToMemref(b, loc, biasB);
  Value C_mr  = tensorToMemref(b, loc, Cbase);

  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value K = memrefDimAsI32(b, loc, A_mr, 1);
  Value N = memrefDimAsI32(b, loc, B_mr, 1);

  Value A_ptr  = memrefBasePtr(b, loc, A_mr);
  Value B_ptr  = memrefBasePtr(b, loc, B_mr);
  Value Bi_ptr = memrefBasePtr(b, loc, Bi_mr);
  Value C_ptr  = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module,
      "polygeist_cublaslt_matmul_bias_relu", argTypes, b);
  b.create<func::CallOp>(loc, shim,
      ValueRange{M, N, K, A_ptr, B_ptr, Bi_ptr, C_ptr});

  Value updated = memrefToTensor(b, loc, C_mr, Cbase.getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

// @cublasDsyrk_alias(%A_view, %A_view, %C_view) — fired by the matcher
// when a gemm-shape composition's two inputs resolve to the same
// underlying tensor (AᵀA or A·Aᵀ).
//
// After resolving submap, the three operands are:
//   - A: 2D (same SSA value for operand 0 and 1)
//   - A again (same as #0)
//   - C: 2D, symmetric (only upper triangle written by syrk)
//
// Routes to polygeist_cublas_dsyrk(N, K, A*, C*) — cublasDsyrk_v2 does
// the rank-K update in half the flops of the equivalent gemm.
static LogicalResult lowerCublasDsyrkAlias(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cublasDsyrk_alias: expected 3 operands");
  Value A0 = resolveSubmapBase(launch.getOperand(0));
  Value A1 = resolveSubmapBase(launch.getOperand(1));
  Value Cbase = resolveSubmapBase(launch.getOperand(2));
  if (A0 != A1)
    return launch.emitError(
        "cublasDsyrk_alias: matcher emitted this launch but the two "
        "input operands don't resolve to the same underlying tensor "
        "(matcher invariant violated)");
  auto At = dyn_cast<RankedTensorType>(A0.getType());
  auto Ct = dyn_cast<RankedTensorType>(Cbase.getType());
  if (!At || !Ct || At.getRank() != 2 || Ct.getRank() != 2 ||
      !At.getElementType().isF32() || !Ct.getElementType().isF32())
    return launch.emitError("cublasDsyrk_alias: A and C must be 2D f32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, A0);
  Value C_mr = tensorToMemref(b, loc, Cbase);

  // For AᵀA: A is K×N, C is N×N. So N = dim(A, 1), K = dim(A, 0).
  Value K = memrefDimAsI32(b, loc, A_mr, 0);
  Value N = memrefDimAsI32(b, loc, A_mr, 1);
  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_dsyrk",
                                      argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, K, A_ptr, C_ptr});

  Value updated = memrefToTensor(b, loc, C_mr, Cbase.getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

// @cublasGemmFor1x1Conv(%A_view, %F_view, %C_view) — 1×1 conv routed
// to gemm. After resolving submap → 3 base tensors:
//   - A: 4D (B, IC, H, W)
//   - F: 4D (OC, IC, 1, 1)
//   - C: 4D (B, OC, H, W)
//
// Reshape semantics: a 1×1 conv with stride 1 is exactly
//   C_flat[m, n] = sum_k A_flat[m, k] * F_flat[k, n]
// where m = B·H·W (flattened), k = IC, n = OC. So we call cublasSgemm
// with M=B·H·W, N=OC, K=IC.
//
// The matrix layout works out perfectly *if* the NCHW data is in row-
// major IC-strided form. For NCHW: A[b,c,h,w] is at byte
// b·IC·H·W + c·H·W + h·W + w. To view as (B·H·W, IC) row-major, we'd
// need bytes at (b·H·W + h·W + w)·IC + c. *Not the same layout.*
//
// So a strict NCHW→(B·H·W, IC) reshape requires a transpose. For now
// we route NHWC-equivalent flattening: cublas computes C_col such
// that C_col[m,n] = sum_k A_col[k, m] * F_col[n, k]. Pick op flags to
// match. The harness should be aware that the routed gemm semantics
// differ slightly from a "true" 1×1 conv — for inference workloads
// with matched layouts this is the right call, and the math we
// validate against (CPU 3-loop reference) does the same flattening.
static LogicalResult lowerCublasGemmFor1x1Conv(LaunchOp launch,
                                                 ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError(
        "cublasGemmFor1x1Conv: expected 3 operands, got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cublasGemmFor1x1Conv: expected 1 result");

  Value Abase = resolveSubmapBase(launch.getOperand(0));
  Value Fbase = resolveSubmapBase(launch.getOperand(1));
  Value Cbase = resolveSubmapBase(launch.getOperand(2));

  auto At = dyn_cast<RankedTensorType>(Abase.getType());
  auto Ft = dyn_cast<RankedTensorType>(Fbase.getType());
  auto Ct = dyn_cast<RankedTensorType>(Cbase.getType());
  if (!At || !Ft || !Ct || At.getRank() != 4 || Ft.getRank() != 4 ||
      Ct.getRank() != 4)
    return launch.emitError(
        "cublasGemmFor1x1Conv: input/filter/output must each be 4D");
  Type elemTy = At.getElementType();
  if (!elemTy.isF32())
    return launch.emitError("cublasGemmFor1x1Conv: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_mr = tensorToMemref(b, loc, Abase);
  Value F_mr = tensorToMemref(b, loc, Fbase);
  Value C_mr = tensorToMemref(b, loc, Cbase);

  // Pass B, IC, OC, HW = H*W (the batched gemm shim does B independent
  // (OC, HW) = (OC, IC) × (IC, HW) gemms in one cublasSgemmStridedBatched).
  Value Bdim = memrefDimAsI32(b, loc, A_mr, 0);
  Value IC   = memrefDimAsI32(b, loc, A_mr, 1);
  Value H    = memrefDimAsI32(b, loc, A_mr, 2);
  Value W    = memrefDimAsI32(b, loc, A_mr, 3);
  Value OC   = memrefDimAsI32(b, loc, F_mr, 0);
  Value HW   = b.create<arith::MulIOp>(loc, H, W);

  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value F_ptr = memrefBasePtr(b, loc, F_mr);
  Value C_ptr = memrefBasePtr(b, loc, C_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {
      b.getI32Type(), b.getI32Type(), b.getI32Type(), b.getI32Type(),
      ptrTy, ptrTy, ptrTy,
  };
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_sgemm_1x1conv",
                                      argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{Bdim, IC, OC, HW,
                                                A_ptr, F_ptr, C_ptr});

  Value updated = memrefToTensor(b, loc, C_mr, Cbase.getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct LowerKernelLaunchToCuBLASPass
    : public mlir::polygeist::LowerKernelLaunchToCuBLASBase<
          LowerKernelLaunchToCuBLASPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Track the set of kernel symbols we lower; after launches are gone we
    // delete any kernel.defn carrying one of these symbols, since no users
    // remain and downstream LLVM lowering doesn't know what kernel.defn is.
    llvm::SmallSet<StringRef, 4> loweredSymbols;

    SmallVector<LaunchOp> launches;
    module.walk([&](LaunchOp op) { launches.push_back(op); });

    // Pre-pass: elide redundant memset_zero_{1D,2D} launches that
    // immediately precede a launch whose runtime shim uses β=0
    // (cublasDsyrk_alias today; could be extended to any overwriting
    // op). The two launches show up as separate matches because the
    // matcher's gemm-2-step template requires `Out*β` for the first
    // step, not `Lit(0)`. After this pre-pass the memset is gone, so
    // the dataflow chain is just the syrk shim's input.
    SmallVector<LaunchOp> deadMemsets;
    for (LaunchOp launch : launches) {
      auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!sym) continue;
      if (sym.getLeafReference().getValue() != "cublasDsyrk_alias")
        continue;
      // Walk the syrk's output operand chain back to find the memset.
      Value v = launch.getOperand(2);
      for (int hops = 0; hops < 16; ++hops) {
        Operation *def = v.getDefiningOp();
        if (!def) break;
        if (auto sm = dyn_cast<polygeist::SubmapOp>(def)) {
          v = sm.getBase(); continue;
        }
        if (auto inv = dyn_cast<polygeist::SubmapInverseOp>(def)) {
          v = inv.getOperand(1); continue;
        }
        if (auto memsetLaunch = dyn_cast<LaunchOp>(def)) {
          auto msym = memsetLaunch->getAttrOfType<SymbolRefAttr>("kernel");
          if (msym && (msym.getLeafReference().getValue() == "memset_zero_2D" ||
                       msym.getLeafReference().getValue() == "memset_zero_2D_f32" ||
                       msym.getLeafReference().getValue() == "memset_zero_1D")) {
            // Replace memset result uses with its first operand (the
            // pre-init tensor). cublasSsyrk writes with β=0 anyway, so
            // the prior contents don't matter.
            if (memsetLaunch.getNumResults() == 1)
              memsetLaunch.getResult(0).replaceAllUsesWith(
                  memsetLaunch.getOperand(0));
            deadMemsets.push_back(memsetLaunch);
          }
          break;
        }
        break;
      }
    }
    for (LaunchOp m : deadMemsets) m.erase();
    // Re-collect launches now that some have been erased.
    launches.clear();
    module.walk([&](LaunchOp op) { launches.push_back(op); });

    if (deviceResidentCutensornet) {
      for (LaunchOp launch : launches) {
        auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
        if (!sym)
          continue;
        StringRef name = sym.getLeafReference().getValue();
        if (name == "cutensornetContraction2_f64" ||
            name == "cutensornetContraction2_f64_r4r5r4" ||
            name == "cutensornetContraction2_f64_r5r4r4" ||
            name == "cutensornetContraction2_f64_r5r5r4" ||
            name.starts_with("cutensornetNetwork_f32") ||
            name.starts_with("cutensornetNetwork_f64"))
          launch->setAttr("polygeist.device_resident",
                          UnitAttr::get(module.getContext()));
      }
    }

    for (LaunchOp launch : launches) {
      auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!sym) {
        launch.emitError(
            "kernel.launch missing 'kernel' symbol ref attribute");
        return signalPassFailure();
      }
      StringRef libSym = sym.getLeafReference().getValue();
      // Symbols claimed by other backend passes (e.g. PVA for int8/int16
      // conv2d) intentionally fall through — they're not errors here,
      // just "not our problem". Their own pass will lower them.
      if (libSym == "cudnnConvolution2D_9tap_i8" ||
          libSym == "cudnnConvolution2D_9tap_i16")
        continue;
      StringRef shim = shimSymbolFor(libSym);
      if (shim.empty()) {
        launch.emitError(
            "lower-kernel-launch-to-cublas: no shim ABI lowering for "
            "library symbol @")
            << libSym
            << ". Extend `shimSymbolFor` in "
               "LowerKernelLaunchToCuBLAS.cpp to add one.";
        return signalPassFailure();
      }

      LogicalResult r = failure();
      if (libSym == "cubHistogramEvenI32ShiftZero_memref") {
        r = lowerCubHistogramEvenI32ShiftZero(launch, module);
      } else if (libSym == "cublasDtrsvLowerRowMajor_memref") {
        r = lowerDtrsvLowerRowMajor(launch, module);
      } else if (libSym == "cusolverDnDpotrfLowerRowMajor_memref") {
        r = lowerDpotrfLowerRowMajor(launch, module);
      } else if (libSym == "cusparseSpMV_CSR_f32_memref" ||
          libSym == "cusparseSpMV_CSR_f64_memref") {
        r = lowerCusparseCsrSpmv(launch, module);
      } else if (libSym == "cusparseSpMM_CSR_f32_memref") {
        r = lowerCusparseSpmm(
            launch, module, "polygeist_cusparse_spmm_csr_f32_sized", false);
      } else if (libSym == "cusparseSpMM_COO_f32_memref") {
        r = lowerCusparseSpmm(
            launch, module, "polygeist_cusparse_spmm_coo_f32_sized", true);
      } else if (libSym == "cusparseSpMM_BSR_f32_memref") {
        r = lowerCusparseBsrSpmm(launch, module);
      } else if (libSym == "cusparseSpMV_JDS_f32_memref") {
        r = lowerCusparseJdsSpmv(launch, module);
      } else if (libSym.starts_with("cubSegmentedPrefix")) {
        r = libSym.ends_with("_memref")
                ? lowerCubSegmentedPrefixMemref(
                      launch, module,
                      libSym == "cubSegmentedPrefixSum_f32_memref")
                : lowerCubSegmentedPrefix(launch, module, libSym);
      } else if (libSym == "cubSegmentedSum_f32_memref" ||
                 libSym == "cubSegmentedSum_f64_memref" ||
                 libSym == "cubSegmentedMin_f32_memref" ||
                 libSym == "cubSegmentedMax_f32_memref" ||
                 libSym == "cubSegmentedBitXor_i32_memref") {
        r = lowerCubSegmentedFullMemref(launch, module, libSym);
      } else if (libSym.starts_with("cutensorPermute_f32_r") &&
                 libSym.ends_with("_tensor")) {
        r = lowerCutensorPermuteF32(launch, module, libSym);
      } else if (libSym == "cubSegmentedLogicalSelect_i32_tensor") {
        r = lowerCubSegmentedLogicalSelectI32(launch, module);
      } else if (libSym.starts_with("cubSegmented") &&
                 libSym.ends_with("_i32")) {
        r = lowerCubSegmentedI32(launch, module, libSym);
      } else if (libSym.starts_with("cudnnReduce")) {
        r = lowerCudnnReduction(launch, module, libSym);
      } else if (libSym.starts_with("cutensorUnary_") && libSym.ends_with("_f32")) {
        r = lowerCutensorUnaryF32(launch, module, libSym);
      } else if (libSym == "cublasDgemm") {
        r = lowerDgemm(launch, module);
      } else if (libSym == "cublasDgemm_simple" ||
                 libSym == "cublasDgemm_subtract" ||
                 libSym == "cublasDgemm_zero" ||
                 libSym == "cublasDgemm_alpha_only") {
        r = lowerDgemmVariant(launch, module, libSym);
      } else if (libSym ==
                 "cublasSgemm_broadcast3d_colmajor_nt_alpha_beta") {
        r = lowerSgemmBroadcast3DColMajorNTAlphaBeta(launch, module);
      } else if (libSym ==
                 "cublasSgemm_flat_colmajor_nt_alpha_beta") {
        r = lowerSgemmFlatColMajorNTAlphaBeta(launch, module);
      } else if (libSym == "cublasSgemm_nn" || libSym == "cublasSgemm_nn_zero" ||
                 libSym == "cublasSgemm_nt_zero" ||
                 libSym == "cublasSgemm_tn_zero" ||
                 libSym == "cublasSgemm_tt_zero" ||
                 libSym == "cublasSgemm_nt" ||
                 libSym == "cublasSgemm_tn" || libSym == "cublasSgemm_tt" ||
                 (libSym.starts_with("cublasSgemm_") &&
                  (libSym.ends_with("_alpha") ||
                   libSym.ends_with("_alpha_beta")))) {
        r = lowerSgemmTranspose(launch, module, libSym);
      } else if (libSym == "cublasSgemm_strided_batched_nn_zero") {
        r = lowerSgemmStridedBatched(launch, module);
      } else if (libSym == "cublasDgemm_strided_batched_subtract") {
        r = lowerDgemmStridedBatchedSubtract(launch, module);
      } else if (libSym == "cublasDgemv_strided_batched_subtract") {
        r = lowerDgemvStridedBatchedSubtract(launch, module);
      } else if (libSym == "cublasSgemm_broadcast3d_simple") {
        r = lowerSgemmBroadcast3DSimple(launch, module);
      } else if (libSym == "cublasSgemm_broadcast3d_memref") {
        r = lowerSgemmBroadcast3DMemRef(launch, module);
      } else if (libSym ==
                 "cublasSgemm_strided_batched_broadcast_rhs") {
        r = lowerSgemmStridedBatchedBroadcastRhs(launch, module);
      } else if (libSym == "cublasDgeam_scale2D") {
        r = lowerDgeamScale2D(launch, module);
      } else if (libSym == "cublasDgemv") {
        r = lowerDgemv(launch, module);
      } else if (libSym == "cublasDgemv_T") {
        r = lowerDgemvT(launch, module);
      } else if (libSym == "cublasDgemv_T_zero") {
        r = lowerDgemvTZero(launch, module);
      } else if (libSym == "cublasDgemv_subtract") {
        r = lowerDgemvSubtract(launch, module, /*transpose=*/false);
      } else if (libSym == "cublasDgemv_subtract_T") {
        r = lowerDgemvSubtract(launch, module, /*transpose=*/true);
      } else if (libSym == "cublasSgemv") {
        r = lowerSgemv(launch, module);
      } else if (libSym == "cublasSgemv_T") {
        r = lowerSgemvT(launch, module);
      } else if (libSym == "cublasDgemv_alpha") {
        r = lowerDgemvAlpha(launch, module);
      } else if (libSym == "cublasDaxpby") {
        r = lowerDaxpby(launch, module);
      } else if (libSym == "cublasSaxpby") {
        r = lowerSaxpby(launch, module);
      } else if (libSym == "cublasSscal") {
        r = lowerSscal(launch, module);
      } else if (libSym == "cublasDaxpy_unit") {
        r = lowerDaxpyUnit(launch, module);
      } else if (libSym == "cublasDger_rank2") {
        r = lowerDgerRank2(launch, module);
      } else if (libSym == "cublasDgemm_outer_product") {
        r = lowerDgemmOuterProduct(launch, module);
      } else if (libSym == "memset_zero_2D" ||
                 libSym == "memset_zero_2D_f32") {
        r = lowerMemsetZero2D(launch, module);
      } else if (libSym == "memset_zero_1D" ||
                 libSym == "memset_zero_1D_f32") {
        r = lowerMemsetZero1D(launch, module, libSym);
      } else if (libSym == "cudnnConvolution2D_9tap" ||
                 libSym == "cudnnConvolution2D_9tap_f32" ||
                 libSym == "cudnnConvolution2D_9tap_f16" ||
                 libSym == "cudnnConvolution2D_9tap_bf16" ||
                 libSym == "cudnnConvolution2D_9tap_i32") {
        // i8/i16 are handled by LowerKernelLaunchToPVA and aren't claimed
        // here by shimSymbolFor, so they're skipped above before we ever
        // reach this dispatch.
        r = lowerCudnnConv2D9tap(launch, module, shim);
      } else if (libSym == "cudnnConvolution2D_25tap" ||
                 libSym == "cudnnConvolution2D_25tap_f32") {
        r = lowerCudnnConv2D25tap(launch, module, shim);
      } else if (libSym == "cudnnConvolution2D_ntap" ||
                 libSym == "cudnnConvolution2D_ntap_f32") {
        r = lowerCudnnConv2DNtapPacked(launch, module, shim);
      } else if (libSym == "custenStencil2DXY_f64_memref") {
        r = lowerCustenStencil2D(launch, module, shim);
      } else if (libSym == "cudnnConvolution2D_ntap_tensor" ||
                 libSym == "cudnnConvolution2D_ntap_f32_tensor") {
        r = lowerCudnnConv2DNtapTensor(launch, module, shim);
      } else if (libSym == "custenStencil2DXY_f64_tensor") {
        r = lowerCustenStencil2D(launch, module, shim);
      } else if (libSym == "cudnnConvolution3D_ntap_tensor" ||
                 libSym == "cudnnConvolution3D_ntap_f32_tensor") {
        r = lowerCudnnConv3DNtapTensor(launch, module, shim);
      } else if (libSym == "cudnnStencil3D7pt_f32_flat_tensor") {
        r = lowerCudnnStencil3D7ptFlat(launch, module, shim);
      } else if (libSym == "cudnnConvolution3D_f32" ||
                 libSym == "cudnnConvolution3D_f32_bias") {
        r = lowerCudnnConv3DChannelsF32(
            launch, module, libSym == "cudnnConvolution3D_f32_bias");
      } else if (libSym == "cudnnConvolution1D_f32_bias") {
        r = lowerCudnnConv1DBiasF32(launch, module);
      } else if (libSym == "cudnnConvolution2D_f32_dilated") {
        r = lowerCudnnConv2DDilatedF32(launch, module);
      } else if (libSym == "cublasGemmEx_i8_i32_tensor") {
        r = lowerCublasGemmExI8I32(launch, module);
      } else if (libSym == "cublasSnrm2_f32_memref") {
        r = lowerCublasSnrm2F32(launch, module);
      } else if (libSym == "cublasJointMaxAbsProduct_f32_memref") {
        r = lowerCublasJointMaxAbsProductF32(launch, module);
      } else if (libSym == "cudnnFeatureMaskScale_f32_tensor") {
        r = lowerCudnnFeatureMaskScaleF32(launch, module);
      } else if (libSym == "cudnnConvolutionTranspose2D_f32_memref") {
        r = lowerCudnnConvTranspose2DF32(launch, module);
      } else if (libSym == "cudnnConvolutionTranspose3D_f32_memref") {
        r = lowerCudnnConvTranspose3DF32(launch, module);
      } else if (libSym == "cudnnConvolutionBackwardFilter3D_f32_memref") {
        r = lowerCudnnConvBackwardFilter3DF32(launch, module);
      } else if (libSym == "cudnnDepthwiseConvolution2D_f32_memref") {
        r = lowerCudnnDepthwiseConv2DF32(launch, module);
      } else if (libSym == "cutensorKroneckerProduct2D_f32_memref") {
        r = lowerCudnnKroneckerProduct2DF32(launch, module);
      } else if (libSym == "cudnnBinaryCrossEntropyMean_f32_memref") {
        r = lowerCudnnBinaryCrossEntropyMeanF32(launch, module);
      } else if (libSym == "cudnnConvolutionTBC_f32_memref") {
        r = lowerCudnnConvTBCF32(launch, module);
      } else if (libSym == "cudnnConvolutionTBCBackward_f32_memref") {
        r = lowerCudnnConvTBCBackwardF32(launch, module);
      } else if (libSym == "cudnnTransformBiasRescaleQKV_f32_memref") {
        r = lowerCudnnTransformBiasRescaleQKVF32(launch, module);
      } else if (libSym == "cudnnAddrElementwise_f32_memref") {
        r = lowerCudnnAddrElementwiseF32(launch, module);
      } else if (libSym == "cudnnLogSigmoid_f32_memref") {
        r = lowerCudnnLogSigmoidF32(launch, module);
      } else if (libSym == "cubSegmentedLogicalAnd_i32_memref") {
        r = lowerCubSegmentedLogicalMemrefI32(launch, module, false);
      } else if (libSym == "cubSegmentedLogicalSelect_i32_memref") {
        r = lowerCubSegmentedLogicalMemrefI32(launch, module, true);
      } else if (libSym == "cufftZ2Z_1D_tensor" ||
                 libSym == "cufftC2C_1D_tensor") {
        r = lowerCufftC2C1DTensor(launch, module, shim);
      } else if (libSym == "cutensornetTensorProduct3D_f32_tensor" ||
                 libSym == "cutensornetTensorProduct3D_f64_tensor") {
        r = lowerCutensornetTensorProduct3D(
            launch, module,
            libSym == "cutensornetTensorProduct3D_f64_tensor");
      } else if (libSym == "cutensornetContraction2_f64" ||
                 libSym == "cutensornetContraction2_f64_r4r5r4" ||
                 libSym == "cutensornetContraction2_f64_r5r4r4" ||
                 libSym == "cutensornetContraction2_f64_r5r5r4") {
        r = lowerCutensornetContraction2F64(launch, module);
      } else if (libSym.starts_with("cutensornetNetwork_f32") ||
                 libSym.starts_with("cutensornetNetwork_f64")) {
        r = lowerCutensornetNetwork(
            launch, module, libSym.starts_with("cutensornetNetwork_f64"));
      } else if (libSym == "cudnnConvolutionFwd_batched") {
        r = lowerCudnnConv2dBatched(launch, module);
      } else if (libSym == "cudnnConvolution2DWindow_f32") {
        r = lowerCudnnUniformWindowConv2DF32(launch, module);
      } else if (libSym == "cudnnAvgPoolWindow_f32") {
        r = lowerCudnnAvgPoolWindowF32(launch, module);
      } else if (libSym.starts_with("cudnnAdaptivePool_f32_") ||
                 libSym.starts_with("cudnnAveragePool_f32_")) {
        r = lowerCudnnAdaptivePoolF32(launch, module);
      } else if (libSym == "cudnnBatchNormBackward_f32_full" ||
                 libSym == "cudnnBatchNormBackward_f32_dx") {
        r = lowerCudnnBatchNormBackwardF32(
            launch, module, libSym == "cudnnBatchNormBackward_f32_full");
      } else if (libSym == "cudnnConvolutionFwd_im2col_gemm") {
        r = lowerCudnnConv2dIm2colGemm(launch, module);
      } else if (libSym == "cudnnMaxPoolFwd_batched") {
        r = lowerCudnnMaxpoolBatched(launch, module);
      } else if (libSym == "cudnnBatchNormalizationForwardInference") {
        r = lowerCudnnBatchnormInference(launch, module);
      } else if (libSym == "cudnnAddTensor_batched") {
        r = lowerCudnnAddTensorBatched(launch, module);
      } else if (libSym == "cudnnConvBnReluFwdFused") {
        r = lowerCudnnConvBnReluFused(launch, module);
      } else if (libSym == "cudnnConvBiasReluAddFwdFused") {
        r = lowerCudnnConvBiasReluAdd(launch, module);
      } else if (libSym == "cudnnPointwiseAffineRelu_f32") {
        r = lowerCudnnPointwiseAffineReluF32(launch, module);
      } else if (libSym == "cudnnPointwiseGraph_f32") {
        r = lowerCudnnPointwiseGraphF32(launch, module);
      } else if (libSym == "cubInclusiveSum1D_f32_tensor") {
        r = lowerCubInclusiveSum1DF32(launch, module);
      } else if (libSym == "cubSegmentedInclusiveProduct2D_f32_tensor") {
        r = lowerCubSegmentedInclusiveProduct2DF32(launch, module);
      } else if (libSym == "cubExclusiveSum1D_i32_memref") {
        r = lowerCubExclusiveSum1DI32(launch, module);
      } else if (libSym == "cubCountNonzero1D_f32_tensor" ||
                 libSym == "cubSegmentedCountNonzero2D_f32_tensor" ||
                 libSym == "cubEqualAll1D_f32_tensor") {
        r = lowerCubPredicateReduction(launch, module, libSym);
      } else if (libSym == "whisperExpShiftSum_f32_tensor") {
        r = lowerWhisperExpShiftSumF32(launch, module);
      } else if (libSym == "cublasDdot" || libSym == "cublasSdot") {
        r = lowerCublasDot(launch, module, libSym == "cublasSdot");
      } else if (libSym == "cublasSdot_memref" ||
                 libSym == "cublasDdot_memref") {
        r = lowerCublasDotMemref(
            launch, module, libSym == "cublasSdot_memref");
      } else if (libSym == "cubSegmentedArgMax_f32_i32_memref" ||
                 libSym == "cubSegmentedArgMin_f32_i32_memref") {
        r = lowerCubSegmentedArgReduceF32(
            launch, module, libSym == "cubSegmentedArgMin_f32_i32_memref");
      } else if (libSym == "cublasSgemvTZero_memref") {
        r = lowerCublasSgemvTZeroMemref(launch, module);
      } else if (libSym == "cudnnSinc_f32_memref") {
        r = lowerCudnnSpecialGraph(launch, module);
      } else if (libSym == "cubSegmentedSortDescending_f32_i32_memref") {
        r = lowerCubSegmentedSort(launch, module, false);
      } else if (libSym == "cubSegmentedTopKDescending_f32_i32_memref") {
        r = lowerCubSegmentedSort(launch, module, true);
      } else if (libSym == "cubSegmentReduceLengths_f32_memref") {
        r = lowerSegmentReduceLengths(launch, module);
      } else if (libSym == "cudnnSoftmaxForward" ||
                 libSym == "cudnnSoftmaxForward_tensor") {
        r = lowerCudnnSoftmaxForwardF32(launch, module);
      } else if (libSym == "cudnnSoftmaxForwardOut_tensor") {
        r = lowerCudnnSoftmaxForwardOutF32(launch, module);
      } else if (libSym == "cudaCopy1D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/1);
      } else if (libSym == "cudaCopy2D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/2);
      } else if (libSym == "cublasBroadcastAxis0_f32") {
        r = lowerCublasBroadcastF32(launch, module, /*axis=*/0);
      } else if (libSym == "cublasBroadcastAxis1_f32") {
        r = lowerCublasBroadcastF32(launch, module, /*axis=*/1);
      } else if (libSym == "cudaCopy3D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/3);
      } else if (libSym == "cudaCopy6D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/6);
      } else if (libSym == "cudaAdd_f32_tensor") {
        r = lowerCudaAddF32(launch, module);
      } else if (libSym == "cudaMaskSelect_f32_tensor") {
        r = lowerCudaMaskSelectF32(launch, module);
      } else if (libSym == "cudaSwiGLU_f32_tensor") {
        r = lowerCudaSwiGLUF32(launch, module);
      } else if (libSym == "cudaRopeMulMulSub_f32_tensor") {
        r = lowerCudaRopeMulMulF32(launch, module, /*add=*/false);
      } else if (libSym == "cudaRopeMulMulAdd_f32_tensor") {
        r = lowerCudaRopeMulMulF32(launch, module, /*add=*/true);
      } else if (libSym == "cublasLtMatmulBiasReluFused") {
        r = lowerCublasLtMatmulBiasRelu(launch, module);
      } else if (libSym == "cublasDsyrk_alias") {
        r = lowerCublasDsyrkAlias(launch, module);
      } else if (libSym == "cublasGemmFor1x1Conv") {
        r = lowerCublasGemmFor1x1Conv(launch, module);
      } else {
        launch.emitError("internal: shimSymbolFor recognised @")
            << libSym << " but no lowering branch dispatched";
        return signalPassFailure();
      }
      if (failed(r))
        return signalPassFailure();
      loweredSymbols.insert(libSym);
    }

    // Remove any kernel.defn that is now use-empty. After lowering, the
    // stub defns we injected to satisfy the verifier are dead — and
    // downstream LLVM lowering doesn't know what kernel.defn is.
    // (Don't filter by loweredSymbols: scripts often inject stubs for
    // every symbol the matcher might produce, only some of which the
    // input actually used.)
    SmallVector<DefnOp> deadDefns;
    module.walk([&](DefnOp d) {
      if (SymbolTable::symbolKnownUseEmpty(d, module))
        deadDefns.push_back(d);
    });
    for (DefnOp d : deadDefns)
      d.erase();

    // One-shot bufferization can materialize a copy for a tensor.insert_slice
    // that writes a destination-style launch result back into the exact same
    // subview.  CSE canonicalizes the two equivalent subview operations to one
    // SSA value; at that point the copy is unconditionally a no-op.  Remove it
    // here so an N-element identity copy cannot survive into the CPU epilogue.
    SmallVector<memref::CopyOp> identityCopies;
    module.walk([&](memref::CopyOp copy) {
      if (copy.getSource() == copy.getTarget())
        identityCopies.push_back(copy);
    });
    for (memref::CopyOp copy : identityCopies)
      copy.erase();
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerKernelLaunchToCuBLASPass() {
  return std::make_unique<LowerKernelLaunchToCuBLASPass>();
}
} // namespace polygeist
} // namespace mlir
