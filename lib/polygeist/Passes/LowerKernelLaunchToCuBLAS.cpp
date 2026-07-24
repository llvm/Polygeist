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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
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
  if (libSym == "cublasDgemm") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_simple") return "polygeist_cublas_dgemm";
  if (libSym == "cublasDgemm_alpha_only") return "polygeist_cublas_dgemm";
  if (libSym == "cublasSgemm_broadcast3d_simple")
    return "polygeist_cublas_sgemm";
  if (libSym == "cublasSgemm_broadcast3d_memref")
    return "polygeist_cublas_sgemm";
  if (libSym == "cublasDgeam_scale2D") return "polygeist_cublas_dscal_2d";
  if (libSym == "memset_zero_2D") return "polygeist_cublas_memset_zero_2d";
  if (libSym == "memset_zero_2D_f32")
    return "polygeist_cublas_memset_zero_2d_f32";
  if (libSym == "memset_zero_1D") return "polygeist_cublas_memset_zero_1d";
  if (libSym == "memset_zero_1D_f32")
    return "polygeist_cublas_memset_zero_1d_f32";
  if (libSym == "cublasDgemv") return "polygeist_cublas_dgemv";
  if (libSym == "cublasDgemv_T") return "polygeist_cublas_dgemv_T";
  if (libSym == "cublasSgemv") return "polygeist_cublas_sgemv";
  if (libSym == "cublasSgemv_T") return "polygeist_cublas_sgemv_T";
  if (libSym == "cublasDgemv_alpha") return "polygeist_cublas_dgemv_alpha";
  if (libSym == "cublasDaxpby") return "polygeist_cublas_daxpby";
  if (libSym == "cublasDaxpy_unit") return "polygeist_cublas_daxpy_unit";
  if (libSym == "cublasDger_rank2") return "polygeist_cublas_dger_rank2";
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
  if (libSym == "customStencil3D7pt_f64_tensor" ||
      libSym == "customStencil3D7ptCoeff_f64_tensor" ||
      libSym == "customStencil3D7ptExtra_f64_tensor")
    return "polygeist_custom_stencil3d_7pt_flat_f64";
  if (libSym == "cufftZ2Z_1D_tensor")
    return "polygeist_cufft_z2z_1d";
  if (libSym == "cufftC2C_1D_tensor")
    return "polygeist_cufft_c2c_1d";
  if (libSym == "cutensornetTensorProduct3D_f32_tensor")
    return "polygeist_cutensornet_tensor_product_3d_f32";
  if (libSym == "cutensornetTensorProduct3D_f64_tensor")
    return "polygeist_cutensornet_tensor_product_3d_f64";
  if (libSym == "cutensornetContraction2_f64_r4r5r4" ||
      libSym == "cutensornetContraction2_f64_r5r4r4" ||
      libSym == "cutensornetContraction2_f64_r5r5r4")
    return "polygeist_cutensornet_contraction2_f64";
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
  if (libSym == "rmsnorm_f32")
    return "polygeist_rmsnorm_f32";
  if (libSym == "rmsnorm_f32_tensor")
    return "polygeist_rmsnorm_f32";
  if (libSym == "rmsnorm_unweighted_f32_tensor")
    return "polygeist_rmsnorm_unweighted_f32";
  if (libSym == "gelu_tanh_f32_tensor")
    return "polygeist_cuda_gelu_tanh_f32";
  if (libSym == "whisperExpShiftSum_f32_tensor")
    return "polygeist_whisper_exp_shift_sum_f32";
  if (libSym == "cublasDdot")
    return "polygeist_cublas_dot_f32";
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

// Bufferize a tensor operand to a memref so the runtime can take a pointer.
// For now we use `bufferization.to_memref` which one-shot-bufferize would
// usually emit; downstream passes will fold these.
static Value tensorToMemref(OpBuilder &b, Location loc, Value t) {
  auto tt = cast<RankedTensorType>(t.getType());
  auto memrefType = MemRefType::get(tt.getShape(), tt.getElementType());
  return b.create<bufferization::ToMemrefOp>(loc, memrefType, t);
}

static Value valueToMemref(OpBuilder &b, Location loc, Value v) {
  if (isa<MemRefType>(v.getType()))
    return v;
  return tensorToMemref(b, loc, v);
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
  if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>())
    return toTensor;
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

static Value pointerForTensorOrMemref(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
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

static Value numElementsForTensorOrMemref(OpBuilder &b, Location loc, Value v) {
  Value stripped = stripTensorCasts(v);
  if (auto slice = stripped.getDefiningOp<tensor::ExtractSliceOp>()) {
    Value total = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                              b.getI32IntegerAttr(1));
    for (OpFoldResult size : slice.getMixedSizes())
      total = b.create<arith::MulIOp>(loc, total,
                                      opFoldResultAsI32(b, loc, size));
    return total;
  }
  Value mr = valueToMemrefPreservingSlice(b, loc, v);
  return memrefNumElementsAsI32(b, loc, mr);
}

static Value dimForTensorOrMemrefAsI32(OpBuilder &b, Location loc, Value v,
                                       int64_t axis) {
  Value stripped = stripTensorCasts(v);
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

// Inverse of the above — wrap a memref back into a tensor for downstream
// SSA uses. The `restrict` + `writable` attributes promise this is the
// only alias of the memref, which is true for fresh launch results.
static Value memrefToTensor(OpBuilder &b, Location loc, Value m, Type tensorType) {
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
                                          Value updatedBaseTensor) {
  if (launch.getNumResults() == 0) return;
  Value res = launch.getResult(0);
  SmallVector<tensor::InsertSliceOp> inserts;
  if (updatedBaseTensor) {
    for (Operation *user : res.getUsers()) {
      if (auto insert = dyn_cast<tensor::InsertSliceOp>(user))
        if (insert.getSource() == res)
          inserts.push_back(insert);
    }
  }
  for (auto insert : inserts) {
    insert.getResult().replaceAllUsesWith(updatedBaseTensor);
    insert.erase();
  }
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
  Value C_mr = tensorToMemref(b, loc, C);

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
  launch.getResult(0).replaceAllUsesWith(resultTensor);
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
  } else {  // _simple
    beta = one;
    alpha = one;
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
  Value C_mr = valueToMemrefPreservingSlice(b, loc, C);
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

static LogicalResult lowerCustomStencil3D7ptF64Tensor(LaunchOp launch,
                                                      ModuleOp module,
                                                      StringRef libSym) {
  bool hasCoeff = libSym == "customStencil3D7ptCoeff_f64_tensor";
  bool hasExtra = libSym == "customStencil3D7ptExtra_f64_tensor";
  unsigned expected = hasCoeff || hasExtra ? 19 : 18;
  if (launch.getNumOperands() != expected)
    return launch.emitError("customStencil3D7pt lowering: expected ")
           << expected << " operands, got " << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("customStencil3D7pt lowering: expected 1 result");

  SmallVector<Value> taps;
  taps.reserve(7);
  for (unsigned i = 0; i < 7; ++i)
    taps.push_back(launch.getOperand(i));
  unsigned idx = 7;
  Value extra;
  Value coeff;
  if (hasExtra)
    extra = launch.getOperand(idx++);
  if (hasCoeff)
    coeff = launch.getOperand(idx++);
  Value out = launch.getOperand(idx++);
  SmallVector<Value> scalars;
  for (; idx < launch.getNumOperands(); ++idx)
    scalars.push_back(launch.getOperand(idx));
  if (scalars.size() != 10)
    return launch.emitError("customStencil3D7pt lowering: expected 10 scalar "
                            "coefficients");

  auto outTy = dyn_cast<RankedTensorType>(out.getType());
  auto resTy = dyn_cast<RankedTensorType>(launch.getResult(0).getType());
  if (!outTy || !resTy || outTy.getRank() != 3 || resTy.getRank() != 3 ||
      !outTy.getElementType().isF64() || !resTy.getElementType().isF64())
    return launch.emitError(
        "customStencil3D7pt lowering: output/result must be rank-3 f64 tensors");
  for (Value tap : taps) {
    auto ty = dyn_cast<RankedTensorType>(tap.getType());
    if (!ty || ty.getRank() != 3 || !ty.getElementType().isF64())
      return launch.emitError(
          "customStencil3D7pt lowering: all tap operands must be rank-3 f64 tensors");
  }
  if (hasExtra) {
    auto ty = dyn_cast<RankedTensorType>(extra.getType());
    if (!ty || ty.getRank() != 3 || !ty.getElementType().isF64())
      return launch.emitError(
          "customStencil3D7pt lowering: extra operand must be rank-3 f64 tensor");
  }
  if (hasCoeff) {
    auto ty = dyn_cast<RankedTensorType>(coeff.getType());
    if (!ty || ty.getRank() != 3 || !ty.getElementType().isF64())
      return launch.emitError(
          "customStencil3D7pt lowering: coeff operand must be rank-3 f64 tensor");
  }
  for (Value s : scalars)
    if (!s.getType().isF64())
      return launch.emitError(
          "customStencil3D7pt lowering: scalar coefficients must be f64");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  Value nullPtr = b.create<LLVM::ZeroOp>(loc, ptrTy);
  Value N = numElementsForTensorOrMemref(b, loc, out);

  SmallVector<Value> callOperands;
  callOperands.push_back(N);
  for (Value tap : taps)
    callOperands.push_back(pointerForTensorOrMemref(b, loc, tap));
  callOperands.push_back(hasExtra ? pointerForTensorOrMemref(b, loc, extra)
                                  : nullPtr);
  callOperands.push_back(hasCoeff ? pointerForTensorOrMemref(b, loc, coeff)
                                  : nullPtr);
  callOperands.push_back(pointerForTensorOrMemref(b, loc, out));
  callOperands.append(scalars.begin(), scalars.end());

  SmallVector<Type> argTypes;
  argTypes.push_back(b.getI32Type());
  for (unsigned i = 0; i < 10; ++i)
    argTypes.push_back(ptrTy);
  for (unsigned i = 0; i < 10; ++i)
    argTypes.push_back(b.getF64Type());
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_custom_stencil3d_7pt_flat_f64", argTypes, b);
  b.create<func::CallOp>(loc, shim, callOperands);

  Value updatedBase = tensorForSliceSource(b, loc, out);
  Value updated = updatedBase ? Value()
      : memrefToTensor(b, loc, valueToMemrefPreservingSlice(b, loc, out),
                       launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated, updatedBase);
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

struct ContractionViewMetadata {
  Value base;
  SmallVector<Value, 5> extents;
  SmallVector<Value, 5> strides;
  SmallVector<int64_t, 5> modes;
};

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

static FailureOr<ContractionViewMetadata>
buildContractionViewMetadata(OpBuilder &b, Location loc, Value operand,
                             AffineMap accessMap) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.getElementType().isF64() ||
      operandType.getRank() > 5 ||
      accessMap.getNumResults() != (unsigned)operandType.getRank())
    return failure();

  SmallVector<int64_t, 5> logicalModes;
  for (AffineExpr result : accessMap.getResults()) {
    auto dim = result.dyn_cast<AffineDimExpr>();
    if (!dim)
      return failure();
    logicalModes.push_back(dim.getPosition());
  }

  Value stripped = stripTensorCasts(operand);
  SmallVector<Value, 5> logicalExtents;
  SmallVector<Value, 5> logicalStrides;
  Value base = resolveSubmapBase(stripped);
  if (auto submap = stripped.getDefiningOp<polygeist::SubmapOp>()) {
    auto baseType = dyn_cast<RankedTensorType>(base.getType());
    if (!baseType || baseType.getRank() != 1 ||
        submap.getMap().getNumResults() != 1 ||
        submap.getSizes().size() != (unsigned)operandType.getRank())
      return failure();
    AffineExpr flatExpr = submap.getMap().getResult(0);
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

// Generic two-input FP64 Einstein contraction for MFEM's mode-wise
// sum-factorization stages. Metadata layout (all i64):
//   [rankA, rankB, rankC,
//    A.extent[5], A.stride[5], A.mode[5],
//    B.extent[5], B.stride[5], B.mode[5],
//    C.extent[5], C.stride[5], C.mode[5]]
// Unused slots are extent=1, stride=0, mode=-1.
static LogicalResult lowerCutensornetContraction2F64(LaunchOp launch,
                                                     ModuleOp module) {
  if (launch.getNumOperands() != 3 || launch.getNumResults() != 1)
    return launch.emitError(
        "cuTensorNet contraction: expected A/B/C operands and one result");
  auto mapsAttr = launch->getAttrOfType<ArrayAttr>("contraction_maps");
  if (!mapsAttr || mapsAttr.size() != 3)
    return launch.emitError(
        "cuTensorNet contraction: expected three contraction_maps");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  SmallVector<ContractionViewMetadata, 3> metadata;
  for (unsigned i = 0; i < 3; ++i) {
    auto mapAttr = dyn_cast<AffineMapAttr>(mapsAttr[i]);
    if (!mapAttr)
      return launch.emitError(
          "cuTensorNet contraction: contraction_maps must be affine maps");
    auto view =
        buildContractionViewMetadata(b, loc, launch.getOperand(i),
                                     mapAttr.getValue());
    if (failed(view))
      return launch.emitError(
          "cuTensorNet contraction: unsupported operand view/map layout");
    metadata.push_back(*view);
  }
  if (metadata[2].modes.size() != 4)
    return launch.emitError(
        "cuTensorNet contraction: output must have four non-broadcast modes");

  constexpr int64_t kMaxRank = 5;
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
  for (const ContractionViewMetadata &view : metadata) {
    Value memref = valueToMemref(b, loc, view.base);
    memrefs.push_back(memref);
    pointers.push_back(memrefBasePtr(b, loc, memref));
  }
  Value metadataPtr = memrefBasePtr(b, loc, metadataBuffer);
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes(4, ptrType);
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_cutensornet_contraction2_f64", argTypes, b);
  b.create<func::CallOp>(
      loc, shim,
      ValueRange{pointers[0], pointers[1], pointers[2], metadataPtr});

  Value updatedOutput =
      memrefToTensor(b, loc, memrefs[2], metadata[2].base.getType());
  rewireLaunchResult(launch, updatedOutput);
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
  SmallVector<Value> callOperands = {M, N, K, alpha, A_ptr, K,
                                     B_ptr, N, beta, C_ptr, N};
  b.create<func::CallOp>(loc, shim, callOperands);

  Value updatedBaseTensor = memrefToTensor(b, loc, C_mr, C_base.getType());
  rewireLaunchResult(launch, updatedBaseTensor);
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
                                       bool transpose, bool useF32);

static LogicalResult lowerDgemv(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/false, /*useF32=*/false);
}

static LogicalResult lowerDgemvT(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true, /*useF32=*/false);
}

static LogicalResult lowerSgemv(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/false, /*useF32=*/true);
}

static LogicalResult lowerSgemvT(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true, /*useF32=*/true);
}

// @cublasDgemv(%A : tensor<MxNxf64>, %x : tensor<Nxf64>, %y : tensor<Mxf64>)
//   -> tensor<Mxf64>
// Computes y = A * x. Matched body has α=1, β=0 (the matcher fissions any
// scale/accumulate into a separate generic), so we hardcode them here.
//
// cuBLAS gemv signature (in our row-major convention):
//   polygeist_cublas_dgemv(M, N, alpha, A*, lda, x*, beta, y*)
static LogicalResult lowerDgemvImpl(LaunchOp launch, ModuleOp module,
                                       bool transpose, bool useF32) {
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
  TypedAttr zeroAttr = useF32 ? b.getF32FloatAttr(0.0f)
                              : b.getF64FloatAttr(0.0);
  Value one = b.create<arith::ConstantOp>(loc, scalarTy, oneAttr);
  Value zero = b.create<arith::ConstantOp>(loc, scalarTy, zeroAttr);

  Value A_mr = tensorToMemref(b, loc, A);
  Value x_mr = tensorToMemref(b, loc, x);
  Value y_mr = tensorToMemref(b, loc, y);

  Value M = memrefDimAsI32(b, loc, A_mr, 0);
  Value N = memrefDimAsI32(b, loc, A_mr, 1);
  Value lda = N;  // row-major

  Value A_ptr = memrefBasePtr(b, loc, A_mr);
  Value x_ptr = memrefBasePtr(b, loc, x_mr);
  Value y_ptr = memrefBasePtr(b, loc, y_mr);

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
      ValueRange{M, N, one, A_ptr, lda, x_ptr, zero, y_ptr});

  Value out = memrefToTensor(b, loc, y_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
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
  Value V_mr = tensorToMemref(b, loc, V);
  Value len = memrefDimAsI32(b, loc, V_mr, 0);
  Value V_ptr = memrefBasePtr(b, loc, V_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy};
  StringRef shimName = isF32Variant ? "polygeist_cublas_memset_zero_1d_f32"
                                    : "polygeist_cublas_memset_zero_1d";
  func::FuncOp shim = ensureShimDecl(module, shimName, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{len, V_ptr});

  Value out = memrefToTensor(b, loc, V_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
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
  Value O_mr = tensorToMemref(b, loc, outputBase);

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
  rewireLaunchResult(launch, updated);
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
  Value O_mr = tensorToMemref(b, loc, outBase);
  Value B  = memrefDimAsI32(b, loc, A_mr, 0);
  Value C  = memrefDimAsI32(b, loc, A_mr, 1);
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
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

// @cudnnBatchNormalizationForwardInference(
//     %scale_view, %A_view, %mean_view, %inv_std_view, %bias_view,
//     %output_view)
//
// All 6 operands are submap views. The raise pass orders them
// (scale, A, mean, inv_std, bias) — see the matcher template
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

  Value scaleBase   = resolveSubmapBase(launch.getOperand(0));
  Value aBase       = resolveSubmapBase(launch.getOperand(1));
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
  Value O_mr  = tensorToMemref(b, loc, outBase);

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
  rewireLaunchResult(launch, updated);
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
  rewireLaunchResult(launch, updated);
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

// @rmsnorm(%x, %weight, %out), FP32 1D memref/tensor operands.
// Runtime computes:
//   out[i] = weight[i] * x[i] * rsqrt(sum_j x[j]^2 / N + 1e-5)
static LogicalResult lowerRmsnormF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("rmsnorm: expected 3 operands (x, weight, out)");
  if (launch.getNumResults() > 1)
    return launch.emitError("rmsnorm: expected zero or one result");

  Value x = resolveSubmapBase(launch.getOperand(0));
  Value weight = resolveSubmapBase(launch.getOperand(1));
  Value out = resolveSubmapBase(launch.getOperand(2));

  ShapedType xTy = getRankedShapedType(x);
  ShapedType wTy = getRankedShapedType(weight);
  ShapedType oTy = getRankedShapedType(out);
  if (!xTy || !wTy || !oTy || xTy.getRank() != 1 || wTy.getRank() != 1 ||
      oTy.getRank() != 1)
    return launch.emitError("rmsnorm: x/weight/out must be ranked 1D");
  if (!xTy.getElementType().isF32() ||
      wTy.getElementType() != xTy.getElementType() ||
      oTy.getElementType() != xTy.getElementType())
    return launch.emitError("rmsnorm: only f32 x/weight/out supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemref(b, loc, x);
  Value wMr = valueToMemref(b, loc, weight);
  Value oMr = valueToMemref(b, loc, out);

  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value wPtr = memrefBasePtr(b, loc, wMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_rmsnorm_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, wPtr, oPtr});

  if (launch.getNumResults() == 1) {
    Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
    rewireLaunchResult(launch, updated);
  }

  launch.erase();
  return success();
}

// @rmsnorm_unweighted_f32_tensor(%x, %out), FP32 1D tensor operands.
// Runtime computes:
//   out[i] = x[i] * rsqrt(sum_j x[j]^2 / N + 1e-5)
static LogicalResult lowerRmsnormUnweightedF32(LaunchOp launch,
                                               ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError(
        "rmsnorm_unweighted: expected 2 operands (x, out)");
  if (launch.getNumResults() != 1)
    return launch.emitError("rmsnorm_unweighted: expected one result");

  Value x = resolveSubmapBase(launch.getOperand(0));
  Value out = resolveSubmapBase(launch.getOperand(1));

  ShapedType xTy = getRankedShapedType(x);
  ShapedType oTy = getRankedShapedType(out);
  if (!xTy || !oTy || xTy.getRank() != 1 || oTy.getRank() != 1)
    return launch.emitError("rmsnorm_unweighted: x/out must be ranked 1D");
  if (!xTy.getElementType().isF32() ||
      oTy.getElementType() != xTy.getElementType())
    return launch.emitError("rmsnorm_unweighted: only f32 supported");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(
      module, "polygeist_rmsnorm_unweighted_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
  launch.erase();
  return success();
}

static LogicalResult lowerCublasDotF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cublasDdot: expected 3 operands (x, y, out)");
  if (launch.getNumResults() != 1)
    return launch.emitError("cublasDdot: expected one result");

  Value x = resolveSubmapBase(launch.getOperand(0));
  Value y = resolveSubmapBase(launch.getOperand(1));
  Value out = resolveSubmapBase(launch.getOperand(2));

  ShapedType xTy = getRankedShapedType(x);
  ShapedType yTy = getRankedShapedType(y);
  ShapedType oTy = getRankedShapedType(out);
  if (!xTy || !yTy || !oTy || xTy.getRank() != 1 || yTy.getRank() != 1 ||
      oTy.getRank() != 0)
    return launch.emitError("cublasDdot: x/y must be 1D and out rank-0");
  if (!xTy.getElementType().isF32() ||
      yTy.getElementType() != xTy.getElementType() ||
      oTy.getElementType() != xTy.getElementType())
    return launch.emitError("cublasDdot: only f32 supported in this ABI");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value yMr = valueToMemrefPreservingSlice(b, loc, y);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value yPtr = memrefBasePtr(b, loc, yMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cublas_dot_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, yPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireLaunchResult(launch, updated);
  launch.erase();
  return success();
}

static LogicalResult lowerGeluTanhF32(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 2)
    return launch.emitError("gelu_tanh_f32: expected 2 operands (x, out)");
  if (launch.getNumResults() != 1)
    return launch.emitError("gelu_tanh_f32: expected one result");

  Value x = launch.getOperand(0);
  Value out = launch.getOperand(1);
  auto xTy = dyn_cast<RankedTensorType>(x.getType());
  auto oTy = dyn_cast<RankedTensorType>(out.getType());
  if (!xTy || !oTy || xTy.getRank() != 1 || oTy.getRank() != 1 ||
      !xTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return launch.emitError("gelu_tanh_f32: operands must be 1D f32 tensors");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value xMr = valueToMemrefPreservingSlice(b, loc, x);
  Value oMr = valueToMemrefPreservingSlice(b, loc, out);
  Value N = memrefDimAsI32(b, loc, xMr, 0);
  Value xPtr = memrefBasePtr(b, loc, xMr);
  Value oPtr = memrefBasePtr(b, loc, oMr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_gelu_tanh_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, xPtr, oPtr});

  Value updated = memrefToTensor(b, loc, oMr, launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated,
                                tensorForSliceSource(b, loc, out));
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
  Value N = numElementsForTensorOrMemref(b, loc, src);
  Value sPtr = pointerForTensorOrMemref(b, loc, src);
  Value oPtr = pointerForTensorOrMemref(b, loc, out);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy, ptrTy};
  func::FuncOp shim =
      ensureShimDecl(module, "polygeist_cuda_copy_f32", argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{N, sPtr, oPtr});

  Value updatedBase = tensorForSliceSource(b, loc, out);
  Value updated = updatedBase ? Value()
      : memrefToTensor(b, loc, valueToMemrefPreservingSlice(b, loc, out),
                       launch.getResult(0).getType());
  rewireTensorSliceLaunchResult(launch, updated, updatedBase);
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
      if (libSym == "cublasDgemm") {
        r = lowerDgemm(launch, module);
      } else if (libSym == "cublasDgemm_simple" ||
                 libSym == "cublasDgemm_alpha_only") {
        r = lowerDgemmVariant(launch, module, libSym);
      } else if (libSym == "cublasSgemm_broadcast3d_simple") {
        r = lowerSgemmBroadcast3DSimple(launch, module);
      } else if (libSym == "cublasSgemm_broadcast3d_memref") {
        r = lowerSgemmBroadcast3DMemRef(launch, module);
      } else if (libSym == "cublasDgeam_scale2D") {
        r = lowerDgeamScale2D(launch, module);
      } else if (libSym == "cublasDgemv") {
        r = lowerDgemv(launch, module);
      } else if (libSym == "cublasDgemv_T") {
        r = lowerDgemvT(launch, module);
      } else if (libSym == "cublasSgemv") {
        r = lowerSgemv(launch, module);
      } else if (libSym == "cublasSgemv_T") {
        r = lowerSgemvT(launch, module);
      } else if (libSym == "cublasDgemv_alpha") {
        r = lowerDgemvAlpha(launch, module);
      } else if (libSym == "cublasDaxpby") {
        r = lowerDaxpby(launch, module);
      } else if (libSym == "cublasDaxpy_unit") {
        r = lowerDaxpyUnit(launch, module);
      } else if (libSym == "cublasDger_rank2") {
        r = lowerDgerRank2(launch, module);
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
      } else if (libSym == "cudnnConvolution2D_ntap_tensor" ||
                 libSym == "cudnnConvolution2D_ntap_f32_tensor") {
        r = lowerCudnnConv2DNtapTensor(launch, module, shim);
      } else if (libSym == "cudnnConvolution3D_ntap_tensor" ||
                 libSym == "cudnnConvolution3D_ntap_f32_tensor") {
        r = lowerCudnnConv3DNtapTensor(launch, module, shim);
      } else if (libSym == "customStencil3D7pt_f64_tensor" ||
                 libSym == "customStencil3D7ptCoeff_f64_tensor" ||
                 libSym == "customStencil3D7ptExtra_f64_tensor") {
        r = lowerCustomStencil3D7ptF64Tensor(launch, module, libSym);
      } else if (libSym == "cufftZ2Z_1D_tensor" ||
                 libSym == "cufftC2C_1D_tensor") {
        r = lowerCufftC2C1DTensor(launch, module, shim);
      } else if (libSym == "cutensornetTensorProduct3D_f32_tensor" ||
                 libSym == "cutensornetTensorProduct3D_f64_tensor") {
        r = lowerCutensornetTensorProduct3D(
            launch, module,
            libSym == "cutensornetTensorProduct3D_f64_tensor");
      } else if (libSym == "cutensornetContraction2_f64_r4r5r4" ||
                 libSym == "cutensornetContraction2_f64_r5r4r4" ||
                 libSym == "cutensornetContraction2_f64_r5r5r4") {
        r = lowerCutensornetContraction2F64(launch, module);
      } else if (libSym == "cudnnConvolutionFwd_batched") {
        r = lowerCudnnConv2dBatched(launch, module);
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
      } else if (libSym == "rmsnorm_f32" ||
                 libSym == "rmsnorm_f32_tensor") {
        r = lowerRmsnormF32(launch, module);
      } else if (libSym == "rmsnorm_unweighted_f32_tensor") {
        r = lowerRmsnormUnweightedF32(launch, module);
      } else if (libSym == "gelu_tanh_f32_tensor") {
        r = lowerGeluTanhF32(launch, module);
      } else if (libSym == "whisperExpShiftSum_f32_tensor") {
        r = lowerWhisperExpShiftSumF32(launch, module);
      } else if (libSym == "cublasDdot") {
        r = lowerCublasDotF32(launch, module);
      } else if (libSym == "cudnnSoftmaxForward" ||
                 libSym == "cudnnSoftmaxForward_tensor") {
        r = lowerCudnnSoftmaxForwardF32(launch, module);
      } else if (libSym == "cudnnSoftmaxForwardOut_tensor") {
        r = lowerCudnnSoftmaxForwardOutF32(launch, module);
      } else if (libSym == "cudaCopy1D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/1);
      } else if (libSym == "cudaCopy2D_f32_tensor") {
        r = lowerCudaCopyF32(launch, module, /*expectedRank=*/2);
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
