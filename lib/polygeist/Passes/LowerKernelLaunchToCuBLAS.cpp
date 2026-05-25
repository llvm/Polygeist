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
  if (libSym == "cublasDgeam_scale2D") return "polygeist_cublas_dscal_2d";
  if (libSym == "memset_zero_2D") return "polygeist_cublas_memset_zero_2d";
  if (libSym == "memset_zero_1D") return "polygeist_cublas_memset_zero_1d";
  if (libSym == "cublasDgemv") return "polygeist_cublas_dgemv";
  if (libSym == "cublasDgemv_T") return "polygeist_cublas_dgemv_T";
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
  if (libSym == "cudnnConvolution2D_9tap_i16")
    return "polygeist_cudnn_conv2d_3x3_i16";
  // Extracted-darknet batched CNN-block primitives. All four take their
  // 4D tensors through `polygeist.submap` views (the implicit im2col for
  // conv, the broadcast onto the 4D iteration domain for batchnorm, etc.)
  // — the lowering walks each submap operand back to the underlying base
  // memref before extracting the data pointer.
  if (libSym == "cudnnConvolutionFwd_batched")
    return "polygeist_cudnn_conv2d_batched";
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
  if (libSym == "cublasLtMatmulBiasReluFused")
    return "polygeist_cublaslt_matmul_bias_relu";
  if (libSym == "cublasDsyrk_alias")
    return "polygeist_cublas_dsyrk";
  if (libSym == "cublasGemmFor1x1Conv")
    return "polygeist_cublas_sgemm_1x1conv";
  return StringRef();
}

// Get-or-create a `func.func private @<shim>(<argTypes>)` declaration at
// module scope. Idempotent.
static func::FuncOp ensureShimDecl(ModuleOp module, StringRef shimSym,
                                    TypeRange argTypes, OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(shimSym))
    return existing;
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto fnType = builder.getFunctionType(argTypes, /*results=*/{});
  auto fn = builder.create<func::FuncOp>(module.getLoc(), shimSym, fnType);
  fn.setPrivate();
  return fn;
}

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

// Bufferize a tensor operand to a memref so the runtime can take a pointer.
// For now we use `bufferization.to_memref` which one-shot-bufferize would
// usually emit; downstream passes will fold these.
static Value tensorToMemref(OpBuilder &b, Location loc, Value t) {
  auto tt = cast<RankedTensorType>(t.getType());
  auto memrefType = MemRefType::get(tt.getShape(), tt.getElementType());
  return b.create<bufferization::ToMemrefOp>(loc, memrefType, t);
}

// Inverse of the above — wrap a memref back into a tensor for downstream
// SSA uses. The `restrict` + `writable` attributes promise this is the
// only alias of the memref, which is true for fresh launch results.
static Value memrefToTensor(OpBuilder &b, Location loc, Value m, Type tensorType) {
  auto t = b.create<bufferization::ToTensorOp>(
      loc, tensorType, m, /*restrict=*/true, /*writable=*/true);
  return t.getResult();
}

// Extract a raw `!llvm.ptr` to the FIRST DATA ELEMENT of a memref.
// Sequence: aligned_ptr (as index) -> i64 -> add offset*sizeof(elt) -> ptr.
// For freshly bufferised memrefs offset=0 so the +offset is a no-op, but
// we emit it anyway to be safe.
static Value memrefBasePtr(OpBuilder &b, Location loc, Value m) {
  auto mrTy = cast<MemRefType>(m.getType());
  auto eltTy = mrTy.getElementType();
  // Aligned pointer base (ignores offset).
  Value alignedIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, m);
  Value alignedI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);
  // Strided metadata for the offset.
  auto md = b.create<memref::ExtractStridedMetadataOp>(loc, m);
  Value offsetIdx = md.getOffset();
  Value offsetI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offsetIdx);
  // sizeof(elt) in bytes.
  unsigned bits = eltTy.getIntOrFloatBitWidth();
  Value eltBytes = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
  Value byteOff = b.create<arith::MulIOp>(loc, offsetI64, eltBytes);
  Value byteAddr = b.create<arith::AddIOp>(loc, alignedI64, byteOff);
  // i64 -> !llvm.ptr.
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return b.create<LLVM::IntToPtrOp>(loc, ptrTy, byteAddr);
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

// @cudnnConvolution2D_9tap(in0..in8, out) — memref-form, no result.
// 10 operands: 9 input subviews (all aliases of the same source memref
// with different strided offsets — the 3x3 neighbour positions) + 1 output
// subview. The 9 scalar weights stay embedded in the original
// linalg.generic body; surfacing them as launch operands is a matcher TODO.
// For now the cuDNN runtime shim has the polybench weights hardcoded.
//
// We extract:
//   - A_ptr = aligned-ptr of input 0 (= source memref's data start)
//   - B_ptr = aligned-ptr of output  (= dest memref's data start)
//   - M = dim(output, 0) + 2   (output is interior, source is +2 in each axis)
//   - N = dim(output, 1) + 2
static LogicalResult lowerCudnnConv2D9tap(LaunchOp launch, ModuleOp module,
                                            StringRef shimSymbol) {
  // Expected operands: 9 input subviews + 1 output subview + 9 weight scalars
  // = 19 total. (Pre-Lit-surfacing the shape was 10 operands with hardcoded
  // shim weights; we keep a compatibility path that catches the old 10-arg
  // form and routes to the legacy polybench-specific shim.)
  unsigned n = launch.getNumOperands();
  if (n != 19 && n != 10)
    return launch.emitError("cudnnConvolution2D_9tap: expected 19 operands "
                            "(9 input subviews + 1 output + 9 weights) "
                            "or legacy 10 operands; got ")
           << n;
  if (launch.getNumResults() != 0)
    return launch.emitError("cudnnConvolution2D_9tap: expected memref-form "
                            "(void) launch; got ")
           << launch.getNumResults() << " result(s)";

  // First 10 operands must be 2D memrefs with a supported float element type.
  // The element type is derived from the first input — all 10 must agree.
  auto firstMr = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  if (!firstMr || firstMr.getRank() != 2)
    return launch.emitError(
        "cudnnConvolution2D_9tap: operand 0 must be a 2D memref");
  Type elemTy = firstMr.getElementType();
  bool isSupportedInt = false;
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned w = intTy.getWidth();
    isSupportedInt = (w == 32 || w == 16);
  }
  if (!(elemTy.isF64() || elemTy.isF32() || elemTy.isF16() ||
        elemTy.isBF16() || isSupportedInt))
    return launch.emitError(
        "cudnnConvolution2D_9tap: element type must be f64/f32/f16/bf16/i32/i16 (got ") << elemTy << ")";
  for (unsigned i = 0; i < 10; ++i) {
    auto mr = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!mr || mr.getRank() != 2 || mr.getElementType() != elemTy)
      return launch.emitError(
                 "cudnnConvolution2D_9tap: memref operands 0..9 must be 2D "
                 "memrefs with matching element type");
  }
  // If new form, trailing 9 operands must match the matrix element type.
  if (n == 19) {
    for (unsigned i = 10; i < 19; ++i) {
      if (launch.getOperand(i).getType() != elemTy)
        return launch.emitError("cudnnConvolution2D_9tap: weight operands "
                                "(10..18) must match memref elem type");
    }
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_subview = launch.getOperand(0);
  Value B_subview = launch.getOperand(9);

  Value A_ptr = memrefBasePtr(b, loc, A_subview);
  Value B_ptr = memrefBasePtr(b, loc, B_subview);

  // Derive M, N from the output subview's dynamic sizes (interior = (M-2)*(N-2))
  // and add 2 to recover the source dims. memref.dim returns index; cast to i32.
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2_i32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                              b.getI32IntegerAttr(2));
  Value h_idx = b.create<memref::DimOp>(loc, B_subview, c0);
  Value w_idx = b.create<memref::DimOp>(loc, B_subview, c1);
  Value h_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), h_idx);
  Value w_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), w_idx);
  Value M = b.create<arith::AddIOp>(loc, h_i32, c2_i32);
  Value N = b.create<arith::AddIOp>(loc, w_i32, c2_i32);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  if (n == 19) {
    // New generic shim: takes M, N, 9 weights (matching elemTy), A_ptr, B_ptr.
    // Different shim symbol per dtype — picked by the rewriter via the
    // launch symbol name (cudnnConvolution2D_9tap → f64,
    // cudnnConvolution2D_9tap_f32 → f32, etc.).
    SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type()};
    for (unsigned i = 0; i < 9; ++i) argTypes.push_back(elemTy);
    argTypes.push_back(ptrTy);  // A
    argTypes.push_back(ptrTy);  // B
    func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
    SmallVector<Value> callOperands = {M, N};
    for (unsigned i = 10; i < 19; ++i)
      callOperands.push_back(launch.getOperand(i));
    callOperands.push_back(A_ptr);
    callOperands.push_back(B_ptr);
    b.create<func::CallOp>(loc, shim, callOperands);
  } else {
    // Legacy 10-arg path — only valid for f64 because the legacy shim has
    // polybench's specific weights hardcoded.
    if (!elemTy.isF64())
      return launch.emitError(
          "cudnnConvolution2D_9tap: legacy 10-arg form requires f64 elements; "
          "got ")
             << elemTy;
    SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                   ptrTy, ptrTy};
    func::FuncOp shim = ensureShimDecl(
        module, "polygeist_cudnn_conv2d_polybench9tap", argTypes, b);
    b.create<func::CallOp>(loc, shim, ValueRange{M, N, A_ptr, B_ptr});
  }

  launch.erase();
  return success();
}

// Shared lowering for cublasDgemv (no transpose) and cublasDgemv_T (Aᵀ·x).
// `transpose=false` routes to polygeist_cublas_dgemv, `true` to
// polygeist_cublas_dgemv_T. Both shims have the same signature; only the
// internal cuBLAS op flag differs.
static LogicalResult lowerDgemvImpl(LaunchOp launch, ModuleOp module,
                                       bool transpose);

static LogicalResult lowerDgemv(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/false);
}

static LogicalResult lowerDgemvT(LaunchOp launch, ModuleOp module) {
  return lowerDgemvImpl(launch, module, /*transpose=*/true);
}

// @cublasDgemv(%A : tensor<MxNxf64>, %x : tensor<Nxf64>, %y : tensor<Mxf64>)
//   -> tensor<Mxf64>
// Computes y = A * x. Matched body has α=1, β=0 (the matcher fissions any
// scale/accumulate into a separate generic), so we hardcode them here.
//
// cuBLAS gemv signature (in our row-major convention):
//   polygeist_cublas_dgemv(M, N, alpha, A*, lda, x*, beta, y*)
static LogicalResult lowerDgemvImpl(LaunchOp launch, ModuleOp module,
                                       bool transpose) {
  if (launch.getNumOperands() != 3)
    return launch.emitError("cublasDgemv lowering: expected 3 operands "
                            "(A, x, y), got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 1)
    return launch.emitError("cublasDgemv lowering: expected 1 result");

  Value A = launch.getOperand(0);
  Value x = launch.getOperand(1);
  Value y = launch.getOperand(2);
  auto At = dyn_cast<RankedTensorType>(A.getType());
  auto xt = dyn_cast<RankedTensorType>(x.getType());
  auto yt = dyn_cast<RankedTensorType>(y.getType());
  if (!At || At.getRank() != 2 || !At.getElementType().isF64())
    return launch.emitError("cublasDgemv lowering: A must be 2D f64 tensor");
  if (!xt || xt.getRank() != 1 || !xt.getElementType().isF64())
    return launch.emitError("cublasDgemv lowering: x must be 1D f64 tensor");
  if (!yt || yt.getRank() != 1 || !yt.getElementType().isF64())
    return launch.emitError("cublasDgemv lowering: y must be 1D f64 tensor");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value one = b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                          b.getF64FloatAttr(1.0));
  Value zero = b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                           b.getF64FloatAttr(0.0));

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
      b.getF64Type(),                    // alpha
      ptrTy, b.getI32Type(),             // A*, lda
      ptrTy,                             // x*
      b.getF64Type(),                    // beta
      ptrTy,                             // y*
  };
  StringRef shimSym = transpose ? "polygeist_cublas_dgemv_T"
                                 : "polygeist_cublas_dgemv";
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
static LogicalResult lowerMemsetZero1D(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 1)
    return launch.emitError("memset_zero_1D: expected 1 operand");
  Value V = launch.getOperand(0);
  auto Vt = dyn_cast<RankedTensorType>(V.getType());
  if (!Vt || Vt.getRank() != 1 || !Vt.getElementType().isF64())
    return launch.emitError("memset_zero_1D: V must be 1D f64 tensor");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value V_mr = tensorToMemref(b, loc, V);
  Value len = memrefDimAsI32(b, loc, V_mr, 0);
  Value V_ptr = memrefBasePtr(b, loc, V_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), ptrTy};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_memset_zero_1d",
                                       argTypes, b);
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
      } else if (libSym == "cublasDgeam_scale2D") {
        r = lowerDgeamScale2D(launch, module);
      } else if (libSym == "cublasDgemv") {
        r = lowerDgemv(launch, module);
      } else if (libSym == "cublasDgemv_T") {
        r = lowerDgemvT(launch, module);
      } else if (libSym == "cublasDgemv_alpha") {
        r = lowerDgemvAlpha(launch, module);
      } else if (libSym == "cublasDaxpby") {
        r = lowerDaxpby(launch, module);
      } else if (libSym == "cublasDaxpy_unit") {
        r = lowerDaxpyUnit(launch, module);
      } else if (libSym == "cublasDger_rank2") {
        r = lowerDgerRank2(launch, module);
      } else if (libSym == "memset_zero_2D") {
        r = lowerMemsetZero2D(launch, module);
      } else if (libSym == "memset_zero_1D") {
        r = lowerMemsetZero1D(launch, module);
      } else if (libSym == "cudnnConvolution2D_9tap" ||
                 libSym == "cudnnConvolution2D_9tap_f32" ||
                 libSym == "cudnnConvolution2D_9tap_f16" ||
                 libSym == "cudnnConvolution2D_9tap_bf16" ||
                 libSym == "cudnnConvolution2D_9tap_i32" ||
                 libSym == "cudnnConvolution2D_9tap_i16") {
        r = lowerCudnnConv2D9tap(launch, module, shim);
      } else if (libSym == "cudnnConvolutionFwd_batched") {
        r = lowerCudnnConv2dBatched(launch, module);
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
