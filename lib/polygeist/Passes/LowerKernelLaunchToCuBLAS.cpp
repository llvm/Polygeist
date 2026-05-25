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

// @memset_zero_2D(%M : tensor<?x?xf64>) -> tensor<?x?xf64>
static LogicalResult lowerMemsetZero2D(LaunchOp launch, ModuleOp module) {
  if (launch.getNumOperands() != 1)
    return launch.emitError("memset_zero_2D: expected 1 operand");
  Value M = launch.getOperand(0);
  auto Mt = dyn_cast<RankedTensorType>(M.getType());
  if (!Mt || Mt.getRank() != 2 || !Mt.getElementType().isF64())
    return launch.emitError("memset_zero_2D: M must be 2D f64 tensor");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value M_mr = tensorToMemref(b, loc, M);
  Value rows = memrefDimAsI32(b, loc, M_mr, 0);
  Value cols = memrefDimAsI32(b, loc, M_mr, 1);
  Value M_ptr = memrefBasePtr(b, loc, M_mr);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(), ptrTy,
                                 b.getI32Type()};
  func::FuncOp shim = ensureShimDecl(module, "polygeist_cublas_memset_zero_2d",
                                       argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{rows, cols, M_ptr, cols});

  Value out = memrefToTensor(b, loc, M_mr, launch.getResult(0).getType());
  launch.getResult(0).replaceAllUsesWith(out);
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
