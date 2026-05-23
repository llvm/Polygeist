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
      } else {
        launch.emitError("internal: shimSymbolFor recognised @")
            << libSym << " but no lowering branch dispatched";
        return signalPassFailure();
      }
      if (failed(r))
        return signalPassFailure();
      loweredSymbols.insert(libSym);
    }

    // Remove kernel.defn declarations whose symbol we just lowered. They
    // were carrying the symbol that the launches referenced; now that the
    // launches are gone, the defns are dead and downstream LLVM lowering
    // would choke on them.
    SmallVector<DefnOp> deadDefns;
    module.walk([&](DefnOp d) {
      if (loweredSymbols.contains(d.getSymName()) &&
          SymbolTable::symbolKnownUseEmpty(d, module))
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
