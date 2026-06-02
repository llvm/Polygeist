//===- KernelLaunchLoweringUtils.cpp - shared kernel.launch helpers ------===//

#include "KernelLaunchLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Kernel/KernelOps.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace mlir {
namespace polygeist {

func::FuncOp ensureShimDecl(ModuleOp module, StringRef shimSym,
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

Value memrefBasePtr(OpBuilder &b, Location loc, Value m) {
  auto mrTy = cast<MemRefType>(m.getType());
  auto eltTy = mrTy.getElementType();
  Value alignedIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, m);
  Value alignedI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);
  auto md = b.create<memref::ExtractStridedMetadataOp>(loc, m);
  Value offsetIdx = md.getOffset();
  Value offsetI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offsetIdx);
  unsigned bits = eltTy.getIntOrFloatBitWidth();
  Value eltBytes = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(bits / 8));
  Value byteOff = b.create<arith::MulIOp>(loc, offsetI64, eltBytes);
  Value byteAddr = b.create<arith::AddIOp>(loc, alignedI64, byteOff);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return b.create<LLVM::IntToPtrOp>(loc, ptrTy, byteAddr);
}

static LogicalResult lowerCudnnConv2DNtap(LaunchOp launch, ModuleOp module,
                                           StringRef shimSymbol,
                                           unsigned filterWidth,
                                           bool allowLegacy9tap) {
  unsigned taps = filterWidth * filterWidth;
  unsigned weightedOperands = taps + 1 + taps;
  unsigned n = launch.getNumOperands();
  if (n != weightedOperands && !(allowLegacy9tap && n == 10))
    return launch.emitError("cudnnConvolution2D_")
           << taps << "tap: expected " << weightedOperands << " operands "
           << "(" << taps << " input subviews + 1 output + " << taps
           << " weights)"
           << (allowLegacy9tap ? " or legacy 10 operands; got " : "; got ")
           << n;
  if (launch.getNumResults() != 0)
    return launch.emitError("cudnnConvolution2D_")
           << taps << "tap: expected memref-form (void) launch; got "
           << launch.getNumResults() << " result(s)";

  auto firstMr = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  if (!firstMr || firstMr.getRank() != 2)
    return launch.emitError("cudnnConvolution2D_")
           << taps << "tap: operand 0 must be a 2D memref";
  Type elemTy = firstMr.getElementType();
  bool isSupportedInt = false;
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned w = intTy.getWidth();
    isSupportedInt = (w == 32 || w == 16 || w == 8);
  }
  if (!(elemTy.isF64() || elemTy.isF32() || elemTy.isF16() ||
        elemTy.isBF16() || isSupportedInt))
    return launch.emitError("cudnnConvolution2D_")
           << taps
           << "tap: element type must be f64/f32/f16/bf16/i32/i16/i8 (got "
           << elemTy << ")";
  for (unsigned i = 0; i < taps + 1; ++i) {
    auto mr = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!mr || mr.getRank() != 2 || mr.getElementType() != elemTy)
      return launch.emitError("cudnnConvolution2D_")
             << taps << "tap: input/output memref operands must be 2D "
             << "memrefs with matching element type";
  }
  if (n == weightedOperands) {
    for (unsigned i = taps + 1; i < weightedOperands; ++i) {
      if (launch.getOperand(i).getType() != elemTy)
        return launch.emitError("cudnnConvolution2D_")
               << taps << "tap: weight operands must match memref elem type";
    }
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_subview = launch.getOperand(0);
  Value B_subview = launch.getOperand(taps);

  Value A_ptr = memrefBasePtr(b, loc, A_subview);
  Value B_ptr = memrefBasePtr(b, loc, B_subview);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value border_i32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(filterWidth - 1));
  Value h_idx = b.create<memref::DimOp>(loc, B_subview, c0);
  Value w_idx = b.create<memref::DimOp>(loc, B_subview, c1);
  Value h_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), h_idx);
  Value w_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), w_idx);
  Value M = b.create<arith::AddIOp>(loc, h_i32, border_i32);
  Value N = b.create<arith::AddIOp>(loc, w_i32, border_i32);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  if (n == weightedOperands) {
    SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type()};
    for (unsigned i = 0; i < taps; ++i) argTypes.push_back(elemTy);
    argTypes.push_back(ptrTy);
    argTypes.push_back(ptrTy);
    func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
    SmallVector<Value> callOperands = {M, N};
    for (unsigned i = taps + 1; i < weightedOperands; ++i)
      callOperands.push_back(launch.getOperand(i));
    callOperands.push_back(A_ptr);
    callOperands.push_back(B_ptr);
    b.create<func::CallOp>(loc, shim, callOperands);
  } else {
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

LogicalResult lowerCudnnConv2D9tap(LaunchOp launch, ModuleOp module,
                                    StringRef shimSymbol) {
  return lowerCudnnConv2DNtap(launch, module, shimSymbol,
                              /*filterWidth=*/3, /*allowLegacy9tap=*/true);
}

LogicalResult lowerCudnnConv2D25tap(LaunchOp launch, ModuleOp module,
                                     StringRef shimSymbol) {
  return lowerCudnnConv2DNtap(launch, module, shimSymbol,
                              /*filterWidth=*/5, /*allowLegacy9tap=*/false);
}

LogicalResult lowerCudnnConv2DNtapPacked(LaunchOp launch, ModuleOp module,
                                          StringRef shimSymbol) {
  if (launch.getNumOperands() != 4)
    return launch.emitError("cudnnConvolution2D_ntap: expected 4 operands "
                            "(input subview, output subview, weights, K); got ")
           << launch.getNumOperands();
  if (launch.getNumResults() != 0)
    return launch.emitError("cudnnConvolution2D_ntap: expected memref-form "
                            "(void) launch; got ")
           << launch.getNumResults() << " result(s)";

  Value A_subview = launch.getOperand(0);
  Value B_subview = launch.getOperand(1);
  Value W_memref = launch.getOperand(2);
  Value K = launch.getOperand(3);

  auto aTy = dyn_cast<MemRefType>(A_subview.getType());
  auto bTy = dyn_cast<MemRefType>(B_subview.getType());
  auto wTy = dyn_cast<MemRefType>(W_memref.getType());
  if (!aTy || aTy.getRank() != 2 || !bTy || bTy.getRank() != 2)
    return launch.emitError(
        "cudnnConvolution2D_ntap: input/output must be 2D memrefs");
  if (!wTy || wTy.getRank() != 1)
    return launch.emitError(
        "cudnnConvolution2D_ntap: weights must be a 1D memref");
  Type elemTy = aTy.getElementType();
  if (bTy.getElementType() != elemTy || wTy.getElementType() != elemTy)
    return launch.emitError(
        "cudnnConvolution2D_ntap: input/output/weights dtypes must match");
  if (!(elemTy.isF64() || elemTy.isF32()))
    return launch.emitError(
        "cudnnConvolution2D_ntap: only f64/f32 packed weights are supported");
  if (!K.getType().isInteger(32))
    return launch.emitError("cudnnConvolution2D_ntap: K must be i32");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_ptr = memrefBasePtr(b, loc, A_subview);
  Value B_ptr = memrefBasePtr(b, loc, B_subview);
  Value W_ptr = memrefBasePtr(b, loc, W_memref);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value oneI32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(1));
  Value border = b.create<arith::SubIOp>(loc, K, oneI32);
  Value h_idx = b.create<memref::DimOp>(loc, B_subview, c0);
  Value w_idx = b.create<memref::DimOp>(loc, B_subview, c1);
  Value h_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), h_idx);
  Value w_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), w_idx);
  Value M = b.create<arith::AddIOp>(loc, h_i32, border);
  Value N = b.create<arith::AddIOp>(loc, w_i32, border);

  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                b.getI32Type(), ptrTy, ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{M, N, K, W_ptr, A_ptr, B_ptr});

  launch.erase();
  return success();
}

LogicalResult lowerImageFilter2Operand(kernel::LaunchOp launch,
                                        ModuleOp module,
                                        StringRef shimSymbol) {
  unsigned n = launch.getNumOperands();
  if (n != 2)
    return launch.emitError(
               "image-filter-2op lowering: expected 2 operands "
               "(input subview + output subview); got ")
           << n;
  if (launch.getNumResults() != 0)
    return launch.emitError(
               "image-filter-2op lowering: expected memref-form (void) "
               "launch; got ")
           << launch.getNumResults() << " result(s)";

  auto inMr = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  auto outMr = dyn_cast<MemRefType>(launch.getOperand(1).getType());
  if (!inMr || inMr.getRank() != 2 || !outMr || outMr.getRank() != 2)
    return launch.emitError(
        "image-filter-2op lowering: both operands must be 2D memrefs");
  Type elemTy = inMr.getElementType();
  if (outMr.getElementType() != elemTy)
    return launch.emitError(
        "image-filter-2op lowering: input/output dtypes must match");
  auto intTy = dyn_cast<IntegerType>(elemTy);
  if (!intTy || !(intTy.getWidth() == 8 || intTy.getWidth() == 16))
    return launch.emitError(
        "image-filter-2op lowering: only i8 / i16 supported by PVA");

  OpBuilder b(launch);
  Location loc = launch.getLoc();
  Value A_subview = launch.getOperand(0);
  Value B_subview = launch.getOperand(1);

  Value A_ptr = memrefBasePtr(b, loc, A_subview);
  Value B_ptr = memrefBasePtr(b, loc, B_subview);

  // Same dim-recovery convention as the 9-tap conv lowering: the output
  // subview describes the (M-2)×(N-2) interior, so M/N = dim + 2.
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
  SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type(),
                                 ptrTy, ptrTy};
  func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
  b.create<func::CallOp>(loc, shim, ValueRange{M, N, A_ptr, B_ptr});
  launch.erase();
  return success();
}

} // namespace polygeist
} // namespace mlir
