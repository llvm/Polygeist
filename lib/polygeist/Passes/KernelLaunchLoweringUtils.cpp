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

LogicalResult lowerCudnnConv2D9tap(LaunchOp launch, ModuleOp module,
                                    StringRef shimSymbol) {
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

  auto firstMr = dyn_cast<MemRefType>(launch.getOperand(0).getType());
  if (!firstMr || firstMr.getRank() != 2)
    return launch.emitError(
        "cudnnConvolution2D_9tap: operand 0 must be a 2D memref");
  Type elemTy = firstMr.getElementType();
  bool isSupportedInt = false;
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned w = intTy.getWidth();
    isSupportedInt = (w == 32 || w == 16 || w == 8);
  }
  if (!(elemTy.isF64() || elemTy.isF32() || elemTy.isF16() ||
        elemTy.isBF16() || isSupportedInt))
    return launch.emitError(
        "cudnnConvolution2D_9tap: element type must be f64/f32/f16/bf16/i32/i16/i8 (got ") << elemTy << ")";
  for (unsigned i = 0; i < 10; ++i) {
    auto mr = dyn_cast<MemRefType>(launch.getOperand(i).getType());
    if (!mr || mr.getRank() != 2 || mr.getElementType() != elemTy)
      return launch.emitError(
                 "cudnnConvolution2D_9tap: memref operands 0..9 must be 2D "
                 "memrefs with matching element type");
  }
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
    SmallVector<Type> argTypes = {b.getI32Type(), b.getI32Type()};
    for (unsigned i = 0; i < 9; ++i) argTypes.push_back(elemTy);
    argTypes.push_back(ptrTy);
    argTypes.push_back(ptrTy);
    func::FuncOp shim = ensureShimDecl(module, shimSymbol, argTypes, b);
    SmallVector<Value> callOperands = {M, N};
    for (unsigned i = 10; i < 19; ++i)
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
