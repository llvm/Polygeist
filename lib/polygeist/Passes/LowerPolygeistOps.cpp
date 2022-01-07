//===- TrivialUse.cpp - Remove trivial use instruction ---------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace polygeist;
using namespace mlir::arith;

namespace {

struct SubIndexToReinterpretCast
    : public OpConversionPattern<polygeist::SubIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polygeist::SubIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcMemRefType = op.source().getType().cast<MemRefType>();
    auto resMemRefType = op.result().getType().cast<MemRefType>();
    auto shape = srcMemRefType.getShape();

    if (!resMemRefType.hasStaticShape())
      return failure();

    int64_t innerSize = resMemRefType.getNumElements();
    auto offset = rewriter.create<arith::MulIOp>(
        op.getLoc(), op.index(),
        rewriter.create<ConstantIndexOp>(op.getLoc(), innerSize));

    llvm::SmallVector<OpFoldResult> sizes, strides;
    for (auto dim : shape.drop_front()) {
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resMemRefType, op.source(), offset.getResult(), sizes, strides);

    return success();
  }
};

struct LowerPolygeistOpsPass
    : public LowerPolygeistOpsBase<LowerPolygeistOpsPass> {

  void runOnFunction() override {
    auto op = getOperation();
    auto ctx = op.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<SubIndexToReinterpretCast>(ctx);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<polygeist::PolygeistDialect>();
    target.addLegalDialect<arith::ArithmeticDialect, mlir::StandardOpsDialect,
                           memref::MemRefDialect>();

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerPolygeistOpsPass() {
  return std::make_unique<LowerPolygeistOpsPass>();
}

} // namespace polygeist
} // namespace mlir
