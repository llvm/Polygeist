//===- ForBreakToWhile.cpp - scf.for(scf.if) to scf.while lowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::polygeist;

namespace {

struct ForBreakLoweringPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // The only operation is an scf.if.
    Block *body = forOp.getBody();
    if (!llvm::hasNItems(*body, 2))
      return failure();
    auto conditional = dyn_cast<scf::IfOp>(body->front());
    if (!conditional)
      return failure();

    // Its condition comes directly from iterargs.
    auto condition = conditional.getCondition().dyn_cast<BlockArgument>();
    if (!condition || condition.getOwner()->getParentOp() != forOp)
      return failure();
    unsigned iterArgPos = condition.getArgNumber() - 1;

    // The condition is initially <value> and remains false once changed to
    // false. Moveover, values don't change after the condition is set to false.
    auto yield = cast<scf::YieldOp>(body->back());
    auto yieldedCondition = yield.getOperand(iterArgPos).dyn_cast<OpResult>();
    if (yieldedCondition.getOwner() != conditional)
      return failure();

    Block *elseBlock = &conditional.getElseRegion().front();
    if (!llvm::hasSingleElement(*elseBlock))
      return failure();

    auto elseYield = cast<scf::YieldOp>(elseBlock->front());

    Block *forBegin = &forOp.getRegion().front();
    Block *forEnd = &forOp.getRegion().back();
    auto forYield = cast<scf::YieldOp>(forEnd->getTerminator());
    for (auto op : llvm::enumerate(forYield->getOperands())) {
      auto opp = op.value().dyn_cast<OpResult>();
      if (!opp) {
        return failure();
      }
      if (opp.getOwner() != conditional)
        return failure();

      auto BA =
          elseYield.getOperand(opp.getResultNumber()).dyn_cast<BlockArgument>();
      if (!BA) {
        if (iterArgPos == op.index())
          if (matchPattern(elseYield.getOperand(opp.getResultNumber()),
                           m_Zero()))
            continue;

        return failure();
      }
      if (BA.getOwner() != forBegin) {
        return failure();
      }
      if (BA.getArgNumber() != op.index() + 1) {
        return failure();
      }
    }

    SmallVector<Value> continueArgs;
    for (auto op : forYield->getOperands()) {
      if (auto opp = op.dyn_cast<OpResult>()) {
        if (opp.getOwner() == conditional) {
          continueArgs.push_back(
              conditional.thenYield()->getOperand(opp.getResultNumber()));
          continue;
        }
      }
      continueArgs.push_back(op);
    }

    auto loc = forOp.getLoc();

    // Build scf.WhileOp

    // Split the current block before the WhileOp to create the inlining point.
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    Block *continuation;
    if (forOp.getNumResults() == 0) {
      continuation = remainingOpsBlock;
    } else {
      continuation = rewriter.createBlock(
          remainingOpsBlock, forOp.getResultTypes(),
          SmallVector<Location>(forOp.getNumResults(), loc));
      rewriter.create<cf::BranchOp>(loc, remainingOpsBlock);
    }

    rewriter.inlineRegionBefore(forOp.getRegion(), continuation);

    Block *thenBegin = &conditional.getThenRegion().front();
    Block *thenEnd = &conditional.getThenRegion().back();

    rewriter.inlineRegionBefore(conditional.getThenRegion(), continuation);
    rewriter.replaceOp(conditional, continueArgs);

    rewriter.setInsertionPointToEnd(currentBlock);
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());
    SmallVector<Value> preInitArgs(initArgs);
    preInitArgs.erase(preInitArgs.begin());
    rewriter.create<cf::CondBranchOp>(
        loc,
        rewriter.create<arith::AndIOp>(
            loc,
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           forOp.getLowerBound(),
                                           forOp.getUpperBound()),
            preInitArgs[iterArgPos]),
        forBegin, initArgs, continuation, preInitArgs);

    rewriter.eraseOp(forYield);
    rewriter.setInsertionPointToEnd(forEnd);

    rewriter.create<cf::BranchOp>(loc, thenBegin, ValueRange());

    rewriter.eraseOp(cast<scf::YieldOp>(thenEnd->getTerminator()));
    rewriter.setInsertionPointToEnd(thenEnd);
    auto next = rewriter.create<arith::AddIOp>(loc, forBegin->getArgument(0),
                                               forOp.getStep());
    SmallVector<Value> innerExitArgs(continueArgs);
    continueArgs.insert(continueArgs.begin(), next);

    Value cmpOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 next, forOp.getUpperBound());
    Value andOp =
        rewriter.create<arith::AndIOp>(loc, cmpOp, innerExitArgs[iterArgPos]);
    rewriter.create<cf::CondBranchOp>(loc, andOp, forBegin, continueArgs,
                                      continuation, innerExitArgs);

    rewriter.replaceOp(forOp, continuation->getArguments());

    return success();
  }
};

struct ForBreakToWhileLoop : public ForBreakToWhileBase<ForBreakToWhileLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForBreakLoweringPattern>(patterns.getContext(), /*benefit=*/3);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace

void mlir::polygeist::populateForBreakToWhilePatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForBreakLoweringPattern>(patterns.getContext(), /*benefit=*/3);
}

std::unique_ptr<Pass> mlir::polygeist::createForBreakToWhilePass() {
  return std::make_unique<ForBreakToWhileLoop>();
}
