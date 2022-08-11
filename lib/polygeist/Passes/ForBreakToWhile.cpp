//===- ForBreakToWhile.cpp - scf.for(scf.if) to scf.while lowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

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
    auto conditional = cast<scf::IfOp>(body->front());
    if (!conditional)
      return failure();

    // Its condition comes directly from iterargs.
    auto condition = conditional.getCondition().dyn_cast<BlockArgument>();
    if (!condition || condition.getOwner()->getParentOp() != forOp)
      return failure();
    unsigned iterArgPos = condition.getArgNumber() - 1;

    // The condition is initially true and remains false once changed to false.
    auto yield = cast<scf::YieldOp>(body->back());
    auto yieldedCondition = yield.getOperand(iterArgPos).dyn_cast<OpResult>();
    if (yieldedCondition.getOwner() != conditional)
      return failure();
    unsigned conditionResPos = yieldedCondition.getResultNumber();
    Block *elseBlock = &conditional.getElseRegion().front();
    if (!llvm::hasSingleElement(*elseBlock))
      return failure();
    auto elseYield = cast<scf::YieldOp>(elseBlock->front());
    if (!matchPattern(elseYield.getOperand(conditionResPos), m_Zero()) ||
        !matchPattern(forOp.getOpOperandForRegionIterArg(condition).get(),
                      m_One()))
      return failure();

    // Generate type signature for the loop-carried values. The induction
    // variable is placed first, followed by the forOp.iterArgs.
    SmallVector<Type> lcvTypes;
    SmallVector<Location> lcvLocs;
    lcvTypes.push_back(forOp.getInductionVar().getType());
    lcvLocs.push_back(forOp.getInductionVar().getLoc());
    for (Value value : forOp.getInitArgs()) {
      lcvTypes.push_back(value.getType());
      lcvLocs.push_back(value.getLoc());
    }

    // Build scf.WhileOp
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());
    auto whileOp = rewriter.create<WhileOp>(forOp.getLoc(), lcvTypes, initArgs,
                                            forOp->getAttrs());

    // 'before' region contains the loop condition and its conjunction with the
    // conditional condition, as well as forwarding of iteration arguments to
    // the 'after' region.
    auto *beforeBlock = rewriter.createBlock(
        &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
    rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
    Value cmpOp = rewriter.create<arith::CmpIOp>(
        whileOp.getLoc(), arith::CmpIPredicate::slt,
        beforeBlock->getArgument(0), forOp.getUpperBound());
    Value andOp = rewriter.create<arith::AndIOp>(
        whileOp.getLoc(), cmpOp, whileOp.getBeforeArguments()[iterArgPos + 1]);
    // TODO: consider not forwarding the condition variable.
    rewriter.create<scf::ConditionOp>(whileOp.getLoc(), andOp,
                                      beforeBlock->getArguments());

    // Inline conditional body into the "after" region.
    auto *afterBlock = rewriter.createBlock(
        &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);

    // Rewrite uses of the conditional block arguments to the new while-loop
    // "after" arguments
    SmallVector<Value> arguments;
    for (BlockArgument barg : conditional.getBody(0)->getArguments()) {
      auto conditionalOperand = conditional->getOperand(barg.getArgNumber())
                                    .dyn_cast<BlockArgument>();
      if (!conditionalOperand ||
          conditionalOperand.getOwner()->getParentOp() != forOp) {
        arguments.push_back(conditional->getOperand(barg.getArgNumber()));
      } else {
        arguments.push_back(
            afterBlock->getArgument(conditionalOperand.getArgNumber()));
      }
    }

    // Update uses of block args of the original loop.
    for (BlockArgument arg : forOp.getBody()->getArguments()) {
      for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
        rewriter.updateRootInPlace(use.getOwner(), [&] {
          use.set(afterBlock->getArgument(arg.getArgNumber()));
        });
      }
    }

    // Inline the conditional body operations into 'after' region.
    rewriter.mergeBlocks(conditional.getBody(0), afterBlock, arguments);

    // Add induction variable increment.
    rewriter.setInsertionPoint(&afterBlock->back());
    auto ivIncOp = rewriter.create<arith::AddIOp>(
        whileOp.getLoc(), afterBlock->getArgument(0), forOp.getStep());

    // Create a new yield.
    auto thenYield = cast<scf::YieldOp>(afterBlock->back());
    rewriter.setInsertionPointToEnd(afterBlock);
    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(1 + yield.getNumOperands());
    yieldOperands.push_back(ivIncOp);
    for (Value operand : yield.getOperands()) {
      auto operandOpResult = operand.dyn_cast<OpResult>();
      if (operandOpResult && operandOpResult.getOwner() == conditional) {
        yieldOperands.push_back(
            thenYield.getOperand(operandOpResult.getResultNumber()));
      } else {
        yieldOperands.push_back(operand);
      }
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYield, yieldOperands);
    rewriter.replaceOp(forOp, whileOp.getResults().drop_front());
    return success();
  }
};

struct ForBreakToWhileLoop : public ForBreakToWhileBase<ForBreakToWhileLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForBreakLoweringPattern>(ctx);
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
