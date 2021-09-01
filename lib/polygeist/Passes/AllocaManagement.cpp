//===- AllocaManagement.cpp - Passes for dealing with memref.alloca -------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;

namespace {

struct AllocaToAlloc : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getBlock()->isEntryBlock() &&
        isa<FuncOp, LLVM::LLVMFuncOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(
          op, "not replacing allocas in the entry block");

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, op.getType(),
                                                 op.getOperands());
    return success();
  }
};

struct HoistAllocasFromParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<Value> iterationCounts;
    op.walk([&](memref::AllocaOp alloca) {
      Operation *parent = alloca->getParentOp();
      for (; parent != nullptr; parent = parent->getParentOp()) {
        if (parent->hasTrait<OpTrait::AutomaticAllocationScope>())
          break;
      }
      if (parent != op)
        return WalkResult::advance();

      if (!changed) {
        iterationCounts = emitIterationCounts(rewriter, op);
        changed = true;
      }
      // TODO(zinenko): replace nullptr with a value of corresponding type.
      Value buffer = allocateTemporaryBuffer<memref::AllocaOp>(
          rewriter, nullptr, iterationCounts, /*alloca=*/true);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(alloca);
      for (Value iv : op.getInductionVars()) {
        auto type = buffer.getType().cast<MemRefType>();
        auto newType =
            MemRefType::get(type.getShape().drop_front(), type.getElementType(),
                            /*affineMaps=*/{}, type.getMemorySpace());
        buffer = rewriter.create<polygeist::SubIndexOp>(alloca.getLoc(),
                                                        newType, buffer, iv);
      }
      rewriter.replaceOp(alloca, buffer);

      return WalkResult::advance();
    });
    return success(changed);
  }
};

struct AllocaToAllocPass : public AllocaToAllocBase<AllocaToAllocPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<AllocaToAlloc>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
      signalPassFailure();
  }
};

struct HoistAllocasPass : public HoistAllocasBase<HoistAllocasPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<HoistAllocasFromParallel>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<OperationPass<FuncOp>> createAllocaToAllocPass() {
  return std::make_unique<AllocaToAllocPass>();
}

std::unique_ptr<OperationPass<FuncOp>> createHoistAllocasPass() {
  return std::make_unique<HoistAllocasPass>();
}
} // namespace polygeist
} // namespace mlir
