//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

static bool couldWrite(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects<MemoryEffects::Write>(localEffects);
    return localEffects.size() > 0;
  }
  return true;
}
/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findValuesUsedBelow(Operation *op,
                                llvm::SetVector<Value> &crossing) {
  llvm::SetVector<Operation *> descendantsUsed;

  // A set of pre-barrier operations which are potentially captured by a
  // subsequent pre-barrier operation.
  SmallVector<Operation *> Allocas;

  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<memref::AllocaOp, LLVM::AllocaOp>(it))
      Allocas.push_back(it);
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {

        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
        }
      }
    }
  }

  llvm::SmallVector<std::pair<Operation *, Operation *>> todo;
  for (auto A : Allocas)
    todo.emplace_back(A, A);

  llvm::SetVector<Operation *> preserveAllocas;

  std::map<Operation *, SmallPtrSet<Operation *, 2>> descendants;
  while (todo.size()) {
    auto current = todo.back();
    todo.pop_back();
    if (descendants[current.first].count(current.second))
      continue;
    descendants[current.first].insert(current.second);
    for (Value value : current.first->getResults()) {
      for (Operation *user : value.getUsers()) {
        Operation *origUser = user;
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (!op->isBeforeInBlock(user)) {
          if (couldWrite(origUser) ||
              origUser->hasTrait<OpTrait::IsTerminator>()) {
            preserveAllocas.insert(current.second);
          }
          if (!isa<LLVM::LoadOp, memref::LoadOp, AffineLoadOp>(origUser)) {
            for (auto res : origUser->getResults()) {
              if (crossing.contains(res)) {
                preserveAllocas.insert(current.second);
              }
            }
            todo.emplace_back(user, current.second);
          }
        }
      }
    }
  }
  for (auto op : preserveAllocas)
    crossing.insert(op->getResult(0));
}

/// Returns `true` if the given operation has a BarrierOp transitively nested in
/// one of its regions, but not within any nested ParallelOp.
static bool hasNestedBarrier(Operation *op, Operation *direct = nullptr) {
  auto result = op->walk([=](polygeist::BarrierOp barrier) {
    if (!direct || barrier->getParentOp() == direct) {
      // If there is a `parallel` op nested inside the given op (alternatively,
      // the `parallel` op is not an ancestor of `op` or `op` itself), the
      // barrier is considered nested in that `parallel` op and _not_ in `op`.
      auto parallel = barrier->getParentOfType<scf::ParallelOp>();
      if (!parallel->isAncestor(op))
        return WalkResult::skip();

      return WalkResult::interrupt();
    } else {
      return WalkResult::skip();
    }
  });
  return result.wasInterrupted();
}

namespace {
/// Replaces an conditional with a loop that may iterate 0 or 1 time, that is:
///
/// scf.if %cond {
///   @then()
/// } else {
///   @else()
/// }
///
/// is replaced with
///
/// scf.for %i = 0 to %cond step 1 {
///   @then()
/// }
/// scf.for %i = 0 to %cond - 1 step 1 {
///   @else()
/// }
struct ReplaceIfWithFors : public OpRewritePattern<scf::IfOp> {
  ReplaceIfWithFors(MLIRContext *ctx) : OpRewritePattern<scf::IfOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.getCondition().getType().isInteger(1));

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[if-to-for] no nested barrier\n");
      return failure();
    }

    SmallVector<Value, 8> forArgs;
    for (auto a : op.getResults()) {
      forArgs.push_back(
          rewriter.create<LLVM::UndefOp>(op.getLoc(), a.getType()));
    }

    Location loc = op.getLoc();
    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<ConstantIndexOp>(loc, 1);

    auto cond = rewriter.create<IndexCastOp>(
        loc, rewriter.getIndexType(),
        rewriter.create<ExtUIOp>(loc, op.getCondition(),
                                 mlir::IntegerType::get(one.getContext(), 64)));
    auto thenLoop = rewriter.create<scf::ForOp>(loc, zero, cond, one, forArgs);
    if (forArgs.size() == 0)
      rewriter.eraseOp(&thenLoop.getBody()->back());
    rewriter.mergeBlocks(op.getBody(0), thenLoop.getBody(0));

    SmallVector<Value> vals;

    if (!op.getElseRegion().empty()) {
      auto negCondition = rewriter.create<SubIOp>(loc, one, cond);
      scf::ForOp elseLoop =
          rewriter.create<scf::ForOp>(loc, zero, negCondition, one, forArgs);
      if (forArgs.size() == 0)
        rewriter.eraseOp(&elseLoop.getBody()->back());
      rewriter.mergeBlocks(op.getBody(1), elseLoop.getBody(0));

      for (auto tup : llvm::zip(thenLoop.getResults(), elseLoop.getResults())) {
        vals.push_back(rewriter.create<SelectOp>(op.getLoc(), op.getCondition(),
                                                 std::get<0>(tup),
                                                 std::get<1>(tup)));
      }
    }

    rewriter.replaceOp(op, vals);
    return success();
  }
};

/// Returns `true` if `value` is defined outside of the region that contains
/// `user`.
static bool isDefinedAbove(Value value, Operation *user) {
  return value.getParentRegion()->isProperAncestor(user->getParentRegion());
}

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ForOp op) {
  return isDefinedAbove(op.getLowerBound(), op) &&
         isDefinedAbove(op.getStep(), op);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bound and unit step.
struct NormalizeLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op) || !isa<scf::ParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] loop already normalized\n");
      return failure();
    }
    if (op.getNumResults()) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] not handling reduction loops\n");
      return failure();
    }

    OpBuilder::InsertPoint point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOp());
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    rewriter.restoreInsertionPoint(point);

    Value difference = rewriter.create<SubIOp>(op.getLoc(), op.getUpperBound(),
                                               op.getLowerBound());
    Value tripCount = rewriter.create<AddIOp>(
        op.getLoc(),
        rewriter.create<DivUIOp>(
            op.getLoc(), rewriter.create<SubIOp>(op.getLoc(), difference, one),
            op.getStep()),
        one);
    // rewriter.create<CeilDivSIOp>(op.getLoc(), difference, op.getStep());
    auto newForOp =
        rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value scaled = rewriter.create<MulIOp>(
        op.getLoc(), newForOp.getInductionVar(), op.getStep());
    Value iv = rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound(), scaled);
    rewriter.mergeBlockBefore(op.getBody(), &newForOp.getBody()->back(), {iv});
    rewriter.eraseOp(&newForOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ParallelOp op) {
  auto isZero = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isNullValue();
  };
  auto isOne = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isOneValue();
  };
  return llvm::all_of(op.getLowerBound(), isZero) &&
         llvm::all_of(op.getStep(), isOne);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bounds and unit steps.
struct NormalizeParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] loop already normalized\n");
      return failure();
    }
    if (op->getNumResults() != 0) {
      LLVM_DEBUG(
          DBGS() << "[normalize-parallel] not processing reduction loops\n");
      return failure();
    }
    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] no nested barrier\n");
      return failure();
    }

    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
    auto newOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), SmallVector<Value>(iterationCounts.size(), zero),
        iterationCounts, SmallVector<Value>(iterationCounts.size(), one));

    SmallVector<Value> inductionVars;
    inductionVars.reserve(iterationCounts.size());
    rewriter.setInsertionPointToStart(newOp.getBody());
    for (unsigned i = 0, e = iterationCounts.size(); i < e; ++i) {
      Value scaled = rewriter.create<MulIOp>(
          op.getLoc(), newOp.getInductionVars()[i], op.getStep()[i]);
      Value shifted =
          rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound()[i], scaled);
      inductionVars.push_back(shifted);
    }

    rewriter.mergeBlockBefore(op.getBody(), &newOp.getBody()->back(),
                              inductionVars);
    rewriter.eraseOp(&newOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Checks if `op` may need to be wrapped in a pair of barriers. This is a
/// necessary but insufficient condition.
static LogicalResult canWrapWithBarriers(Operation *op) {
  if (!op->getParentOfType<scf::ParallelOp>()) {
    LLVM_DEBUG(DBGS() << "[wrap] not nested in a pfor\n");
    return failure();
  }

  if (op->getNumResults() != 0) {
    LLVM_DEBUG(DBGS() << "[wrap] ignoring loop with reductions\n");
    return failure();
  }

  if (!hasNestedBarrier(op)) {
    LLVM_DEBUG(DBGS() << "[wrap] no nested barrier\n");
    return failure();
  }

  return success();
}

/// Puts a barrier before and/or after `op` if there isn't already one.
/// `extraPrevCheck` is called on the operation immediately preceding `op` and
/// can be used to look further upward if the immediately preceding operation is
/// not a barrier.
static LogicalResult wrapWithBarriers(
    Operation *op, PatternRewriter &rewriter,
    llvm::function_ref<bool(Operation *)> extraPrevCheck = nullptr) {
  Operation *prevOp = op->getPrevNode();
  Operation *nextOp = op->getNextNode();
  bool hasPrevBarrierLike =
      prevOp == nullptr || isa<polygeist::BarrierOp>(prevOp);
  if (extraPrevCheck && !hasPrevBarrierLike)
    hasPrevBarrierLike = extraPrevCheck(prevOp);
  bool hasNextBarrierLike =
      nextOp == &op->getBlock()->back() || isa<polygeist::BarrierOp>(nextOp);

  if (hasPrevBarrierLike && hasNextBarrierLike) {
    LLVM_DEBUG(DBGS() << "[wrap] already has sufficient barriers\n");
    return failure();
  }

  if (!hasPrevBarrierLike)
    rewriter.create<polygeist::BarrierOp>(op->getLoc());

  if (!hasNextBarrierLike) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    rewriter.create<polygeist::BarrierOp>(op->getLoc());
  }

  // We don't actually change the op, but the pattern infra wants us to. Just
  // pretend we changed it in-place.
  rewriter.updateRootInPlace(op, [] {});
  LLVM_DEBUG(DBGS() << "[wrap] wrapped '" << op->getName().getStringRef()
                    << "' with barriers\n");
  return success();
}

/// Puts a barrier before and/or after an "if" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.
struct WrapIfWithBarrier : public OpRewritePattern<scf::IfOp> {
  WrapIfWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::IfOp>(ctx) {}
  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(canWrapWithBarriers(op)))
      return failure();

    return wrapWithBarriers(op, rewriter, [&](Operation *prevOp) {
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (loadOp.result() == op.getCondition() &&
            loadOp.indices() ==
                cast<scf::ParallelOp>(op->getParentOp()).getInductionVars()) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isa<polygeist::BarrierOp>(prevOp);
        }
      }
      return false;
    });
  }
};

/// Puts a barrier before and/or after a "for" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.

struct WrapForWithBarrier : public OpRewritePattern<scf::ForOp> {
  WrapForWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::ForOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(canWrapWithBarriers(op)))
      return failure();

    if (!isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[wrap-for] non-normalized loop\n");
      return failure();
    }

    return wrapWithBarriers(op, rewriter, [&](Operation *prevOp) {
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (loadOp.result() == op.getUpperBound() &&
            loadOp.indices() ==
                cast<scf::ParallelOp>(op->getParentOp()).getInductionVars()) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isa<polygeist::BarrierOp>(prevOp);
        }
      }
      return false;
    });
  }
};

/// Puts a barrier before and/or after a "while" operation if there isn't
/// already one.
struct WrapWhileWithBarrier : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0 ||
        !llvm::hasSingleElement(op.getAfter().front())) {
      LLVM_DEBUG(DBGS() << "[wrap-while] ignoring non-rotated loop\n";);
      return failure();
    }

    if (failed(canWrapWithBarriers(op)))
      return failure();

    return wrapWithBarriers(op, rewriter);
  }
};

/// Moves the body from `ifOp` contained in `op` to a parallel op newly
/// created at the start of `newIf`.
static void moveBodies(PatternRewriter &rewriter, scf::ParallelOp op,
                       scf::IfOp ifOp, scf::IfOp newIf) {
  rewriter.startRootUpdate(op);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newIf.thenBlock());
    auto newParallel = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());

    for (auto tup :
         llvm::zip(newParallel.getInductionVars(), op.getInductionVars())) {
      std::get<1>(tup).replaceUsesWithIf(std::get<0>(tup), [&](OpOperand &op) {
        return ifOp.getThenRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
    }

    rewriter.mergeBlockBefore(ifOp.thenBlock(), &newParallel.getBody()->back());
    rewriter.eraseOp(&newParallel.getBody()->back());
  }

  if (ifOp.getElseRegion().getBlocks().size() > 0) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newIf.elseBlock());
    auto newParallel = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());

    for (auto tup :
         llvm::zip(newParallel.getInductionVars(), op.getInductionVars())) {
      std::get<1>(tup).replaceUsesWithIf(std::get<0>(tup), [&](OpOperand &op) {
        return ifOp.getElseRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
    }
    rewriter.mergeBlockBefore(ifOp.elseBlock(), &newParallel.getBody()->back());
    rewriter.eraseOp(&newParallel.getBody()->back());
  }

  rewriter.eraseOp(ifOp);
  rewriter.eraseOp(op);
  rewriter.finalizeRootUpdate(op);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.
struct InterchangeIfPFor : public OpRewritePattern<scf::ParallelOp> {
  InterchangeIfPFor(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: an "if"
    // and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::IfOp>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(if) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto ifOp = cast<scf::IfOp>(op.getBody()->front());
    if (op.getNumResults() != 0 || ifOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    if (!hasNestedBarrier(ifOp)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
      return failure();
    }

    auto newIf = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), TypeRange(), ifOp.getCondition(),
        ifOp.getElseRegion().getBlocks().size() > 0);
    moveBodies(rewriter, op, ifOp, newIf);
    return success();
  }
};

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
struct InterchangeIfPForLoad : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 2) == op.getBody()->end() ||
        std::next(op.getBody()->begin(), 3) != op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] expected two nested ops\n");
      return failure();
    }
    auto loadOp = dyn_cast<memref::LoadOp>(op.getBody()->front());
    auto ifOp = dyn_cast<scf::IfOp>(op.getBody()->front().getNextNode());
    if (!loadOp || !ifOp || loadOp.result() != ifOp.getCondition() ||
        loadOp.indices() != op.getInductionVars()) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] expected pfor(load, for/if)");
      return failure();
    }

    if (!hasNestedBarrier(ifOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] no nested barrier\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.
    Value zero = rewriter.create<ConstantIndexOp>(ifOp.getLoc(), 0);
    Value condition = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(),
        SmallVector<Value>(loadOp.getMemRefType().getRank(), zero));

    auto newIf =
        rewriter.create<scf::IfOp>(ifOp.getLoc(), TypeRange(), condition,
                                   ifOp.getElseRegion().getBlocks().size() > 0);
    moveBodies(rewriter, op, ifOp, newIf);
    return success();
  }
};

/// Moves the body from `forLoop` contained in `op` to a parallel op newly
/// created at the start of `newForLoop`.
static void moveBodies(PatternRewriter &rewriter, scf::ParallelOp op,
                       scf::ForOp forLoop, scf::ForOp newForLoop) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newForLoop.getBody());
  auto newParallel = rewriter.create<scf::ParallelOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());

  // Merge in two stages so we can properly replace uses of two induction
  // varibales defined in different blocks.
  rewriter.mergeBlockBefore(op.getBody(), &newParallel.getBody()->back(),
                            newParallel.getInductionVars());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.mergeBlockBefore(forLoop.getBody(), &newParallel.getBody()->back(),
                            newForLoop.getInductionVar());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.eraseOp(op);
  rewriter.eraseOp(forLoop);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.
struct InterchangeForPFor : public OpRewritePattern<scf::ParallelOp> {
  InterchangeForPFor(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "for"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::ForOp>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(for) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto forLoop = cast<scf::ForOp>(op.getBody()->front());
    if (op.getNumResults() != 0 || forLoop.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    if (!isNormalized(op) || !isNormalized(forLoop)) {
      LLVM_DEBUG(DBGS() << "[interchange] non-normalized loop\n");
    }

    if (!hasNestedBarrier(forLoop)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
      return failure();
    }

    auto newForLoop =
        rewriter.create<scf::ForOp>(forLoop.getLoc(), forLoop.getLowerBound(),
                                    forLoop.getUpperBound(), forLoop.getStep());
    moveBodies(rewriter, op, forLoop, newForLoop);
    return success();
  }
};

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
struct InterchangeForPForLoad : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 2) == op.getBody()->end() ||
        std::next(op.getBody()->begin(), 3) != op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected two nested ops\n");
      return failure();
    }
    auto loadOp = dyn_cast<memref::LoadOp>(op.getBody()->front());
    auto forOp = dyn_cast<scf::ForOp>(op.getBody()->front().getNextNode());
    if (!loadOp || !forOp || loadOp.result() != forOp.getUpperBound() ||
        loadOp.indices() != op.getInductionVars()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected pfor(load, for/if)");
      return failure();
    }

    if (!isNormalized(op) || !isNormalized(forOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-load] non-normalized loop\n");
      return failure();
    }

    if (!hasNestedBarrier(forOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-load] no nested barrier\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.
    Value zero = rewriter.create<ConstantIndexOp>(forOp.getLoc(), 0);
    Value tripCount = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(),
        SmallVector<Value>(loadOp.getMemRefType().getRank(), zero));
    auto newForLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), tripCount, forOp.getStep());

    moveBodies(rewriter, op, forOp, newForLoop);
    return success();
  }
};

/// Returns the insertion point (as block pointer and itertor in it) immediately
/// after the definition of `v`.
static std::pair<Block *, Block::iterator> getInsertionPointAfterDef(Value v) {
  if (Operation *op = v.getDefiningOp())
    return {op->getBlock(), std::next(Block::iterator(op))};

  BlockArgument blockArg = v.cast<BlockArgument>();
  return {blockArg.getParentBlock(), blockArg.getParentBlock()->begin()};
}

/// Returns the insertion point that post-dominates `first` and `second`.
static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    const std::pair<Block *, Block::iterator> &first,
    const std::pair<Block *, Block::iterator> &second,
    const PostDominanceInfo &postDominanceInfo) {
  // Same block, take the last op.
  if (first.first == second.first)
    return first.second->isBeforeInBlock(&*second.second) ? second : first;

  // Same region, use "normal" dominance analysis.
  if (first.first->getParent() == second.first->getParent()) {
    Block *block =
        postDominanceInfo.findNearestCommonDominator(first.first, second.first);
    assert(block);
    if (block == first.first)
      return first;
    if (block == second.first)
      return second;
    return {block, block->begin()};
  }

  if (first.first->getParent()->isAncestor(second.first->getParent()))
    return second;

  assert(second.first->getParent()->isAncestor(first.first->getParent()) &&
         "expected values to be defined in nested regions");
  return first;
}

/// Returns the insertion point that post-dominates all `values`.
static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

/// Interchanges a parallel for loop with a while loop it contains. The while
/// loop is expected to have an empty "after" region.
struct InterchangeWhilePFor : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "while"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::WhileOp>(op.getBody()->front())) {
      LLVM_DEBUG(
          DBGS() << "[interchange-while] not a perfect pfor(while) nest\n");
      return failure();
    }

    auto whileOp = cast<scf::WhileOp>(op.getBody()->front());
    if (whileOp.getNumOperands() != 0 || whileOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange-while] loop-carried values\n");
      return failure();
    }
    if (!llvm::hasSingleElement(whileOp.getAfter().front()) ||
        !isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] non-normalized loop\n");
      return failure();
    }
    if (!hasNestedBarrier(whileOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] no nested barrier\n");
      return failure();
    }

    auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(),
                                                    TypeRange(), ValueRange());
    rewriter.createBlock(&newWhileOp.getAfter());
    rewriter.clone(whileOp.getAfter().front().back());

    rewriter.createBlock(&newWhileOp.getBefore());
    auto newParallelOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());

    auto conditionOp =
        cast<scf::ConditionOp>(whileOp.getBefore().front().back());
    rewriter.mergeBlockBefore(op.getBody(), &newParallelOp.getBody()->back(),
                              newParallelOp.getInductionVars());
    rewriter.eraseOp(newParallelOp.getBody()->back().getPrevNode());
    rewriter.mergeBlockBefore(&whileOp.getBefore().front(),
                              &newParallelOp.getBody()->back());

    Operation *conditionDefiningOp = conditionOp.getCondition().getDefiningOp();
    if (conditionDefiningOp &&
        !isDefinedAbove(conditionOp.getCondition(), conditionOp)) {
      rewriter.setInsertionPoint(newParallelOp);
      SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
      Value allocated = rewriter.create<memref::AllocaOp>(
          conditionDefiningOp->getLoc(),
          MemRefType::get({}, rewriter.getI1Type()));
      rewriter.setInsertionPointAfter(conditionDefiningOp);
      Value cond = rewriter.create<ConstantIntOp>(conditionDefiningOp->getLoc(),
                                                  true, 1);
      for (auto tup : llvm::zip(newParallelOp.getLowerBound(),
                                newParallelOp.getInductionVars())) {
        cond = rewriter.create<AndIOp>(
            conditionDefiningOp->getLoc(),
            rewriter.create<CmpIOp>(conditionDefiningOp->getLoc(),
                                    CmpIPredicate::eq, std::get<0>(tup),
                                    std::get<1>(tup)),
            cond);
      }
      auto ifOp =
          rewriter.create<scf::IfOp>(conditionDefiningOp->getLoc(), cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      rewriter.create<memref::StoreOp>(conditionDefiningOp->getLoc(),
                                       conditionOp.getCondition(), allocated);

      rewriter.setInsertionPointToEnd(&newWhileOp.getBefore().front());

      Value reloaded = rewriter.create<memref::LoadOp>(
          conditionDefiningOp->getLoc(), allocated);
      rewriter.create<scf::ConditionOp>(conditionOp.getLoc(), reloaded,
                                        ValueRange());
      rewriter.eraseOp(conditionOp);
    }

    rewriter.eraseOp(whileOp);
    rewriter.eraseOp(op);

    return success();
  }
};

/// Moves the "after" region of a while loop into its "before" region using a
/// conditional, that is
///
/// scf.while {
///   @before()
///   scf.conditional(%cond)
/// } do {
///   @after()
///   scf.yield
/// }
///
/// is transformed into
///
/// scf.while {
///   @before()
///   scf.if (%cond) {
///     @after()
///   }
///   scf.conditional(%cond)
/// } do {
///   scf.yield
/// }
struct RotateWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::hasSingleElement(op.getAfter().front())) {
      LLVM_DEBUG(DBGS() << "[rotate-while] the after region is empty");
      return failure();
    }
    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[rotate-while] no nested barrier\n");
      return failure();
    }
    if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[rotate-while] loop-carried values\n");
      return failure();
    }

    auto condition = cast<scf::ConditionOp>(op.getBefore().front().back());
    rewriter.setInsertionPoint(condition);
    auto conditional =
        rewriter.create<scf::IfOp>(op.getLoc(), condition.getCondition());
    rewriter.mergeBlockBefore(&op.getAfter().front(),
                              &conditional.getBody()->back());
    rewriter.eraseOp(&conditional.getBody()->back());

    rewriter.createBlock(&op.getAfter());
    rewriter.clone(conditional.getBody()->back());

    LLVM_DEBUG(DBGS() << "[rotate-while] done\n");
    return success();
  }
};

/// Splits a parallel loop around the first barrier it immediately contains.
/// Values defined before the barrier are stored in newly allocated buffers and
/// loaded back when needed.
struct DistributeAroundBarrier : public OpRewritePattern<scf::ParallelOp> {
  DistributeAroundBarrier(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[distribute] not matching reduction loops\n");
      return failure();
    }

    if (!isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[distribute] non-normalized loop\n");
      return failure();
    }

    Operation *barrier;
    {
      auto it =
          llvm::find_if(op.getBody()->getOperations(), [](Operation &nested) {
            return isa<polygeist::BarrierOp>(nested);
          });
      if (it == op.getBody()->end()) {
        LLVM_DEBUG(DBGS() << "[distribute] no barrier in the loop\n");
        return failure();
      }
      barrier = &*it;
    }

    llvm::SetVector<Value> crossing;
    findValuesUsedBelow(barrier, crossing);
    rewriter.setInsertionPoint(op);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);

    // Allocate space for values crossing the barrier.
    SmallVector<Value> allocations;
    allocations.reserve(crossing.size());
    auto mod = barrier->getParentOfType<ModuleOp>();
    DataLayout DLI(mod);
    for (Value v : crossing) {
      if (auto ao = v.getDefiningOp<LLVM::AllocaOp>()) {
        allocations.push_back(allocateTemporaryBuffer<LLVM::CallOp>(
                                  rewriter, v, iterationCounts, true, &DLI)
                                  .getResult(0));
      } else {
        allocations.push_back(allocateTemporaryBuffer<memref::AllocOp>(
            rewriter, v, iterationCounts));
      }
    }

    // Store values crossing the barrier in caches immediately when ready.
    for (auto pair : llvm::zip(crossing, allocations)) {
      Value v = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      if (auto ao = v.getDefiningOp<memref::AllocaOp>()) {
        for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
          rewriter.setInsertionPoint(u.getOwner());
          auto buf = alloc;
          for (auto idx : op.getInductionVars()) {
            auto mt0 = buf.getType().cast<MemRefType>();
            std::vector<int64_t> shape(mt0.getShape());
            shape.erase(shape.begin());
            auto mt = MemRefType::get(shape, mt0.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      // mt0.getLayout(),
                                      mt0.getMemorySpace());
            auto subidx = rewriter.create<polygeist::SubIndexOp>(alloc.getLoc(),
                                                                 mt, buf, idx);
            buf = subidx;
          }
          u.set(buf);
        }
        rewriter.eraseOp(ao);
      } else if (auto ao = v.getDefiningOp<LLVM::AllocaOp>()) {
        Value sz = ao.getArraySize();
        rewriter.setInsertionPointAfter(alloc.getDefiningOp());
        alloc =
            rewriter.create<LLVM::BitcastOp>(ao.getLoc(), ao.getType(), alloc);
        for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
          rewriter.setInsertionPoint(u.getOwner());
          Value idx = nullptr;
          // i0
          // i0 * s1 + i1
          // ( i0 * s1 + i1 ) * s2 + i2
          for (auto pair : llvm::zip(iterationCounts, op.getInductionVars())) {
            if (idx) {
              idx = rewriter.create<arith::MulIOp>(ao.getLoc(), idx,
                                                   std::get<0>(pair));
              idx = rewriter.create<arith::AddIOp>(ao.getLoc(), idx,
                                                   std::get<1>(pair));
            } else
              idx = std::get<1>(pair);
          }
          idx = rewriter.create<MulIOp>(ao.getLoc(), sz,
                                        rewriter.create<arith::IndexCastOp>(
                                            ao.getLoc(), idx, sz.getType()));
          SmallVector<Value> vec = {idx};
          u.set(rewriter.create<LLVM::GEPOp>(ao.getLoc(), ao.getType(), alloc,
                                             idx));
        }
      } else {

        for (auto &u : llvm::make_early_inc_range(v.getUses())) {
          auto user = u.getOwner();
          while (user->getBlock() != barrier->getBlock())
            user = user->getBlock()->getParentOp();

          if (barrier->isBeforeInBlock(user)) {
            rewriter.setInsertionPoint(u.getOwner());
            Value reloaded = rewriter.create<memref::LoadOp>(
                user->getLoc(), alloc, op.getInductionVars());
            rewriter.startRootUpdate(user);
            u.set(reloaded);
            rewriter.finalizeRootUpdate(user);
          }
        }
        rewriter.setInsertionPointAfter(v.getDefiningOp());
        rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                         op.getInductionVars());
      }
    }

    // Insert the terminator for the new loop immediately before the barrier.
    rewriter.setInsertionPoint(barrier);
    rewriter.create<scf::YieldOp>(op.getBody()->back().getLoc());

    // Create the second loop.
    rewriter.setInsertionPointAfter(op);
    auto newLoop = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());
    rewriter.eraseOp(&newLoop.getBody()->back());

    auto freefn = GetOrCreateFreeFunction(mod);
    for (auto alloc : allocations) {
      if (alloc.getType().isa<LLVM::LLVMPointerType>()) {
        Value args[1] = {alloc};
        rewriter.create<LLVM::CallOp>(alloc.getLoc(), freefn, args);
      } else
        rewriter.create<memref::DeallocOp>(alloc.getLoc(), alloc);
    }

    // Recreate the operations in the new loop with new values.
    rewriter.setInsertionPointToStart(newLoop.getBody());
    BlockAndValueMapping mapping;
    mapping.map(op.getInductionVars(), newLoop.getInductionVars());
    SmallVector<Operation *> toDelete;
    toDelete.push_back(barrier);
    for (Operation *o = barrier->getNextNode(); o != nullptr;
         o = o->getNextNode()) {
      rewriter.clone(*o, mapping);
      toDelete.push_back(o);
    }

    // Erase original operations and the barrier.
    for (Operation *o : llvm::reverse(toDelete))
      rewriter.eraseOp(o);

    for (auto ao : allocations)
      if (ao.getDefiningOp<LLVM::AllocaOp>() ||
          ao.getDefiningOp<memref::AllocaOp>())
        rewriter.eraseOp(ao.getDefiningOp());

    LLVM_DEBUG(DBGS() << "[distribute] distributed arround a barrier\n");
    return success();
  }
};

static void loadValues(Location loc, ArrayRef<Value> pointers,
                       PatternRewriter &rewriter,
                       SmallVectorImpl<Value> &loaded) {
  loaded.reserve(loaded.size() + pointers.size());
  for (Value alloc : pointers)
    loaded.push_back(rewriter.create<memref::LoadOp>(loc, alloc, ValueRange()));
}

struct Reg2MemFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasIterOperands() || !hasNestedBarrier(op))
      return failure();

    SmallVector<Value> allocated;
    allocated.reserve(op.getNumIterOperands());
    for (Value operand : op.getIterOperands()) {
      Value alloc = rewriter.create<memref::AllocaOp>(
          op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), operand.getType()),
          ValueRange());
      allocated.push_back(alloc);
      rewriter.create<memref::StoreOp>(op.getLoc(), operand, alloc,
                                       ValueRange());
    }

    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                             op.getUpperBound(), op.getStep());
    rewriter.setInsertionPointToStart(newOp.getBody());
    SmallVector<Value> newRegionArguments;
    newRegionArguments.push_back(newOp.getInductionVar());
    loadValues(op.getLoc(), allocated, rewriter, newRegionArguments);

    auto oldTerminator = cast<scf::YieldOp>(op.getBody()->getTerminator());
    rewriter.mergeBlockBefore(op.getBody(), newOp.getBody()->getTerminator(),
                              newRegionArguments);

    rewriter.setInsertionPoint(newOp.getBody()->getTerminator());
    for (auto en : llvm::enumerate(oldTerminator.getResults())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }
    rewriter.eraseOp(oldTerminator);

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    for (Value alloc : allocated) {
      loaded.push_back(
          rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange()));
    }
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

struct Reg2MemIf : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResults().size() || !hasNestedBarrier(op))
      return failure();

    SmallVector<Value> allocated;
    allocated.reserve(op.getNumResults());
    for (Type opType : op.getResultTypes()) {
      Value alloc = rewriter.create<memref::AllocaOp>(
          op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), opType),
          ValueRange());
      allocated.push_back(alloc);
    }

    auto newOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(),
                                            op.getCondition(), true);

    rewriter.setInsertionPoint(op.thenYield());
    for (auto en : llvm::enumerate(op.thenYield().getOperands())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }
    op.thenYield()->setOperands(ValueRange());

    rewriter.setInsertionPoint(op.elseYield());
    for (auto en : llvm::enumerate(op.elseYield().getOperands())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }
    op.elseYield()->setOperands(ValueRange());

    rewriter.eraseOp(&newOp.thenBlock()->back());
    rewriter.mergeBlocks(op.thenBlock(), newOp.thenBlock());

    rewriter.eraseOp(&newOp.elseBlock()->back());
    rewriter.mergeBlocks(op.elseBlock(), newOp.elseBlock());

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    for (Value alloc : allocated) {
      loaded.push_back(
          rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange()));
    }
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

static void storeValues(Location loc, ValueRange values, ValueRange pointers,
                        PatternRewriter &rewriter) {
  for (auto pair : llvm::zip(values, pointers)) {
    rewriter.create<memref::StoreOp>(loc, std::get<0>(pair), std::get<1>(pair),
                                     ValueRange());
  }
}

static void allocaValues(Location loc, ValueRange values,
                         PatternRewriter &rewriter,
                         SmallVector<Value> &allocated) {
  allocated.reserve(values.size());
  for (Value value : values) {
    Value alloc = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get(ArrayRef<int64_t>(), value.getType()),
        ValueRange());
    allocated.push_back(alloc);
  }
}

struct Reg2MemWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0 && op.getNumResults() == 0)
      return failure();
    if (!hasNestedBarrier(op))
      return failure();

    // Value stackPtr = rewriter.create<LLVM::StackSaveOp>(
    //     op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    SmallVector<Value> beforeAllocated, afterAllocated;
    allocaValues(op.getLoc(), op.getOperands(), rewriter, beforeAllocated);
    storeValues(op.getLoc(), op.getOperands(), beforeAllocated, rewriter);
    allocaValues(op.getLoc(), op.getResults(), rewriter, afterAllocated);

    auto newOp =
        rewriter.create<scf::WhileOp>(op.getLoc(), TypeRange(), ValueRange());
    Block *newBefore =
        rewriter.createBlock(&newOp.getBefore(), newOp.getBefore().begin());
    SmallVector<Value> newBeforeArguments;
    loadValues(op.getLoc(), beforeAllocated, rewriter, newBeforeArguments);
    rewriter.mergeBlocks(&op.getBefore().front(), newBefore,
                         newBeforeArguments);

    auto beforeTerminator =
        cast<scf::ConditionOp>(newOp.getBefore().front().getTerminator());
    rewriter.setInsertionPoint(beforeTerminator);
    storeValues(op.getLoc(), beforeTerminator.getArgs(), afterAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        beforeTerminator, [&] { beforeTerminator.getArgsMutable().clear(); });

    Block *newAfter =
        rewriter.createBlock(&newOp.getAfter(), newOp.getAfter().begin());
    SmallVector<Value> newAfterArguments;
    loadValues(op.getLoc(), afterAllocated, rewriter, newAfterArguments);
    rewriter.mergeBlocks(&op.getAfter().front(), newAfter, newAfterArguments);

    auto afterTerminator =
        cast<scf::YieldOp>(newOp.getAfter().front().getTerminator());
    rewriter.setInsertionPoint(afterTerminator);
    storeValues(op.getLoc(), afterTerminator.getResults(), beforeAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        afterTerminator, [&] { afterTerminator.getResultsMutable().clear(); });

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> results;
    loadValues(op.getLoc(), afterAllocated, rewriter, results);
    // rewriter.create<LLVM::StackRestoreOp>(op.getLoc(), stackPtr);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  CPUifyPass() = default;
  CPUifyPass(StringRef method) { this->method.setValue(method.str()); }
  void runOnFunction() override {
    if (method == "distribute") {
      OwningRewritePatternList patterns(&getContext());
      patterns
          .insert<Reg2MemFor, Reg2MemWhile, Reg2MemIf,
                  // ReplaceIfWithFors,
                  WrapForWithBarrier, WrapIfWithBarrier, WrapWhileWithBarrier,
                  InterchangeForPFor, InterchangeForPForLoad, InterchangeIfPFor,
                  InterchangeIfPForLoad, InterchangeWhilePFor, NormalizeLoop,
                  NormalizeParallel, RotateWhile, DistributeAroundBarrier>(
              &getContext());
      GreedyRewriteConfig config;
      config.maxIterations = 142;
      if (failed(applyPatternsAndFoldGreedily(getFunction(),
                                              std::move(patterns), config)))
        signalPassFailure();
    } else if (method == "omp") {
      SmallVector<polygeist::BarrierOp> toReplace;
      getFunction().walk(
          [&](polygeist::BarrierOp b) { toReplace.push_back(b); });
      for (auto b : toReplace) {
        OpBuilder Builder(b);
        Builder.create<omp::BarrierOp>(b.getLoc());
        b->erase();
      }
    } else {
      llvm::errs() << "unknown cpuify type: " << method << "\n";
      llvm_unreachable("unknown cpuify type");
    }
  }
};

} // end namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createCPUifyPass(StringRef str) {
  return std::make_unique<CPUifyPass>(str);
}
} // namespace polygeist
} // namespace mlir
