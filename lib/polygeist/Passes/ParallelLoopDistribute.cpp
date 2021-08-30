//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

#include "polygeist/Ops.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Passes/Passes.h"

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findValuesUsedBelow(Operation *op,
                                llvm::SetVector<Value> &crossing) {
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {
        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
  }

  // No need to process block arguments, they are assumed to be induction
  // variables and will be replicated.
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
    assert(op.condition().getType().isInteger(1));

    // TODO: we can do this by having "undef" values as inputs, or do reg2mem.
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[if-to-for] 'if' with results, need reg2mem\n";
                 DBGS() << op);
      return failure();
    }

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[if-to-for] no nested barrier\n");
      return failure();
    }

    Location loc = op.getLoc();
    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<ConstantIndexOp>(loc, 1);

    auto cond = rewriter.create<IndexCastOp>(
        loc, rewriter.getIndexType(),
        rewriter.create<ZeroExtendIOp>(
            loc, op.condition(), mlir::IntegerType::get(one.getContext(), 64)));
    auto thenLoop = rewriter.create<scf::ForOp>(loc, zero, cond, one);
    rewriter.mergeBlockBefore(op.getBody(0), &thenLoop.getBody()->back());
    rewriter.eraseOp(&thenLoop.getBody()->back());

    if (!op.elseRegion().empty()) {
      auto negCondition = rewriter.create<SubIOp>(loc, one, cond);
      auto elseLoop = rewriter.create<scf::ForOp>(loc, zero, negCondition, one);
      rewriter.mergeBlockBefore(op.getBody(1), &elseLoop.getBody()->back());
      rewriter.eraseOp(&elseLoop.getBody()->back());
    }

    rewriter.eraseOp(op);
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
  return isDefinedAbove(op.lowerBound(), op) && isDefinedAbove(op.step(), op);
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

    Value difference =
        rewriter.create<SubIOp>(op.getLoc(), op.upperBound(), op.lowerBound());
    Value tripCount =
        rewriter.create<SignedCeilDivIOp>(op.getLoc(), difference, op.step());
    auto newForOp =
        rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value scaled = rewriter.create<MulIOp>(
        op.getLoc(), newForOp.getInductionVar(), op.step());
    Value iv = rewriter.create<AddIOp>(op.getLoc(), op.lowerBound(), scaled);
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
  return llvm::all_of(op.lowerBound(), isZero) &&
         llvm::all_of(op.step(), isOne);
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
          op.getLoc(), newOp.getInductionVars()[i], op.step()[i]);
      Value shifted =
          rewriter.create<AddIOp>(op.getLoc(), op.lowerBound()[i], scaled);
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
        if (loadOp.result() == op.upperBound() &&
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
        !llvm::hasSingleElement(op.after().front())) {
      LLVM_DEBUG(DBGS() << "[wrap-while] ignoring non-rotated loop\n";);
      return failure();
    }

    if (failed(canWrapWithBarriers(op)))
      return failure();

    return wrapWithBarriers(op, rewriter);
  }
};

/// Moves the body from `forLoop` contained in `op` to a parallel op newly
/// created at the start of `newForLoop`.
static void moveBodies(PatternRewriter &rewriter, scf::ParallelOp op,
                       scf::ForOp forLoop, scf::ForOp newForLoop) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newForLoop.getBody());
  auto newParallel = rewriter.create<scf::ParallelOp>(
      op.getLoc(), op.lowerBound(), op.upperBound(), op.step());

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
        rewriter.create<scf::ForOp>(forLoop.getLoc(), forLoop.lowerBound(),
                                    forLoop.upperBound(), forLoop.step());
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
    if (!loadOp || !forOp || loadOp.result() != forOp.upperBound() ||
        loadOp.indices() != op.getInductionVars()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected pfor(load, for)");
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
        forOp.getLoc(), forOp.lowerBound(), tripCount, forOp.step());

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

static std::pair<Block *, Block::iterator>
findInsertionPointAfterLoopOperands(scf::ParallelOp op) {
  // Find the earliest insertion point where loop bounds are fully defined.
  PostDominanceInfo postDominanceInfo(op->getParentOfType<FuncOp>());
  SmallVector<Value> operands;
  llvm::append_range(operands, op.lowerBound());
  llvm::append_range(operands, op.upperBound());
  llvm::append_range(operands, op.step());
  return findNearestPostDominatingInsertionPoint(operands, postDominanceInfo);
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
    if (!llvm::hasSingleElement(whileOp.after().front()) || !isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] non-normalized loop\n");
      return failure();
    }
    if (!hasNestedBarrier(whileOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] no nested barrier\n");
      return failure();
    }

    auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(),
                                                    TypeRange(), ValueRange());
    rewriter.createBlock(&newWhileOp.after());
    rewriter.clone(whileOp.after().front().back());

    rewriter.createBlock(&newWhileOp.before());
    auto newParallelOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.lowerBound(), op.upperBound(), op.step());

    auto conditionOp = cast<scf::ConditionOp>(whileOp.before().front().back());
    rewriter.mergeBlockBefore(op.getBody(), &newParallelOp.getBody()->back(),
                              newParallelOp.getInductionVars());
    rewriter.eraseOp(newParallelOp.getBody()->back().getPrevNode());
    rewriter.mergeBlockBefore(&whileOp.before().front(),
                              &newParallelOp.getBody()->back());

    Operation *conditionDefiningOp = conditionOp.condition().getDefiningOp();
    if (conditionDefiningOp &&
        !isDefinedAbove(conditionOp.condition(), conditionOp)) {
      std::pair<Block *, Block::iterator> insertionPoint =
          findInsertionPointAfterLoopOperands(op);
      rewriter.setInsertionPoint(insertionPoint.first, insertionPoint.second);
      SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
      Value allocated = allocateTemporaryBuffer<memref::AllocaOp>(
          rewriter, conditionOp.condition(), iterationCounts);
      Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);

      rewriter.setInsertionPointAfter(conditionDefiningOp);
      rewriter.create<memref::StoreOp>(conditionDefiningOp->getLoc(),
                                       conditionOp.condition(), allocated,
                                       newParallelOp.getInductionVars());

      rewriter.setInsertionPointToEnd(&newWhileOp.before().front());
      SmallVector<Value> zeros(iterationCounts.size(), zero);
      Value reloaded = rewriter.create<memref::LoadOp>(
          conditionDefiningOp->getLoc(), allocated, zeros);
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
    if (llvm::hasSingleElement(op.after().front())) {
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

    auto condition = cast<scf::ConditionOp>(op.before().front().back());
    rewriter.setInsertionPoint(condition);
    auto conditional =
        rewriter.create<scf::IfOp>(op.getLoc(), condition.condition());
    rewriter.mergeBlockBefore(&op.after().front(),
                              &conditional.getBody()->back());
    rewriter.eraseOp(&conditional.getBody()->back());

    rewriter.createBlock(&op.after());
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

    Operation* barrier;
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
    //std::pair<Block *, Block::iterator> insertPoint =
    //    findInsertionPointAfterLoopOperands(op);
    //rewriter.setInsertionPoint(insertPoint.first, insertPoint.second);
    rewriter.setInsertionPoint(op);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);

    // Allocate space for values crossing the barrier.
    SmallVector<memref::AllocOp> allocations;
    allocations.reserve(crossing.size());
    for (Value v : crossing) {
      allocations.push_back(allocateTemporaryBuffer<memref::AllocOp>(rewriter, v, iterationCounts));
    }

    // Store values crossing the barrier in caches immediately when ready.
    for (auto pair : llvm::zip(crossing, allocations)) {
      Value v = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      rewriter.setInsertionPointAfter(v.getDefiningOp());
      if (auto ao = v.getDefiningOp<memref::AllocaOp>()) {
          for (auto& u : llvm::make_early_inc_range(ao.getResult().getUses())) {
            rewriter.setInsertionPoint(u.getOwner());
              auto buf = alloc;
              for (auto idx : op.getInductionVars()) {
                  auto mt0 = buf.getType().cast<MemRefType>();
                  std::vector<int64_t> shape(mt0.getShape());
                  shape.erase(shape.begin());
                  auto mt = MemRefType::get(shape, mt0.getElementType(),
                                         mt0.getAffineMaps(), mt0.getMemorySpace());
                  auto subidx = rewriter.create<polygeist::SubIndexOp>(alloc.getLoc(), mt, buf, idx);
                  buf = subidx;
              }
              u.set(buf);
          }
      }
      else
        rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                         op.getInductionVars());
    }

    // Insert the terminator for the new loop immediately before the barrier.
    rewriter.setInsertionPoint(barrier);
    rewriter.create<scf::YieldOp>(op.getBody()->back().getLoc());

    // Create the second loop.
    rewriter.setInsertionPointAfter(op);
    auto newLoop = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.lowerBound(), op.upperBound(), op.step());
    rewriter.eraseOp(&newLoop.getBody()->back());

    for (auto alloc : allocations)
        rewriter.create<memref::DeallocOp>(alloc.getLoc(), alloc);

    // Recreate the operations in the new loop with new values.
    rewriter.setInsertionPointToStart(newLoop.getBody());
    BlockAndValueMapping mapping;
    mapping.map(op.getInductionVars(), newLoop.getInductionVars());
    SmallVector<Operation *> toDelete;
    toDelete.push_back(barrier);
    for (Operation *o = barrier->getNextNode(); o != nullptr; o = o->getNextNode()) {
      rewriter.clone(*o, mapping);
      toDelete.push_back(o);
    }

    // Erase original operations and the barrier.
    for (Operation *o : llvm::reverse(toDelete))
      rewriter.eraseOp(o);

    // Replace uses of values defined above the barrier (now, in a different
    // loop) with fresh loads from scratchpad. This may not be the most
    // efficient IR, but this avoids creating new crossing values for the
    // following barriers as opposed to putting loads at the start of the new
    // loop. We expect mem2reg and repeated load elimitation to improve the IR.
    newLoop.getBody()->walk([&](Operation *nested) {
      for (OpOperand &operand : nested->getOpOperands()) {
        auto it = llvm::find(crossing, operand.get());
        if (it == crossing.end())
          continue;

        size_t pos = std::distance(crossing.begin(), it);
        rewriter.setInsertionPoint(nested);

        Value reloaded = rewriter.create<memref::LoadOp>(
              operand.getOwner()->getLoc(), allocations[pos],
              newLoop.getInductionVars());
        rewriter.startRootUpdate(nested);
        operand.set(reloaded);
        rewriter.finalizeRootUpdate(nested);
      }
    });

    LLVM_DEBUG(DBGS() << "[distribute] distributed arround a barrier\n");
    return success();
  }
};

static void loadValues(Location loc, ArrayRef<Value> pointers, Value zero,
                       PatternRewriter &rewriter,
                       SmallVectorImpl<Value> &loaded) {
  loaded.reserve(loaded.size() + pointers.size());
  for (Value alloc : pointers)
    loaded.push_back(rewriter.create<memref::LoadOp>(loc, alloc, zero));
}

struct Reg2MemFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasIterOperands() || !hasNestedBarrier(op))
      return failure();

    // Value stackPtr = rewriter.create<LLVM::StackSaveOp>(
    //    op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value> allocated;
    allocated.reserve(op.getNumIterOperands());
    for (Value operand : op.getIterOperands()) {
      Value alloc = rewriter.create<memref::AllocOp>(
          op.getLoc(), MemRefType::get(1, operand.getType()), ValueRange());
      allocated.push_back(alloc);

      rewriter.create<memref::StoreOp>(op.getLoc(), operand, alloc, zero);
    }

    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), op.lowerBound(),
                                             op.upperBound(), op.step());
    rewriter.setInsertionPointToStart(newOp.getBody());
    SmallVector<Value> newRegionArguments;
    newRegionArguments.push_back(newOp.getInductionVar());
    loadValues(op.getLoc(), allocated, zero, rewriter, newRegionArguments);

    auto oldTerminator = cast<scf::YieldOp>(op.getBody()->getTerminator());
    rewriter.mergeBlockBefore(op.getBody(), newOp.getBody()->getTerminator(),
                              newRegionArguments);

    rewriter.setInsertionPoint(newOp.getBody()->getTerminator());
    for (auto en : llvm::enumerate(oldTerminator.results())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], zero);
    }
    rewriter.eraseOp(oldTerminator);

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    loadValues(op.getLoc(), allocated, zero, rewriter, loaded);
    for (Value alloc : allocated) {
      rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), alloc);
    }
    // rewriter.create<LLVM::StackRestoreOp>(op.getLoc(), stackPtr);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

static void storeValues(Location loc, ValueRange values, ValueRange pointers,
                        Value zero, PatternRewriter &rewriter) {
  for (auto pair : llvm::zip(values, pointers)) {
    rewriter.create<memref::StoreOp>(loc, std::get<0>(pair), std::get<1>(pair),
                                     zero);
  }
}

static void allocaValues(Location loc, ValueRange values, Value zero,
                         PatternRewriter &rewriter,
                         SmallVector<Value> &allocated) {
  allocated.reserve(values.size());
  for (Value value : values) {
    Value alloc = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get(1, value.getType()), ValueRange());
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

    //Value stackPtr = rewriter.create<LLVM::StackSaveOp>(
    //    op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value> beforeAllocated, afterAllocated;
    allocaValues(op.getLoc(), op.getOperands(), zero, rewriter,
                 beforeAllocated);
    storeValues(op.getLoc(), op.getOperands(), beforeAllocated, zero, rewriter);
    allocaValues(op.getLoc(), op.getResults(), zero, rewriter, afterAllocated);

    auto newOp =
        rewriter.create<scf::WhileOp>(op.getLoc(), TypeRange(), ValueRange());
    Block *newBefore =
        rewriter.createBlock(&newOp.before(), newOp.before().begin());
    SmallVector<Value> newBeforeArguments;
    loadValues(op.getLoc(), beforeAllocated, zero, rewriter,
               newBeforeArguments);
    rewriter.mergeBlocks(&op.before().front(), newBefore, newBeforeArguments);

    auto beforeTerminator =
        cast<scf::ConditionOp>(newOp.before().front().getTerminator());
    rewriter.setInsertionPoint(beforeTerminator);
    storeValues(op.getLoc(), beforeTerminator.args(), afterAllocated, zero,
                rewriter);

    rewriter.updateRootInPlace(beforeTerminator,
                               [&] { beforeTerminator.argsMutable().clear(); });

    Block *newAfter =
        rewriter.createBlock(&newOp.after(), newOp.after().begin());
    SmallVector<Value> newAfterArguments;
    loadValues(op.getLoc(), afterAllocated, zero, rewriter, newAfterArguments);
    rewriter.mergeBlocks(&op.after().front(), newAfter, newAfterArguments);

    auto afterTerminator =
        cast<scf::YieldOp>(newOp.after().front().getTerminator());
    rewriter.setInsertionPoint(afterTerminator);
    storeValues(op.getLoc(), afterTerminator.results(), beforeAllocated, zero,
                rewriter);

    rewriter.updateRootInPlace(
        afterTerminator, [&] { afterTerminator.resultsMutable().clear(); });

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> results;
    loadValues(op.getLoc(), afterAllocated, zero, rewriter, results);
    //rewriter.create<LLVM::StackRestoreOp>(op.getLoc(), stackPtr);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  std::string method;
  CPUifyPass(std::string method) : method(method) {}
  void runOnFunction() override {
    if (method == "distribute") {
        OwningRewritePatternList patterns(&getContext());
        patterns
            .insert<Reg2MemFor, Reg2MemWhile, ReplaceIfWithFors, WrapForWithBarrier,
                    WrapWhileWithBarrier, InterchangeForPFor,
                    InterchangeForPForLoad, InterchangeWhilePFor, NormalizeLoop,
                    NormalizeParallel, RotateWhile, DistributeAroundBarrier>(
                &getContext());
        GreedyRewriteConfig config;
        config.maxIterations = 142;
        if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                                config)))
          signalPassFailure();
    } else if (method == "omp") {
        SmallVector<polygeist::BarrierOp> toReplace;
        getFunction().walk([&](polygeist::BarrierOp b) {
            toReplace.push_back(b);
        });
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
std::unique_ptr<Pass> createCPUifyPass(std::string str) {
  return std::make_unique<CPUifyPass>(str);
}
} // namespace polygeist
} // namespace mlir
