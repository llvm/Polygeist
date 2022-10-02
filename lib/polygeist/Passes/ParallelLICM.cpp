#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "polygeist/Passes/Passes.h"

#define DEBUG_TYPE "parallel-licm"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct ParallelLICM : public ParallelLICMBase<ParallelLICM> {
  void runOnOperation() override;
};
} // namespace

static bool canBeParallelHoisted(Operation *op, Operation *scope,
                                 SmallPtrSetImpl<Operation *> &willBeMoved,
                                 bool includeAfter = false) {
  // Helper to check whether an operation is loop invariant wrt. SSA properties.
  LLVM_DEBUG(llvm::dbgs() << "Checking for parallel hoist: " << *op << "\n");
  auto definedOutside = [&](Value value) {
    if (auto BA = value.dyn_cast<BlockArgument>())
      if (willBeMoved.count(BA.getOwner()->getParentOp()))
        return true;
    auto *definingOp = value.getDefiningOp();
    if ((definingOp && !!willBeMoved.count(definingOp)) ||
        cast<LoopLikeOpInterface>(scope).isDefinedOutsideOfLoop(value))
      return true;
    LLVM_DEBUG(llvm::dbgs()
               << " - cannot hoist due to operand: " << value << "\n");
    return false;
  };

  // Check that dependencies are defined outside of loop.
  if (!llvm::all_of(op->getOperands(), definedOutside))
    return false;

  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    memEffect.getEffects(effects);

    SmallVector<MemoryEffects::EffectInstance> readResources;
    SmallVector<MemoryEffects::EffectInstance> writeResources;
    SmallVector<MemoryEffects::EffectInstance> freeResources;
    for (auto effect : effects) {
      if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
        LLVM_DEBUG(llvm::dbgs() << " - cannot hoist due to allocation like\n");
        return false;
      }
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        readResources.push_back(effect);
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        writeResources.push_back(effect);
      if (isa<MemoryEffects::Free>(effect.getEffect()))
        freeResources.push_back(effect);
    }

    std::function<bool(Operation *)> conflicting = [&](Operation *b) {
      if (willBeMoved.count(b))
        return false;

      if (b->hasTrait<OpTrait::HasRecursiveSideEffects>()) {

        for (auto &region : b->getRegions()) {
          for (auto &block : region) {
            for (auto &innerOp : block)
              if (conflicting(&innerOp))
                return true;
          }
        }
        return false;
      }

      auto memEffect = dyn_cast<MemoryEffectOpInterface>(b);
      if (!memEffect) {
        LLVM_DEBUG(llvm::dbgs()
                   << " - cannot hoist due to unknown memeffect conflict: "
                   << *b << "\n");
        return true;
      }
      for (auto res : readResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res.getResource(), effects);
        for (auto effect : effects) {
          if (!mayAlias(effect, res))
            continue;
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to read->alloc conflict: " << *b
                       << "\n");
            return true;
          }
          if (isa<MemoryEffects::Write>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to read->write conflict: " << *b
                       << "\n");
            return true;
          }
        }
      }
      for (auto res : writeResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res.getResource(), effects);
        for (auto effect : effects) {
          if (!mayAlias(effect, res))
            continue;
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to write->alloc conflict: " << *b
                       << "\n");
            return true;
          }
          if (isa<MemoryEffects::Read>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to write->read conflict: " << *b
                       << "\n");
            return true;
          }
        }
      }
      for (auto res : freeResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res.getResource(), effects);
        for (auto effect : effects) {
          if (!mayAlias(effect, res))
            continue;
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to free->alloc conflict: " << *b
                       << "\n");
            return true;
          }
          if (isa<MemoryEffects::Write>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to free->write conflict: " << *b
                       << "\n");
            return true;
          }
          if (isa<MemoryEffects::Read>(effect.getEffect())) {
            LLVM_DEBUG(llvm::dbgs()
                       << " - cannot hoist due to free->read conflict: " << *b
                       << "\n");
            return true;
          }
        }
      }
      return false;
    };

    std::function<bool(Operation *)> hasConflictBefore = [&](Operation *b) {
      if (includeAfter) {
        for (Operation &it : *b->getBlock()) {
          if (conflicting(&it)) {
            return true;
          }
        }
      } else {
        for (Operation *it = b->getPrevNode(); it != nullptr;
             it = it->getPrevNode()) {
          if (conflicting(it)) {
            return true;
          }
        }
      }

      if (b->getParentOp() == scope)
        return false;
      if (hasConflictBefore(b->getParentOp()))
        return true;

      bool conflict = false;
      // If the parent operation is not guaranteed to execute its (single-block)
      // region once, walk the block.
      if (!isa<scf::IfOp, AffineIfOp, memref::AllocaScopeOp>(b))
        b->walk([&](Operation *in) {
          if (conflict)
            return WalkResult::interrupt();
          if (conflicting(in)) {
            conflict = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

      return conflict;
    };
    if ((readResources.size() || writeResources.size() ||
         freeResources.size()) &&
        hasConflictBefore(op))
      return false;
  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    LLVM_DEBUG(llvm::dbgs()
               << " - cannot hoist due to non memory/recursive side effects\n");
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted.
  // We can inductively assume that this op will have its block args available
  // outside the loop
  SmallPtrSet<Operation *, 2> willBeMoved2(willBeMoved.begin(),
                                           willBeMoved.end());
  willBeMoved2.insert(op);
  /*
  for (auto &region : op->getRegions())
    for (auto &block : region)
       for (auto arg : block.getArguments())
           willBeMoved2.insert(&arg);
           */

  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &innerOp : block)
        if (!canBeParallelHoisted(&innerOp, scope, willBeMoved2,
                                  includeAfter)) {
          LLVM_DEBUG(llvm::dbgs()
                     << " - cannot hoist due to inner: " << innerOp << "\n");
          return false;
        } else
          willBeMoved2.insert(&innerOp);
    }
  }
  return true;
}

bool below(AffineExpr expr, size_t numDim, ValueRange operands, int64_t val);

bool below(Value bval, int64_t val) {
  // Unknown size currently unhandled.
  if (val == -1)
    return false;

  if (auto baval = bval.dyn_cast<BlockArgument>()) {
    if (AffineForOp afFor =
            dyn_cast<AffineForOp>(baval.getOwner()->getParentOp())) {
      for (auto ub : afFor.getUpperBoundMap().getResults()) {
        if (!below(ub, afFor.getUpperBoundMap().getNumDims(),
                   afFor.getUpperBoundOperands(), val + 1))
          return false;
      }
      return true;
    }
    if (AffineParallelOp afFor =
            dyn_cast<AffineParallelOp>(baval.getOwner()->getParentOp())) {
      for (auto ub :
           afFor.getUpperBoundMap(baval.getArgNumber()).getResults()) {
        if (!below(ub, afFor.getUpperBoundsMap().getNumDims(),
                   afFor.getUpperBoundsOperands(), val + 1))
          return false;
      }
      return true;
    }

    if (scf::ForOp afFor =
            dyn_cast<scf::ForOp>(baval.getOwner()->getParentOp())) {
      if (baval.getArgNumber() == 0) {
        return below(afFor.getUpperBound(), val + 1);
      }
    }

    if (scf::ParallelOp afFor =
            dyn_cast<scf::ParallelOp>(baval.getOwner()->getParentOp())) {
      return below(afFor.getUpperBound()[baval.getArgNumber()], val + 1);
    }
  }

  IntegerAttr iattr;
  if (matchPattern(bval, m_Constant(&iattr))) {
    return iattr.getValue().getSExtValue() < val;
  }

  return false;
}

bool below(AffineExpr expr, size_t numDim, ValueRange operands, int64_t val) {
  // Unknown size currently unhandled.
  if (val == -1)
    return false;

  if (auto opd = expr.dyn_cast<AffineConstantExpr>()) {
    if (opd.getValue() < val)
      return true;
    return false;
  }
  if (auto opd = expr.dyn_cast<AffineDimExpr>()) {
    return below(operands[opd.getPosition()], val);
  }
  if (auto opd = expr.dyn_cast<AffineSymbolExpr>()) {
    return below(operands[opd.getPosition() + numDim], val);
  }
  return false;
}

bool isSpeculatable(Operation *op) {
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // If the op has no side-effects, it is speculatable.
    if (memInterface.hasNoEffect())
      return true;

    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      Value ptr = load.getMemref();
      if (ptr.getDefiningOp<memref::AllocOp>() ||
          ptr.getDefiningOp<memref::AllocaOp>()) {
        auto S = ptr.getType().cast<MemRefType>().getShape();
        AffineMap map = load.getAffineMapAttr().getValue();
        for (auto idx : llvm::enumerate(map.getResults())) {
          if (!below(idx.value(), map.getNumDims(), load.getMapOperands(),
                     S[idx.index()])) {
            return false;
          }
        }
        return true;
      }
    }

    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      Value ptr = load.getMemref();
      if (ptr.getDefiningOp<memref::AllocOp>() ||
          ptr.getDefiningOp<memref::AllocaOp>()) {
        auto S = ptr.getType().cast<MemRefType>().getShape();
        for (auto idx : llvm::enumerate(load.getIndices())) {
          if (!below(idx.value(), S[idx.index()]))
            return false;
        }
        return true;
      }
    }

    // If the op does not have recursive side effects, then it is not
    // speculatable.
    if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>())
      return false;
  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    // Otherwise, if the op does not implement the memory effect interface and
    // it does not have recursive side effects, then it cannot be speculated.
    return false;
  }

  // Recurse into the regions and ensure that all nested ops can also be moved.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (!isSpeculatable(&op))
        return false;
  return true;
}

void moveParallelLoopInvariantCode(scf::ParallelOp looplike) {

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.

  std::function<void(Operation *, bool)> recur = [&](Operation *metaop,
                                                     bool checkSpeculative) {
    for (Region &region : metaop->getRegions())
      for (Block &block : region)
        for (Operation &op : block.without_terminator())
          if ((!checkSpeculative || isSpeculatable(&op)) &&
              canBeParallelHoisted(&op, looplike, willBeMovedSet)) {
            opsToMove.push_back(&op);
            willBeMovedSet.insert(&op);
          } else {
            recur(&op, /*checkSpeculative*/ true);
          }
  };
  recur(looplike, /*checkSpeculative*/ false);

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  if (!llvm::all_of(opsToMove, isSpeculatable)) {
    OpBuilder b(looplike);
    Value cond = nullptr;
    for (auto pair : llvm::zip(looplike.getLowerBound(),
                               looplike.getUpperBound(), looplike.getStep())) {
      auto val = b.create<arith::CmpIOp>(looplike.getLoc(), CmpIPredicate::slt,
                                         std::get<0>(pair), std::get<1>(pair));
      if (cond == nullptr)
        cond = val;
      else
        cond = b.create<arith::AndIOp>(looplike.getLoc(), cond, val);
    }
    auto ifOp =
        b.create<scf::IfOp>(looplike.getLoc(), looplike.getResultTypes(), cond,
                            /*hasElse*/ !looplike.getResultTypes().empty());
    if (!ifOp.thenBlock()->empty())
      ifOp.thenBlock()->getTerminator()->erase();

    looplike->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->begin());
    looplike.replaceAllUsesWith(ifOp->getResults());
    OpBuilder B(ifOp.thenBlock(), ifOp.thenBlock()->end());
    B.create<scf::YieldOp>(looplike.getLoc(), looplike.getResults());
    if (!looplike.getResultTypes().empty()) {
      B.setInsertionPointToEnd(ifOp.elseBlock());
      B.create<scf::YieldOp>(looplike.getLoc(), looplike.getInitVals());
    }
  }
  for (auto op : opsToMove)
    looplike.moveOutOfLoop(op);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

// TODO affine parallel licm
void moveParallelLoopInvariantCode(AffineParallelOp looplike) {

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  std::function<void(Operation *, bool)> recur = [&](Operation *metaop,
                                                     bool checkSpeculative) {
    for (Region &region : metaop->getRegions())
      for (Block &block : region)
        for (Operation &op : block.without_terminator())
          if ((!checkSpeculative || isSpeculatable(&op)) &&
              canBeParallelHoisted(&op, looplike, willBeMovedSet)) {
            opsToMove.push_back(&op);
            willBeMovedSet.insert(&op);
          } else {
            recur(&op, /*checkSpeculative*/ true);
          }
  };
  recur(looplike, /*checkSpeculative*/ false);

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  if (!llvm::all_of(opsToMove, isSpeculatable)) {
    OpBuilder b(looplike);

    // TODO properly fill exprs and eqflags
    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;

    for (auto step : llvm::enumerate(looplike.getSteps())) {
      for (auto ub : looplike.getUpperBoundMap(step.index()).getResults()) {
        SmallVector<AffineExpr, 4> symbols;
        for (unsigned idx = 0;
             idx < looplike.getUpperBoundsMap().getNumSymbols(); ++idx)
          symbols.push_back(getAffineSymbolExpr(
              idx + looplike.getLowerBoundsMap().getNumSymbols(),
              looplike.getContext()));

        SmallVector<AffineExpr, 4> dims;
        for (unsigned idx = 0; idx < looplike.getUpperBoundsMap().getNumDims();
             ++idx)
          dims.push_back(
              getAffineDimExpr(idx + looplike.getLowerBoundsMap().getNumDims(),
                               looplike.getContext()));

        ub = ub.replaceDimsAndSymbols(dims, symbols);

        for (auto lb : looplike.getLowerBoundMap(step.index()).getResults()) {

          // Bound is whether this expr >= 0, which since we want ub > lb, we
          // rewrite as follows.
          exprs.push_back(ub - lb - step.value());
          eqflags.push_back(false);
        }
      }
    }

    SmallVector<Value> values;
    auto lb_ops = looplike.getLowerBoundsOperands();
    auto ub_ops = looplike.getUpperBoundsOperands();
    for (unsigned idx = 0; idx < looplike.getLowerBoundsMap().getNumDims();
         ++idx) {
      values.push_back(lb_ops[idx]);
    }
    for (unsigned idx = 0; idx < looplike.getUpperBoundsMap().getNumDims();
         ++idx) {
      values.push_back(ub_ops[idx]);
    }
    for (unsigned idx = 0; idx < looplike.getLowerBoundsMap().getNumSymbols();
         ++idx) {
      values.push_back(lb_ops[idx + looplike.getLowerBoundsMap().getNumDims()]);
    }
    for (unsigned idx = 0; idx < looplike.getUpperBoundsMap().getNumSymbols();
         ++idx) {
      values.push_back(ub_ops[idx + looplike.getUpperBoundsMap().getNumDims()]);
    }

    auto iset = IntegerSet::get(
        /*dim*/ looplike.getLowerBoundsMap().getNumDims() +
            looplike.getUpperBoundsMap().getNumDims(),
        /*symbols*/ looplike.getLowerBoundsMap().getNumSymbols() +
            looplike.getUpperBoundsMap().getNumSymbols(),
        exprs, eqflags);
    auto ifOp = b.create<AffineIfOp>(
        looplike.getLoc(), looplike.getResultTypes(), iset, values,
        /*hasElse*/ !looplike.getResultTypes().empty());
    if (!ifOp.getThenBlock()->empty())
      ifOp.getThenBlock()->getTerminator()->erase();

    looplike->moveBefore(ifOp.getThenBlock(), ifOp.getThenBlock()->begin());
    looplike.replaceAllUsesWith(ifOp->getResults());
    OpBuilder B(ifOp.getThenBlock(), ifOp.getThenBlock()->end());
    B.create<AffineYieldOp>(looplike.getLoc(), looplike.getResults());
    if (!looplike.getResultTypes().empty()) {
      B.setInsertionPointToEnd(ifOp.getElseBlock());
      // TODO affine parallel initial value for reductions.
      // B.create<AffineYieldOp>(looplike.getLoc(), looplike.getIterOperands());
    }
  }
  for (auto op : opsToMove)
    looplike.moveOutOfLoop(op);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

void moveSerialLoopInvariantCode(scf::ForOp looplike) {

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  std::function<void(Operation *, bool)> recur = [&](Operation *metaop,
                                                     bool checkSpeculative) {
    for (Region &region : metaop->getRegions())
      for (Block &block : region)
        for (Operation &op : block.without_terminator())
          if ((!checkSpeculative || isSpeculatable(&op)) &&
              canBeParallelHoisted(&op, looplike, willBeMovedSet,
                                   /*checkAfter*/ true)) {
            opsToMove.push_back(&op);
            willBeMovedSet.insert(&op);
          } else {
            recur(&op, /*checkSpeculative*/ true);
          }
  };
  recur(looplike, /*checkSpeculative*/ false);

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  if (!llvm::all_of(opsToMove, isSpeculatable)) {
    OpBuilder b(looplike);
    Value cond = b.create<arith::CmpIOp>(looplike.getLoc(), CmpIPredicate::slt,
                                         looplike.getLowerBound(),
                                         looplike.getUpperBound());
    auto ifOp =
        b.create<scf::IfOp>(looplike.getLoc(), looplike.getResultTypes(), cond,
                            /*hasElse*/ !looplike.getResultTypes().empty());
    if (!ifOp.thenBlock()->empty())
      ifOp.thenBlock()->getTerminator()->erase();

    looplike->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->begin());
    looplike.replaceAllUsesWith(ifOp->getResults());
    OpBuilder B(ifOp.thenBlock(), ifOp.thenBlock()->end());
    B.create<scf::YieldOp>(looplike.getLoc(), looplike.getResults());
    if (!looplike.getResultTypes().empty()) {
      B.setInsertionPointToEnd(ifOp.elseBlock());
      B.create<scf::YieldOp>(looplike.getLoc(), looplike.getIterOperands());
    }
  }
  for (auto op : opsToMove)
    looplike.moveOutOfLoop(op);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

void moveSerialLoopInvariantCode(AffineForOp looplike) {

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  std::function<void(Operation *, bool)> recur = [&](Operation *metaop,
                                                     bool checkSpeculative) {
    for (Region &region : metaop->getRegions())
      for (Block &block : region)
        for (Operation &op : block.without_terminator()) {
          if ((!checkSpeculative || isSpeculatable(&op)) &&
              canBeParallelHoisted(&op, looplike, willBeMovedSet,
                                   /*checkAfter*/ true)) {
            opsToMove.push_back(&op);
            willBeMovedSet.insert(&op);
          } else {
            recur(&op, /*checkSpeculative*/ true);
          }
        }
  };
  recur(looplike, /*checkSpeculative*/ false);

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  if (!llvm::all_of(opsToMove, isSpeculatable)) {
    OpBuilder b(looplike);

    // TODO properly fill exprs and eqflags
    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;

    auto step = looplike.getStep();
    for (auto ub : looplike.getUpperBoundMap().getResults()) {
      SmallVector<AffineExpr, 4> symbols;
      for (unsigned idx = 0; idx < looplike.getUpperBoundMap().getNumSymbols();
           ++idx)
        symbols.push_back(getAffineSymbolExpr(
            idx + looplike.getLowerBoundMap().getNumSymbols(),
            looplike.getContext()));

      SmallVector<AffineExpr, 4> dims;
      for (unsigned idx = 0; idx < looplike.getUpperBoundMap().getNumDims();
           ++idx)
        dims.push_back(
            getAffineDimExpr(idx + looplike.getLowerBoundMap().getNumDims(),
                             looplike.getContext()));

      ub = ub.replaceDimsAndSymbols(dims, symbols);

      for (auto lb : looplike.getLowerBoundMap().getResults()) {

        // Bound is whether this expr >= 0, which since we want ub > lb, we
        // rewrite as follows.
        exprs.push_back(ub - lb - step);
        eqflags.push_back(false);
      }
    }

    SmallVector<Value> values;
    auto lb_ops = looplike.getLowerBoundOperands();
    auto ub_ops = looplike.getUpperBoundOperands();
    for (unsigned idx = 0; idx < looplike.getLowerBoundMap().getNumDims();
         ++idx) {
      values.push_back(lb_ops[idx]);
    }
    for (unsigned idx = 0; idx < looplike.getUpperBoundMap().getNumDims();
         ++idx) {
      values.push_back(ub_ops[idx]);
    }
    for (unsigned idx = 0; idx < looplike.getLowerBoundMap().getNumSymbols();
         ++idx) {
      values.push_back(lb_ops[idx + looplike.getLowerBoundMap().getNumDims()]);
    }
    for (unsigned idx = 0; idx < looplike.getUpperBoundMap().getNumSymbols();
         ++idx) {
      values.push_back(ub_ops[idx + looplike.getUpperBoundMap().getNumDims()]);
    }

    auto iset = IntegerSet::get(
        /*dim*/ looplike.getLowerBoundMap().getNumDims() +
            looplike.getUpperBoundMap().getNumDims(),
        /*symbols*/ looplike.getLowerBoundMap().getNumSymbols() +
            looplike.getUpperBoundMap().getNumSymbols(),
        exprs, eqflags);
    auto ifOp = b.create<AffineIfOp>(
        looplike.getLoc(), looplike.getResultTypes(), iset, values,
        /*hasElse*/ !looplike.getResultTypes().empty());
    if (!ifOp.getThenBlock()->empty())
      ifOp.getThenBlock()->getTerminator()->erase();

    looplike->moveBefore(ifOp.getThenBlock(), ifOp.getThenBlock()->begin());
    looplike.replaceAllUsesWith(ifOp->getResults());
    OpBuilder B(ifOp.getThenBlock(), ifOp.getThenBlock()->end());
    B.create<AffineYieldOp>(looplike.getLoc(), looplike.getResults());
    if (!looplike.getResultTypes().empty()) {
      B.setInsertionPointToEnd(ifOp.getElseBlock());
      B.create<AffineYieldOp>(looplike.getLoc(), looplike.getIterOperands());
    }
  }
  for (auto op : opsToMove)
    looplike.moveOutOfLoop(op);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

void ParallelLICM::runOnOperation() {
  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\nOriginal loop:\n"));
    moveLoopInvariantCode(loopLike);
    if (auto par = dyn_cast<scf::ParallelOp>((Operation *)loopLike)) {
      moveParallelLoopInvariantCode(par);
    } else if (auto par = dyn_cast<AffineParallelOp>((Operation *)loopLike)) {
      moveParallelLoopInvariantCode(par);
    } else if (auto par = dyn_cast<scf::ForOp>((Operation *)loopLike)) {
      moveSerialLoopInvariantCode(par);
    } else if (auto par = dyn_cast<AffineForOp>((Operation *)loopLike)) {
      moveSerialLoopInvariantCode(par);
    }
  });
}

std::unique_ptr<Pass> mlir::polygeist::createParallelLICMPass() {
  return std::make_unique<ParallelLICM>();
}
