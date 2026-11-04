//===- AffineScalRep.cpp - Affine scalar replacement, non-affine aware ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A copy of MLIR's affine scalar replacement (store-to-load forwarding,
// redundant load elimination and dead store elimination: `affineScalarReplace`
// and its helpers in mlir/lib/Dialect/Affine/Utils/Utils.cpp of the LLVM 18
// tree this build uses) with one correctness fix.
//
// The upstream check that no write intervenes between the forwarded store (or
// load) and the replaced load uses the affine dependence analysis, which only
// models affine.for / affine.if: every other op holding a region is
// transparent to it. cgeist runs the pass while its loops are still scf.while
// / scf.for (before --raise-scf-to-affine, and for every loop that is never
// raised), so a write that follows the load in the body of such a loop looks
// like straight-line code after the load and is not seen to reach the load in
// the next iteration: `out[0] = 0; for (...) out[0] += a[i];` became a loop
// storing `a[i] + 0` (O23). Here an affine access found between the two ops
// that sits, together with the replaced load, inside a non-affine op whose
// region may run again (or whose regions / blocks may reach one another) and
// that does not enclose the store / load providing the value counts as an
// intervening effect, unless both accesses use indices that are fixed while
// the op runs and differ by a non-zero constant (index operands computed by
// identical side-effect-free ops, such as the separate arith.index_cast cgeist
// emits for every access, count as one value).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "polygeist-affine-scalrep"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Returns true if `value` is defined inside `op` (by an op nested in it or as
/// an argument of one of its blocks).
static bool isDefinedInside(Operation *op, Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    def = cast<BlockArgument>(value).getOwner()->getParentOp();
  return def && op->isAncestor(def);
}

/// Returns true if `x` and `y` always hold the same value: they are the same
/// SSA value, or the same result of two side-effect-free ops without regions
/// that have the same name, attributes, properties and types and operands that
/// hold the same values (at most `depth` ops deep). cgeist emits a separate
/// arith.index_cast for every access of `o[s]`, and nothing has merged them
/// when the forwarding first runs. An op without operands counts only if it
/// is a constant (two `llvm.mlir.undef` may differ).
static bool holdSameValue(Value x, Value y, unsigned depth = 4) {
  if (x == y)
    return true;
  if (depth == 0 || x.getType() != y.getType())
    return false;
  auto rx = dyn_cast<OpResult>(x), ry = dyn_cast<OpResult>(y);
  if (!rx || !ry || rx.getResultNumber() != ry.getResultNumber())
    return false;
  Operation *ox = rx.getOwner(), *oy = ry.getOwner();
  for (Operation *o : {ox, oy})
    if (o->getNumRegions() != 0 || !isMemoryEffectFree(o) ||
        (o->getNumOperands() == 0 && !o->hasTrait<OpTrait::ConstantLike>()))
      return false;
  return OperationEquivalence::isEquivalentTo(
      ox, oy,
      [&](Value l, Value r) {
        return success(holdSameValue(l, r, depth - 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

/// Returns true if the affine accesses `a` and `b` touch different elements
/// whenever they execute within one execution of `scope`: scalar accesses of
/// the same memref with the identity layout, every index operand defined
/// outside `scope` (so fixed while it runs) and access functions that differ
/// by a non-zero constant in some dimension.
static bool accessesDisjointWithin(Operation *scope, Operation *a,
                                   Operation *b) {
  // A vector access covers several elements from its index on.
  if (isa<AffineVectorLoadOp, AffineVectorStoreOp>(a) ||
      isa<AffineVectorLoadOp, AffineVectorStoreOp>(b))
    return false;
  MemRefAccess accA(a), accB(b);
  if (accA.memref != accB.memref)
    return false;
  // Distinct indices are distinct elements only for an injective layout.
  if (!cast<MemRefType>(accA.memref.getType()).getLayout().isIdentity())
    return false;
  AffineValueMap mapA, mapB, diff;
  accA.getAccessMap(&mapA);
  accB.getAccessMap(&mapB);
  for (Value v : mapA.getOperands())
    if (isDefinedInside(scope, v))
      return false;
  for (Value v : mapB.getOperands())
    if (isDefinedInside(scope, v))
      return false;
  if (mapA.getNumResults() != mapB.getNumResults())
    return false;
  // Give operands of `b` that hold the same value as an operand of `a` that
  // operand, so the difference below cancels them.
  SmallVector<Value> operandsB(mapB.getOperands());
  for (Value &vb : operandsB)
    for (Value va : mapA.getOperands())
      if (holdSameValue(va, vb)) {
        vb = va;
        break;
      }
  mapB.reset(mapB.getAffineMap(), operandsB);
  AffineValueMap::difference(mapA, mapB, &diff);
  for (AffineExpr e : diff.getAffineMap().getResults())
    if (auto c = e.dyn_cast<AffineConstantExpr>())
      if (c.getValue() != 0)
        return true;
  return false;
}

/// Returns the region of `op` that contains `nested`.
static Region *getRegionContaining(Operation *op, Operation *nested) {
  for (Region &r : op->getRegions())
    if (r.findAncestorOpInRegion(*nested))
      return &r;
  return nullptr;
}

/// The affine dependence analysis (`mayHaveEffect`) treats ops other than
/// affine.for / affine.if as transparent. Returns true if the affine access
/// `accessOp`, found while checking the ops between `start` and `memOp`, may
/// execute after `start` and before `memOp` through such an op: one that
/// encloses `accessOp` and `memOp` but not `start` (an op that encloses
/// `start` as well re-executes `start` before `memOp`) and whose region
/// holding `memOp` may run again after `accessOp` (a loop, or an op that does
/// not describe its control flow), or that holds the two in different regions
/// that may reach one another, or in different blocks of one region. Program
/// order inside one block of a region that runs at most once per execution of
/// the op is what the analysis models correctly.
static bool mayReachThroughNonAffineOp(Operation *start, Operation *memOp,
                                       Operation *accessOp) {
  for (Operation *x = memOp->getParentOp(); x && !x->isAncestor(start);
       x = x->getParentOp()) {
    if (!x->isProperAncestor(accessOp))
      continue;
    if (isa<AffineForOp, AffineIfOp>(x))
      continue;
    Region *memRegion = getRegionContaining(x, memOp);
    Region *accRegion = getRegionContaining(x, accessOp);
    if (!memRegion || !accRegion)
      return true;
    bool repeats = true;
    auto branch = dyn_cast<RegionBranchOpInterface>(x);
    if (branch && !isa<LoopLikeOpInterface>(x))
      repeats = branch.isRepetitiveRegion(memRegion->getRegionNumber());
    if (!repeats) {
      if (memRegion == accRegion) {
        if (memRegion->findAncestorBlockInRegion(*memOp->getBlock()) ==
            memRegion->findAncestorBlockInRegion(*accessOp->getBlock()))
          continue;
      } else if (insideMutuallyExclusiveRegions(accessOp, memOp)) {
        continue;
      }
    }
    if (accessesDisjointWithin(x, accessOp, memOp))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "intervening through " << x->getName() << ": "
                            << *accessOp << "\n");
    return true;
  }
  return false;
}

/// Returns true if the memory operation of `destAccess` depends on `srcAccess`
/// inside of the innermost common surrounding affine loop between the two
/// accesses.
static bool mustReachAtInnermost(const MemRefAccess &srcAccess,
                                 const MemRefAccess &destAccess) {
  // Affine dependence analysis is possible only if both ops in the same
  // AffineScope.
  if (getAffineScope(srcAccess.opInst) != getAffineScope(destAccess.opInst))
    return false;

  unsigned nsLoops =
      getNumCommonSurroundingLoops(*srcAccess.opInst, *destAccess.opInst);
  DependenceResult result =
      checkMemrefAccessDependence(srcAccess, destAccess, nsLoops + 1);
  return hasDependence(result);
}

/// Returns true if `srcMemOp` may have an effect on `destMemOp` within the
/// scope of the outermost `minSurroundingLoops` loops that surround them.
/// `srcMemOp` and `destMemOp` are expected to be affine read/write ops.
static bool mayHaveEffect(Operation *srcMemOp, Operation *destMemOp,
                          unsigned minSurroundingLoops) {
  MemRefAccess srcAccess(srcMemOp);
  MemRefAccess destAccess(destMemOp);

  // Affine dependence analysis here is applicable only if both ops operate on
  // the same memref and if `srcMemOp` and `destMemOp` are in the same
  // AffineScope. Also, we can only check if our affine scope is isolated from
  // above; otherwise, values can from outside of the affine scope that the
  // check below cannot analyze.
  Region *srcScope = getAffineScope(srcMemOp);
  if (srcAccess.memref == destAccess.memref &&
      srcScope == getAffineScope(destMemOp)) {
    unsigned nsLoops = getNumCommonSurroundingLoops(*srcMemOp, *destMemOp);
    FlatAffineValueConstraints dependenceConstraints;
    for (unsigned d = nsLoops + 1; d > minSurroundingLoops; d--) {
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, destAccess, d, &dependenceConstraints,
          /*dependenceComponents=*/nullptr);
      // A dependence failure or the presence of a dependence implies a
      // side effect.
      if (!noDependence(result))
        return true;
    }
    // No side effect was seen.
    return false;
  }
  // TODO: Check here if the memrefs alias: there is no side effect if
  // `srcAccess.memref` and `destAccess.memref` don't alias.
  return true;
}

/// Returns true if no operation that may have the effect `EffectType` on the
/// location `memOp` accesses can execute after `start` and before `memOp`.
template <typename EffectType, typename T>
static bool hasNoInterveningEffect(Operation *start, T memOp) {
  auto isLocallyAllocated = [](Value memref) {
    auto *defOp = memref.getDefiningOp();
    return defOp && hasSingleEffect<MemoryEffects::Allocate>(defOp, memref);
  };

  // A boolean representing whether an intervening operation could have impacted
  // memOp.
  bool hasSideEffect = false;

  // Check whether the effect on memOp can be caused by a given operation op.
  Value memref = memOp.getMemRef();
  std::function<void(Operation *)> checkOperation = [&](Operation *op) {
    // If the effect has alreay been found, early exit,
    if (hasSideEffect)
      return;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      bool opMayHaveEffect = false;
      for (auto effect : effects) {
        // If op causes EffectType on a potentially aliasing location for
        // memOp, mark as having the effect.
        if (isa<EffectType>(effect.getEffect())) {
          // TODO: This should be replaced with a check for no aliasing.
          // Aliasing information should be passed to this method.
          if (effect.getValue() && effect.getValue() != memref &&
              isLocallyAllocated(memref) &&
              isLocallyAllocated(effect.getValue()))
            continue;
          opMayHaveEffect = true;
          break;
        }
      }

      if (!opMayHaveEffect)
        return;

      // If the side effect comes from an affine read or write, try to
      // prove the side effecting `op` cannot reach `memOp`.
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
        // Polygeist: the dependence analysis below cannot see a path through
        // a non-affine loop (O23).
        if (mayReachThroughNonAffineOp(start, memOp, op)) {
          hasSideEffect = true;
          return;
        }
        // For ease, let's consider the case that `op` is a store and
        // we're looking for other potential stores that overwrite memory after
        // `start`, and before being read in `memOp`. In this case, we only
        // need to consider other potential stores with depth >
        // minSurroundingLoops since `start` would overwrite any store with a
        // smaller number of surrounding loops before.
        unsigned minSurroundingLoops =
            getNumCommonSurroundingLoops(*start, *memOp);
        if (mayHaveEffect(op, memOp, minSurroundingLoops))
          hasSideEffect = true;
        return;
      }

      // We have an op with a memory effect and we cannot prove if it
      // intervenes.
      hasSideEffect = true;
      return;
    }

    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      // Recurse into the regions for this op and check whether the internal
      // operations may have the side effect `EffectType` on memOp.
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &op : block)
            checkOperation(&op);
      return;
    }

    // Otherwise, conservatively assume generic operations have the effect
    // on the operation
    hasSideEffect = true;
  };

  // Check all paths from ancestor op `parent` to the operation `to` for the
  // effect. It is known that `to` must be contained within `parent`.
  auto until = [&](Operation *parent, Operation *to) {
    // TODO check only the paths from `parent` to `to`.
    // Currently we fallback and check the entire parent op, rather than
    // just the paths from the parent path, stopping after reaching `to`.
    // This is conservatively correct, but could be made more aggressive.
    assert(parent->isAncestor(to));
    checkOperation(parent);
  };

  // Check for all paths from operation `from` to operation `untilOp` for the
  // given memory effect.
  std::function<void(Operation *, Operation *)> recur =
      [&](Operation *from, Operation *untilOp) {
        assert(
            from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
            "Checking for side effect between two operations without a common "
            "ancestor");

        // If the operations are in different regions, recursively consider all
        // path from `from` to the parent of `to` and all paths from the parent
        // of `to` to `to`.
        if (from->getParentRegion() != untilOp->getParentRegion()) {
          recur(from, untilOp->getParentOp());
          until(untilOp->getParentOp(), untilOp);
          return;
        }

        // Now, assuming that `from` and `to` exist in the same region, perform
        // a CFG traversal to check all the relevant operations.

        // Additional blocks to consider.
        SmallVector<Block *, 2> todoBlocks;
        {
          // First consider the parent block of `from` an check all operations
          // after `from`.
          for (auto iter = ++from->getIterator(), end = from->getBlock()->end();
               iter != end && &*iter != untilOp; ++iter) {
            checkOperation(&*iter);
          }

          // If the parent of `from` doesn't contain `to`, add the successors
          // to the list of blocks to check.
          if (untilOp->getBlock() != from->getBlock())
            for (Block *succ : from->getBlock()->getSuccessors())
              todoBlocks.push_back(succ);
        }

        SmallPtrSet<Block *, 4> done;
        // Traverse the CFG until hitting `to`.
        while (!todoBlocks.empty()) {
          Block *blk = todoBlocks.pop_back_val();
          if (done.count(blk))
            continue;
          done.insert(blk);
          for (auto &op : *blk) {
            if (&op == untilOp)
              break;
            checkOperation(&op);
            if (&op == blk->getTerminator())
              for (Block *succ : blk->getSuccessors())
                todoBlocks.push_back(succ);
          }
        }
      };
  recur(start, memOp);
  return !hasSideEffect;
}

/// Attempt to eliminate loadOp by replacing it with a value stored into memory
/// which the load is guaranteed to retrieve. This check involves three
/// components: 1) The store and load must be on the same location 2) The store
/// must dominate (and therefore must always occur prior to) the load 3) No
/// other operations will overwrite the memory loaded between the given load
/// and store.  If such a value exists, the replaced `loadOp` will be added to
/// `loadOpsToErase` and its memref will be added to `memrefsToErase`.
static void forwardStoreToLoad(AffineReadOpInterface loadOp,
                               SmallVectorImpl<Operation *> &loadOpsToErase,
                               SmallPtrSetImpl<Value> &memrefsToErase,
                               DominanceInfo &domInfo) {

  // The store op candidate for forwarding that satisfies all conditions
  // to replace the load, if any.
  Operation *lastWriteStoreOp = nullptr;

  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess)
      continue;

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(storeOp, loadOp))
      continue;

    // 3. The store must reach the load. Access function equivalence only
    // guarantees this for accesses in the same block. The load could be in a
    // nested block that is unreachable.
    if (storeOp->getBlock() != loadOp->getBlock() &&
        !mustReachAtInnermost(srcAccess, destAccess))
      continue;

    // 4. Ensure there is no intermediate operation which could replace the
    // value in memory.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp))
      continue;

    // We now have a candidate for forwarding.
    assert(lastWriteStoreOp == nullptr &&
           "multiple simultaneous replacement stores");
    lastWriteStoreOp = storeOp;
  }

  if (!lastWriteStoreOp)
    return;

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine vector
  // loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType())
    return;
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);
}

// This attempts to find stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates writeA.
// 3) There is no potential read between writeA and writeB.
static void findUnusedStore(AffineWriteOpInterface writeA,
                            SmallVectorImpl<Operation *> &opsToErase,
                            PostDominanceInfo &postDominanceInfo) {

  for (Operation *user : writeA.getMemRef().getUsers()) {
    // Only consider writing operations.
    auto writeB = dyn_cast<AffineWriteOpInterface>(user);
    if (!writeB)
      continue;

    // The operations must be distinct.
    if (writeB == writeA)
      continue;

    // Both operations must lie in the same region.
    if (writeB->getParentRegion() != writeA->getParentRegion())
      continue;

    // Both operations must write to the same memory.
    MemRefAccess srcAccess(writeB);
    MemRefAccess destAccess(writeA);

    if (srcAccess != destAccess)
      continue;

    // writeB must postdominate writeA.
    if (!postDominanceInfo.postDominates(writeB, writeA))
      continue;

    // There cannot be an operation which reads from memory between
    // the two writes.
    if (!hasNoInterveningEffect<MemoryEffects::Read>(writeA, writeB))
      continue;

    opsToErase.push_back(writeA);
    break;
  }
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB.
static void loadCSE(AffineReadOpInterface loadA,
                    SmallVectorImpl<Operation *> &loadOpsToErase,
                    DominanceInfo &domInfo) {
  SmallVector<AffineReadOpInterface, 4> loadCandidates;
  for (auto *user : loadA.getMemRef().getUsers()) {
    auto loadB = dyn_cast<AffineReadOpInterface>(user);
    if (!loadB || loadB == loadA)
      continue;

    MemRefAccess srcAccess(loadB);
    MemRefAccess destAccess(loadA);

    // 1. The accesses should be to be to the same location.
    if (srcAccess != destAccess) {
      continue;
    }

    // 2. loadB should dominate loadA.
    if (!domInfo.dominates(loadB, loadA))
      continue;

    // 3. There should not be a write between loadA and loadB.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(loadB.getOperation(),
                                                      loadA))
      continue;

    // Check if two values have the same shape. This is needed for affine vector
    // loads.
    if (loadB.getValue().getType() != loadA.getValue().getType())
      continue;

    loadCandidates.push_back(loadB);
  }

  // Of the legal load candidates, use the one that dominates all others
  // to minimize the subsequent need to loadCSE
  Value loadB;
  for (AffineReadOpInterface option : loadCandidates) {
    if (llvm::all_of(loadCandidates, [&](AffineReadOpInterface depStore) {
          return depStore == option ||
                 domInfo.dominates(option.getOperation(),
                                   depStore.getOperation());
        })) {
      loadB = option.getValue();
      break;
    }
  }

  if (loadB) {
    loadA.getValue().replaceAllUsesWith(loadB);
    // Record this to erase later.
    loadOpsToErase.push_back(loadA);
  }
}

/// Upstream `affine::affineScalarReplace` with the checks above.
static void affineScalarReplace(func::FuncOp f, DominanceInfo &domInfo,
                                PostDominanceInfo &postDomInfo) {
  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineReadOpInterface loadOp) {
    forwardStoreToLoad(loadOp, opsToErase, memrefsToErase, domInfo);
  });
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Walk all store's and perform unused store elimination
  f.walk([&](AffineWriteOpInterface storeOp) {
    findUnusedStore(storeOp, opsToErase, postDomInfo);
  });
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Check if the store fwd'ed memrefs are now left with only stores and
  // deallocs and can thus be completely deleted. Note: the canonicalize pass
  // should be able to do this as well, but we'll do it here since we collected
  // these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been locally alloc'ed, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !hasSingleEffect<MemoryEffects::Allocate>(defOp, memref))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface>(ownerOp) &&
                 !hasSingleEffect<MemoryEffects::Free>(ownerOp, memref);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }

  // To eliminate as many loads as possible, run load CSE after eliminating
  // stores. Otherwise, some stores are wrongly seen as having an intervening
  // effect.
  f.walk([&](AffineReadOpInterface loadOp) {
    loadCSE(loadOp, opsToErase, domInfo);
  });
  for (auto *op : opsToErase)
    op->erase();
}

struct PolygeistAffineScalarReplacement
    : public PassWrapper<PolygeistAffineScalarReplacement,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolygeistAffineScalarReplacement)
  StringRef getArgument() const final { return "polygeist-affine-scalrep"; }
  StringRef getDescription() const final {
    return "affine-scalrep that sees writes through non-affine loops";
  }
  void runOnOperation() override {
    affineScalarReplace(getOperation(), getAnalysis<DominanceInfo>(),
                        getAnalysis<PostDominanceInfo>());
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::polygeist::createPolygeistAffineScalarReplacementPass() {
  return std::make_unique<PolygeistAffineScalarReplacement>();
}
