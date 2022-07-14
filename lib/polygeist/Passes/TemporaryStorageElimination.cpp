//===- TemporaryStorageElimination.cpp - Shared memory-like elimination ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "tmp-storage-elimination"
#define DBGS() llvm::dbgs() << "[" << DEBUG_TYPE << "] "

namespace {
Block *getCommonAncestorBlock(Operation *first, Operation *second) {
  Region *firstRegion = first->getParentRegion();
  Region *secondRegion = second->getParentRegion();
  if (firstRegion->isAncestor(secondRegion))
    return first->getBlock();
  if (secondRegion->isAncestor(firstRegion))
    return second->getBlock();

  for (Region *region = firstRegion->getParentRegion(); region != nullptr;
       region = region->getParentRegion()) {
    if (region->isAncestor(secondRegion)) {
      if (!llvm::hasSingleElement(*region))
        return nullptr;
      return &region->getBlocks().front();
    }
  }
  return nullptr;
}

AffineStoreOp findWriter(AffineLoadOp loadOp, Operation *root) {
  // Find the stores to the same memref.
  AffineStoreOp candidateStoreOp = nullptr;
  WalkResult result = root->walk([&](AffineStoreOp storeOp) {
    if (loadOp.getMemRef() != storeOp.getMemRef())
      return WalkResult::advance();
    if (candidateStoreOp)
      return WalkResult::interrupt();
    candidateStoreOp = storeOp;
    return WalkResult::advance();
  });

  // If there's no or more than one writer, bail out.
  if (result.wasInterrupted() || !candidateStoreOp) {
    LLVM_DEBUG(DBGS() << "could not find the single writer\n");
    return AffineStoreOp();
  }

  // Check that the store happens before the load.
  Block *commonParent = getCommonAncestorBlock(candidateStoreOp, loadOp);
  if (!commonParent) {
    LLVM_DEBUG(
        DBGS() << "could not find a common parent between load and store\n");
    return AffineStoreOp();
  }

  if (!commonParent->findAncestorOpInBlock(*candidateStoreOp)
           ->isBeforeInBlock(commonParent->findAncestorOpInBlock(*loadOp))) {
    LLVM_DEBUG(DBGS() << "the store does not precede the load\n");
    return AffineStoreOp();
  }

  FlatAffineRelation loadRelation, storeRelation;
  if (failed(MemRefAccess(loadOp).getAccessRelation(loadRelation)) ||
      failed(MemRefAccess(candidateStoreOp).getAccessRelation(storeRelation))) {
    LLVM_DEBUG(DBGS() << "could not construct affine access relations\n");
    return AffineStoreOp();
  }
  if (!loadRelation.getRangeSet().isSubsetOf(storeRelation.getRangeSet())) {
    LLVM_DEBUG(
        DBGS()
        << "the set of loaded values is not a subset of written values\n");
    return AffineStoreOp();
  }

  return candidateStoreOp;
}

AffineLoadOp findStoredValueLoad(AffineStoreOp storeOp) {
  return storeOp.getValueToStore().getDefiningOp<AffineLoadOp>();
}

bool hasInterferringWrite(AffineLoadOp loadOp, AffineLoadOp originalLoadOp,
                          Operation *root) {
  WalkResult result = root->walk([&](AffineStoreOp storeOp) {
    // TODO: don't assume no-alias.
    if (storeOp.getMemRef() != originalLoadOp.getMemRef())
      return WalkResult::advance();

    // TODO: check if the store may happen before originalLoadOp and storeOp.
    // For now, conservatively assume it may.
    FlatAffineRelation loadRelation, storeRelation;
    if (failed(MemRefAccess(originalLoadOp).getAccessRelation(loadRelation)) ||
        failed(MemRefAccess(storeOp).getAccessRelation(storeRelation))) {
      LLVM_DEBUG(DBGS() << "could not construct affine access relations in "
                           "interference analysis\n");
      return WalkResult::interrupt();
    }

    if (!storeRelation.getRangeSet()
             .intersect(loadRelation.getRangeSet())
             .isEmpty()) {
      LLVM_DEBUG(DBGS() << "found interferring store: " << *storeOp << "\n");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

AffineExpr tryExtractAffineExpr(const FlatAffineRelation &relation,
                                unsigned rangeDim, MLIRContext *ctx) {
  std::unique_ptr<FlatAffineValueConstraints> clone = relation.clone();

  clone->projectOut(relation.getNumDomainDims(), rangeDim);
  clone->projectOut(relation.getNumDomainDims() + 1,
                    relation.getNumRangeDims() - rangeDim - 1);
  if (clone->getNumEqualities() != 1)
    return AffineExpr();

  // TODO: support for local ids via mods.
  ArrayRef<int64_t> eqCoeffs = clone->getEquality(0);
  if (llvm::any_of(eqCoeffs.slice(relation.getNumDomainDims() + 1,
                                  relation.getNumLocalIds()),
                   [](int64_t coeff) { return coeff != 0; })) {
    return AffineExpr();
  }

  AffineExpr expr = getAffineConstantExpr(eqCoeffs.back(), ctx);
  for (unsigned i = 0, e = relation.getNumDomainDims(); i != e; ++i) {
    expr = expr +
           getAffineConstantExpr(eqCoeffs[i], ctx) * getAffineDimExpr(i, ctx);
  }
  for (unsigned i = 0, e = relation.getNumSymbolIds(); i != e; ++i) {
    expr = expr + getAffineConstantExpr(
                      eqCoeffs[relation.getNumDomainDims() + 1 + i], ctx) *
                      getAffineSymbolExpr(i, ctx);
  }
  return expr;
}

AffineMap tryExtractAffineMap(const FlatAffineRelation &relation,
                              MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  for (unsigned i = 0, e = relation.getNumRangeDims(); i != e; ++i) {
    exprs.push_back(tryExtractAffineExpr(relation, i, ctx));
    if (!exprs.back())
      return AffineMap();
  }
  return AffineMap::get(relation.getNumDomainDims(), relation.getNumSymbolIds(),
                        exprs, ctx);
}

void loadStoreForwarding(Operation *root) {
  root->walk([root](AffineLoadOp loadOp) {
    LLVM_DEBUG(DBGS() << "-----------------------------------------\n");
    LLVM_DEBUG(DBGS() << "considering " << *loadOp << "\n");
    AffineStoreOp storeOp = findWriter(loadOp, root);
    if (!storeOp)
      return;

    AffineLoadOp originalLoadOp = findStoredValueLoad(storeOp);
    if (!originalLoadOp)
      return;

    if (hasInterferringWrite(loadOp, originalLoadOp, root))
      return;

    // Replace the load, need the index remapping.

    // LLoops -> SMem.
    FlatAffineRelation loadRelation;
    // SLoops -> SMem.
    FlatAffineRelation storeRelation;
    // SLoops -> GMem.
    FlatAffineRelation originalLoadRelation;
    if (failed(MemRefAccess(loadOp).getAccessRelation(loadRelation)) ||
        failed(MemRefAccess(storeOp).getAccessRelation(storeRelation)) ||
        failed(MemRefAccess(originalLoadOp)
                   .getAccessRelation(originalLoadRelation))) {
      LLVM_DEBUG(DBGS() << "could not construct affine access in remapping\n");
      return;
    }

    // SMem -> SLoops.
    storeRelation.inverse();
    // LLoops -> SLoops.
    storeRelation.compose(loadRelation);
    // LLoops -> GMem
    originalLoadRelation.compose(storeRelation);

    AffineMap accessMap =
        tryExtractAffineMap(originalLoadRelation, root->getContext());
    if (!accessMap) {
      LLVM_DEBUG(DBGS() << "could not remap the access\n");
      return;
    }

    IRRewriter rewriter(root->getContext());
    rewriter.setInsertionPoint(loadOp);
    rewriter.replaceOpWithNewOp<AffineLoadOp>(
        loadOp, originalLoadOp.getMemRef(), accessMap, loadOp.getIndices());
    LLVM_DEBUG(DBGS() << "replaced\n");
  });
}

void removeWriteOnlyAllocas(Operation *root) {
  SmallVector<Operation *> toErase;
  root->walk([&](memref::AllocaOp allocaOp) {
    auto isWrite = [](Operation *op) {
      return isa<AffineWriteOpInterface, memref::StoreOp>(op);
    };
    if (llvm::all_of(allocaOp.getResult().getUsers(), isWrite)) {
      llvm::append_range(toErase, allocaOp.getResult().getUsers());
      toErase.push_back(allocaOp);
    }
  });
  for (Operation *op : toErase)
    op->erase();
}

struct TemporaryStorageEliminationPass
    : TemporaryStorageEliminationBase<TemporaryStorageEliminationPass> {
  void runOnOperation() override {
    loadStoreForwarding(getOperation());
    removeWriteOnlyAllocas(getOperation());
  }
};

} // namespace

namespace mlir {
namespace polygeist {
void registerTemporaryStorageEliminationPass() {
  PassRegistration<TemporaryStorageEliminationPass> reg;
}

std::unique_ptr<Pass> createTemporaryStorageEliminationPass() {
  return std::make_unique<TemporaryStorageEliminationPass>();
}
} // namespace polygeist
} // namespace mlir
