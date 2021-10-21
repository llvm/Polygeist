//===- FoldSCFIf.cc - Fold scf.if into select --------------C++-===//

#include "polymer/Transforms/FoldSCFIf.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "fold-scf-if"

static bool hasSingleStore(Block *block) {
  llvm::SetVector<Value> memrefs;

  for (Operation &op : block->getOperations()) {
    if (isa<mlir::AffineStoreOp, memref::StoreOp>(op)) {
      Value memref = op.getOperand(1);
      if (memrefs.count(memref))
        return false;

      // The indices should be defined above the current block.
      if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(op)) {
        if (any_of(storeOp.getMapOperands(), [&](Value operand) {
              return operand.getParentBlock() == block;
            }))
          return false;
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (any_of(storeOp.getIndices(), [&](Value operand) {
              return operand.getParentBlock() == block;
            }))
          return false;
      }

      memrefs.insert(memref);
    }
  }
  return true;
}

namespace {
struct MatchIfElsePass : PassWrapper<MatchIfElsePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    // If there is no store in the target block for a specific memref stored in
    // the source block, we will create a dummy load.
    auto matchStore = [&](Block *target, Block *source, Location loc) {
      llvm::SetVector<Value> memrefs;

      for (Operation &op : target->getOperations())
        if (isa<memref::StoreOp, mlir::AffineStoreOp>(op))
          memrefs.insert(op.getOperand(1));

      b.setInsertionPoint(target->getTerminator());
      for (Operation &op : source->getOperations()) {
        if (!isa<mlir::AffineStoreOp, memref::StoreOp>(op))
          continue;
        Value memref = op.getOperand(1);
        if (memrefs.count(memref)) // has been stored to
          continue;

        if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
          Value value = b.create<AffineLoadOp>(loc, storeOp.getMemRef(),
                                               storeOp.getAffineMap(),
                                               storeOp.getMapOperands());
          b.create<AffineStoreOp>(loc, value, storeOp.getMemRef(),
                                  storeOp.getAffineMap(),
                                  storeOp.getMapOperands());
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
          Value value = b.create<memref::LoadOp>(loc, storeOp.getMemRef(),
                                                 storeOp.getIndices());
          b.create<memref::StoreOp>(loc, value, storeOp.getMemRef(),
                                    storeOp.indices());
        }
      }
    };

    f.walk([&](scf::IfOp ifOp) {
      Location loc = ifOp.getLoc();
      OpBuilder::InsertionGuard g(b);

      // If there is no else block, initialize one with a terminating yield.
      if (!ifOp.elseBlock()) {
        ifOp.elseRegion().emplaceBlock();

        b.setInsertionPointToStart(ifOp.elseBlock());
        b.create<scf::YieldOp>(loc);
      }

      if (!hasSingleStore(ifOp.thenBlock()) ||
          !hasSingleStore(ifOp.elseBlock())) {
        LLVM_DEBUG(
            dbgs()
            << "Skipped if:\n"
            << ifOp
            << "\ndue to there are duplicated stores on the same memref.");
        return;
      }

      matchStore(ifOp.elseBlock(), ifOp.thenBlock(), loc);
      matchStore(ifOp.thenBlock(), ifOp.elseBlock(), loc);

      LLVM_DEBUG(dbgs() << "Matched else block:\n" << ifOp << "\n\n");
    });
  }
};
} // namespace

/// ---------------------- LiftStoreOps ------------------------------

static bool hasMatchingStores(ArrayRef<Block *> blocks) {
  if (blocks.size() <= 1)
    return true;

  llvm::SetVector<Value> setUnion;

  for (Block *block : blocks.drop_front()) {
    llvm::SetVector<Value> memrefs;

    for (Operation &op : block->getOperations())
      if (isa<memref::StoreOp, mlir::AffineStoreOp>(op)) {
        Value memref = op.getOperand(1);
        assert(!memrefs.count(memref) &&
               "Should only apply on blocks that contain single store to each "
               "memref.");

        memrefs.insert(op.getOperand(1));
      }

    bool wasEmpty = setUnion.empty();
    if (!wasEmpty && setUnion.set_union(memrefs))
      return false;
  }

  return true;
}

namespace {
struct MemRefStoreInfo {
  unsigned index;
  Type type;
  Operation *source;
  SmallVector<Value> operands;
};
} // namespace

static void getMemRefStoreInfo(Block *block,
                               MapVector<Value, MemRefStoreInfo> &storeInfo) {
  unsigned ord = 0;
  for (Operation &op : block->getOperations())
    if (isa<memref::StoreOp, mlir::AffineStoreOp>(op)) {
      MemRefStoreInfo info;
      info.index = ord++;
      info.type = op.getOperand(0).getType();
      info.source = &op;

      if (auto storeOp = dyn_cast<memref::StoreOp>(op))
        info.operands = storeOp.getIndices();
      else if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(op))
        info.operands = storeOp.getMapOperands();

      storeInfo[op.getOperand(1)] = info;
    }
}

static LogicalResult liftStoreOps(scf::IfOp ifOp, FuncOp f, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  if (!hasMatchingStores({ifOp.thenBlock(), ifOp.elseBlock()}))
    return failure();

  MapVector<Value, MemRefStoreInfo> storeInfo;
  getMemRefStoreInfo(ifOp.thenBlock(), storeInfo);

  SmallVector<Type> storeTypes(storeInfo.size());
  for (auto &info : storeInfo)
    storeTypes[info.second.index] = info.second.type;

  // No need to process further.
  if (storeInfo.empty())
    return failure();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Type> resultTypes(ifOp.getResultTypes());
  resultTypes.append(storeTypes);

  scf::IfOp newIfOp = b.create<scf::IfOp>(loc, resultTypes, ifOp.condition(),
                                          /*withElseRegion=*/true);

  auto cloneBlock = [&](Block *target, Block *source) {
    BlockAndValueMapping vmap;

    scf::YieldOp yieldOp = cast<scf::YieldOp>(source->getTerminator());
    unsigned numExistingResults = yieldOp.getNumOperands();
    SmallVector<Value> results(numExistingResults + storeInfo.size());

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(target);

    for (Operation &op : source->getOperations()) {
      if (isa<memref::StoreOp, mlir::AffineStoreOp>(op)) {
        Value memref = op.getOperand(1);
        Value toStore = op.getOperand(0);
        results[storeInfo[memref].index + numExistingResults] =
            vmap.lookupOrDefault(toStore);
      } else if (!isa<scf::YieldOp>(op)) {
        b.clone(op, vmap);
      }
    }

    for (auto operand : enumerate(yieldOp.getOperands()))
      results[operand.index()] = vmap.lookupOrDefault(operand.value());

    b.create<scf::YieldOp>(loc, results);
  };

  cloneBlock(newIfOp.thenBlock(), ifOp.thenBlock());
  cloneBlock(newIfOp.elseBlock(), ifOp.elseBlock());

  b.setInsertionPointAfter(newIfOp);

  for (auto &p : storeInfo) {
    Value memref;
    MemRefStoreInfo info;
    std::tie(memref, info) = p;

    if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(info.source))
      b.create<mlir::AffineStoreOp>(
          loc, newIfOp.getResult(ifOp.getNumResults() + info.index), memref,
          storeOp.getAffineMap(), info.operands);
    else if (auto storeOp = dyn_cast<memref::StoreOp>(info.source))
      b.create<memref::StoreOp>(
          loc, newIfOp.getResult(ifOp.getNumResults() + info.index), memref,
          info.operands);
  }

  ifOp.erase();

  return success();
}

static bool processLiftStoreOps(FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](scf::IfOp ifOp) {
    if (!hasSingleStore(ifOp.thenBlock()) ||
        (ifOp.elseBlock() && !hasSingleStore(ifOp.elseBlock())))
      return;
    if (failed(liftStoreOps(ifOp, f, b)))
      return;

    changed = true;
  });

  return changed;
}

namespace {
struct LiftStoreOps : PassWrapper<LiftStoreOps, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    // For each scf.if, see if it has single store for each memref on each
    // branch.
    while (processLiftStoreOps(f, b))
      ;
  }
};
} // namespace

/// ---------------------- FoldSCFIf ----------------------------------

static void foldSCFIf(scf::IfOp ifOp, FuncOp f, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  LLVM_DEBUG(dbgs() << "Working on ifOp: " << ifOp << "\n\n");

  if (!hasSingleStore(ifOp.thenBlock()) ||
      (ifOp.elseBlock() && !hasSingleStore(ifOp.elseBlock())))
    return;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Value> thenResults, elseResults;

  auto cloneAfter = [&](Block *block, SmallVectorImpl<Value> &results) {
    BlockAndValueMapping vmap;
    for (Operation &op : block->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        for (Value result : yieldOp.getOperands())
          results.push_back(vmap.contains(result) ? vmap.lookup(result)
                                                  : result);
      else
        b.clone(op, vmap);
    }
  };

  cloneAfter(ifOp.thenBlock(), thenResults);

  // Only an if op can have results when an else block is present.
  if (ifOp.elseBlock()) {
    cloneAfter(ifOp.elseBlock(), elseResults);

    for (auto ifResult : enumerate(ifOp.getResults())) {
      Value newResult = b.create<SelectOp>(loc, ifOp.condition(),
                                           thenResults[ifResult.index()],
                                           elseResults[ifResult.index()]);
      ifResult.value().replaceAllUsesWith(newResult);
    }
  }

  ifOp.erase();
}

/// Return true if anything changed.
static bool process(FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](scf::IfOp ifOp) {
    /// TODO: add verification.
    foldSCFIf(ifOp, f, b);
    changed = true;
  });

  return changed;
}

namespace {
struct FoldSCFIfPass : PassWrapper<FoldSCFIfPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    while (process(f, b))
      ;
  }
};
} // namespace

void polymer::registerFoldSCFIfPass() {
  PassPipelineRegistration<>("fold-scf-if", "Fold scf.if into select.",
                             [](OpPassManager &pm) {
                               pm.addPass(std::make_unique<MatchIfElsePass>());
                               pm.addPass(std::make_unique<LiftStoreOps>());
                               pm.addPass(std::make_unique<FoldSCFIfPass>());
                             });
}
