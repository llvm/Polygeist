//===- Reg2Mem.cc - reg2mem transformation --------------------------------===//
//
// This file implements the reg2mem transformation pass.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/Reg2Mem.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "polymer/Support/PolymerUtils.h"

#include <map>

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "reg2mem"

using DefToUsesMap =
    llvm::MapVector<mlir::Value, llvm::SetVector<mlir::Operation *>>;

/// Build the mapping of values from where they are defined to where they are
/// used. We will need this information to decide whether a Value should be
/// stored in a scratchpad, and if so, what the scratchpad should look like.
/// Note that we only care about those values that are on the use-def chain that
/// ends up with an affine write operation, or one with side effects. We also
/// ignore all the def-use pairs that are in the same block.
static void mapDefToUses(mlir::func::FuncOp f, DefToUsesMap &defToUses) {
  f.walk([&](mlir::Operation *useOp) {
    // Op that belongs to AffineWriteOpInterface (e.g., affine.store) or has
    // recursive side effects will be treated as .
    if (!isa<mlir::affine::AffineWriteOpInterface>(useOp) &&
        !useOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
      return;
    // Should filter out for and if ops.
    if (isa<mlir::affine::AffineForOp, mlir::affine::AffineIfOp>(useOp))
      return;

    // Assuming the def-use chain is acyclic.
    llvm::SmallVector<mlir::Operation *, 8> ops;
    ops.push_back(useOp);

    while (!ops.empty()) {
      mlir::Operation *op = ops.pop_back_val();

      for (mlir::Value v : op->getOperands()) {
        mlir::Operation *defOp = v.getDefiningOp();
        // Don't need to go further if v is defined by the following operations.
        // - AllocOp: we cannot load/store a memref value itself;
        // - DimOp/affine::AffineApplyOp: indices and bounds shouldn't be loaded
        // from memory, otherwise, it would mess up with the dependence
        // analysis.

        if (!defOp ||
            isa<memref::AllocOp, memref::AllocaOp, memref::DimOp,
                mlir::arith::ConstantOp, mlir::affine::AffineApplyOp>(defOp) ||
            (isa<mlir::arith::IndexCastOp>(defOp) &&
             defOp->getOperand(0).isa<BlockArgument>()))
          continue;

        // The block that defines the value is different from the block of the
        // current op.
        if (v.getParentBlock() != op->getBlock())
          defToUses[v].insert(op);

        // No need to look at the operands of the following list of operations.
        if (!isa<mlir::affine::AffineLoadOp>(defOp))
          ops.push_back(defOp);
      }
    }
  });
}

/// Keep a single use for each def in the same block. The reason for doing so is
/// that we only create one load op for each def value in the same block.
static void filterUsesInSameBlock(DefToUsesMap &defToUses) {
  llvm::SmallSet<mlir::Block *, 4> visitedBlocks;

  for (auto &defUsePair : defToUses) {
    visitedBlocks.clear();
    llvm::SetVector<mlir::Operation *> &useOps = defUsePair.second;
    llvm::SetVector<mlir::Operation *> opsToRemove;

    // Iterate every use op, and if it's block has been visited, we put that op
    // into the toRemove set.
    for (mlir::Operation *op : useOps) {
      mlir::Block *block = op->getBlock();
      if (visitedBlocks.contains(block))
        opsToRemove.insert(op);
      else
        visitedBlocks.insert(block);
    }

    for (mlir::Operation *op : opsToRemove)
      useOps.remove(op);
  }
}

/// Add scop.scratchpad to newly created AllocaOp for scratchpad.
static memref::AllocaOp annotateScratchpad(memref::AllocaOp allocaOp) {
  allocaOp->setAttr("scop.scratchpad",
                    mlir::UnitAttr::get(allocaOp.getContext()));
  return allocaOp;
}

/// Creates a single-entry scratchpad memory that stores values from the
/// defining point and can be loaded when needed at the uses.
static memref::AllocaOp createScratchpadAllocaOp(mlir::OpResult val,
                                                 mlir::OpBuilder &b,
                                                 mlir::Block *entryBlock) {
  // Sanity checks on the defining op.
  mlir::Operation *defOp = val.getOwner();

  // Set the allocation point after where the val is defined.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entryBlock);

  // The memref shape is 1 and the type is derived from val.
  return annotateScratchpad(b.create<memref::AllocaOp>(
      defOp->getLoc(), MemRefType::get({1}, val.getType())));
}

/// Creata an affine::AffineStoreOp for the value to be stored on the
/// scratchpad.
static mlir::affine::AffineStoreOp
createScratchpadStoreOp(mlir::Value valToStore, memref::AllocaOp allocaOp,
                        mlir::OpBuilder &b) {
  // Create a storeOp to the memref using address 0. The new storeOp will be
  // placed right after the allocaOp, and its location is hinted by allocaOp.
  // Here we assume that allocaOp is dominated by the defining op of valToStore.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(valToStore);

  return b.create<mlir::affine::AffineStoreOp>(
      allocaOp.getLoc(), valToStore, allocaOp.getResult(),
      b.getConstantAffineMap(0), std::vector<mlir::Value>());
}

/// Create an affine::AffineLoadOp for the value stored in the scratchpad. The
/// insertion point will be at the beginning of the block of the useOp, such
/// that all the subsequent uses of the Value in the scratchpad can re-use the
/// same load result. Note that we don't check whether the useOp is still using
/// the original value that is stored in the scratchpad (some replacement could
/// happen already), you need to do that before calling this function to avoid
/// possible redundancy. This function won't replace uses.
static mlir::affine::AffineLoadOp
createScratchpadLoadOp(memref::AllocaOp allocaOp, mlir::Operation *useOp,
                       mlir::OpBuilder &b) {
  // The insertion point will be at the beginning of the parent block for useOp.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(useOp->getBlock());
  // The location is set to be the useOp that will finally use this newly
  // created load op. The address is set to be 0 since the memory has only one
  // element in it. You will need to replace the input to useOp outside.
  return b.create<mlir::affine::AffineLoadOp>(
      useOp->getLoc(), allocaOp.getResult(), b.getConstantAffineMap(0),
      std::vector<mlir::Value>());
}

namespace polymer {
void demoteRegisterToMemory(mlir::func::FuncOp f, OpBuilder &b) {
  if (f.getBlocks().size() == 0)
    return;

  DefToUsesMap defToUses;
  Block &entryBlock = *f.getBody().begin();

  // Get the mapping from a value to its uses that are in a
  // different block as where the value itself is defined.
  mapDefToUses(f, defToUses);
  // Make sure every def will have a single use in each block.
  filterUsesInSameBlock(defToUses);

  LLVM_DEBUG({
    for (auto &p : defToUses) {
      dbgs() << " -- Defined value: " << p.first << '\n';
      dbgs() << "    Uses:\n";
      for (auto &use : p.second) {
        dbgs() << "      + " << (*use) << '\n';
      }
    }
  });

  // Handle each def-use pair in in the current function.
  for (const auto &defUsesPair : defToUses) {
    // The value to be stored in a scratchpad.
    mlir::Value val = defUsesPair.first;

    // Create the alloca op for the scratchpad.
    memref::AllocaOp allocaOp =
        createScratchpadAllocaOp(dyn_cast<mlir::OpResult>(val), b, &entryBlock);

    // Create the store op that stores val into the scratchpad for future uses.
    createScratchpadStoreOp(val, allocaOp, b);

    // Iterate each use of val, and create the load op, the result of which will
    // replace the original val. After creating this load op, we replaces the
    // uses of the original val in the same block as the load op by the result
    // of it. And for those already replaced, we pop them out of the list to be
    // processed (useOps).
    const llvm::SetVector<mlir::Operation *> &useOps = defUsesPair.second;

    for (mlir::Operation *useOp : useOps) {
      // Create the load op for it.
      mlir::affine::AffineLoadOp loadOp =
          createScratchpadLoadOp(allocaOp, useOp, b);

      // Replace the uses of val in the same region as useOp (or loadOp).
      val.replaceUsesWithIf(loadOp.getResult(), [&](mlir::OpOperand &operand) {
        mlir::Operation *currUseOp = operand.getOwner();
        //  Check the equivalence of the regions.
        return currUseOp->getParentRegion() == useOp->getParentRegion();
      });
    }
  }
}
} // namespace polymer

static IntegerSet getIntegerSetForElse(const IntegerSet &iSet, OpBuilder &b) {
  SmallVector<AffineExpr, 8> newExprs;
  SmallVector<bool, 8> newEqFlags;

  auto eqFlags = iSet.getEqFlags();
  for (unsigned i = 0; i < iSet.getNumConstraints(); i++) {
    AffineExpr expr = iSet.getConstraint(i);

    if (eqFlags[i]) {
      // For equality, we create two inequalities that exclude the target.
      newExprs.push_back(expr - b.getAffineConstantExpr(1));
      newEqFlags.push_back(false);
      newExprs.push_back(-expr - b.getAffineConstantExpr(1));
      newEqFlags.push_back(false);
    } else {
      // For inequality, we simply negate the condition.
      newExprs.push_back(-expr - b.getAffineConstantExpr(1));
      newEqFlags.push_back(eqFlags[i]);
    }
  }

  return IntegerSet::get(iSet.getNumDims(), iSet.getNumSymbols(), newExprs,
                         newEqFlags);
}

namespace polymer {
/// Turns affine.if with else block into two affine.if. It works iteratively:
/// one affine.if op (with else) should be handled at one time.
void separateAffineIfBlocks(mlir::func::FuncOp f, OpBuilder &b) {
  // Get the first affine.if operation that has an else block.
  auto findIfWithElse = [&](auto &f) {
    Operation *opFound = nullptr;
    f.walk([&](mlir::affine::AffineIfOp ifOp) {
      if (!opFound && ifOp.hasElse()) {
        opFound = ifOp;
        return;
      }
    });
    return opFound;
  };

  Operation *op;
  while ((op = findIfWithElse(f)) != nullptr) {
    mlir::affine::AffineIfOp ifOp = dyn_cast<mlir::affine::AffineIfOp>(op);
    assert(ifOp.hasElse() && "The if op found should have an else block.");

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(ifOp);

    // Here we create two new if operations, one for the then block, one for the
    // else block.
    mlir::affine::AffineIfOp newIfThenOp =
        cast<mlir::affine::AffineIfOp>(b.clone(*ifOp.getOperation()));
    newIfThenOp.getElseBlock()->erase(); // TODO: can we don't create it?

    b.setInsertionPointAfter(newIfThenOp);
    mlir::affine::AffineIfOp newIfElseOp =
        cast<mlir::affine::AffineIfOp>(b.clone(*ifOp.getOperation()));

    // Build new integer set.
    newIfElseOp.setConditional(getIntegerSetForElse(ifOp.getIntegerSet(), b),
                               ifOp.getOperands());

    Block *oldThenBlock = newIfElseOp.getThenBlock();
    newIfElseOp.getElseBlock()->moveBefore(oldThenBlock);
    oldThenBlock->erase();

    ifOp.erase();
  }
}
} // namespace polymer

class RegToMemPass
    : public mlir::PassWrapper<RegToMemPass,
                               OperationPass<mlir::func::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());

    if (f->hasAttr("scop.ignored"))
      return;

    polymer::separateAffineIfBlocks(f, builder);
    polymer::demoteRegisterToMemory(f, builder);
  }
};

/// TODO: value analysis
static void insertRedundantLoad(mlir::func::FuncOp f, OpBuilder &b) {
  DominanceInfo dom(f);

  SmallVector<mlir::affine::AffineStoreOp, 4> storeOps;
  f.walk(
      [&storeOps](mlir::affine::AffineStoreOp op) { storeOps.push_back(op); });

  llvm::SetVector<mlir::Operation *> storeOpsToLoad;

  for (mlir::affine::AffineStoreOp storeOp : storeOps) {
    Value valueToStore = storeOp.getValueToStore();
    Value memref = storeOp.getMemRef();

    // TODO: deal with the complexity here.
    for (mlir::Operation *user : valueToStore.getUsers()) {
      if (user == storeOp)
        continue;
      if (user->getBlock() != storeOp->getBlock())
        continue;

      //  This user is another storeOp that dominates the current storeOp.
      if (isa<mlir::affine::AffineStoreOp>(user) &&
          dom.dominates(user, storeOp)) {
        // ... and there is no other storeOp that is beging dominated in
        // between.
        bool hasDominatedMemStore = false;
        for (mlir::Operation *memUser : memref.getUsers()) {
          if (memUser == storeOp || memUser == user)
            continue;
          if (isa<mlir::affine::AffineStoreOp>(memUser) &&
              dom.dominates(user, memUser) && dom.dominates(memUser, storeOp)) {
            hasDominatedMemStore = true;
            break;
          }
        }

        if (hasDominatedMemStore)
          continue;

        storeOpsToLoad.insert(user);
      }
    }
  }

  for (mlir::Operation *op : storeOpsToLoad) {
    mlir::affine::AffineStoreOp storeOpToLoad =
        cast<mlir::affine::AffineStoreOp>(op);

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(storeOpToLoad);

    Value value = storeOpToLoad.getValueToStore();
    Value memref = storeOpToLoad.getMemRef();
    mlir::affine::AffineLoadOp loadOp = b.create<mlir::affine::AffineLoadOp>(
        storeOpToLoad.getLoc(), memref, storeOpToLoad.getAffineMap(),
        storeOpToLoad.getMapOperands());

    value.replaceUsesWithIf(loadOp.getResult(), [=](mlir::OpOperand &operand) {
      return operand.getOwner() != op &&
             operand.getOwner()->getBlock() == op->getBlock();
    });
  }
}

class InsertRedundantLoadPass
    : public mlir::PassWrapper<InsertRedundantLoadPass,
                               OperationPass<mlir::func::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    insertRedundantLoad(f, b);
  }
};

namespace {
using IterArgToMemMap = llvm::MapVector<mlir::Value, mlir::Value>;
}

static void
findReductionLoops(mlir::func::FuncOp f,
                   SmallVectorImpl<mlir::affine::AffineForOp> &forOps) {
  f.walk([&](mlir::affine::AffineForOp forOp) {
    if (!forOp.getInits().empty())
      forOps.push_back(forOp);
  });
}

static mlir::affine::AffineYieldOp
findYieldOp(mlir::affine::AffineForOp forOp) {
  mlir::Operation *retOp;
  forOp.walk([&](mlir::affine::AffineYieldOp yieldOp) { retOp = yieldOp; });

  assert(retOp != nullptr);
  assert(isa<mlir::affine::AffineYieldOp>(retOp));

  return cast<mlir::affine::AffineYieldOp>(retOp);
}

static mlir::Value createIterScratchpad(mlir::Value iterArg,
                                        IterArgToMemMap &iterArgToMem,
                                        mlir::affine::AffineForOp forOp,
                                        OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(forOp);

  assert(!iterArgToMem.count(iterArg) &&
         "A scratchpad has been created for the given iterArg");

  mlir::Value spad = annotateScratchpad(b.create<memref::AllocaOp>(
      forOp.getLoc(), MemRefType::get({1}, iterArg.getType())));
  iterArgToMem[iterArg] = spad;

  return spad;
}

static void storeInitValue(mlir::Value initVal, mlir::Value spad,
                           mlir::affine::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(spad);
  b.create<mlir::affine::AffineStoreOp>(
      spad.getLoc(), initVal, spad, b.getConstantAffineMap(0), ValueRange());
}

static mlir::Value loadIterArg(mlir::Value iterArg,
                               IterArgToMemMap &iterArgToMem,
                               mlir::affine::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(forOp.getBody());

  assert(iterArgToMem.count(iterArg));

  return b.create<mlir::affine::AffineLoadOp>(
      forOp.getLoc(), iterArgToMem[iterArg], b.getConstantAffineMap(0),
      ValueRange());
}

static void replaceIterArg(mlir::Value origIterArg, mlir::Value spadIterArg) {
  origIterArg.replaceAllUsesWith(spadIterArg);
}

static void storeIterArg(int idx, mlir::Value spad,
                         mlir::affine::AffineYieldOp yieldOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(yieldOp);
  b.create<mlir::affine::AffineStoreOp>(
      yieldOp.getLoc(), yieldOp.getOperand(idx), spad,
      b.getConstantAffineMap(0), ValueRange());
}

static mlir::Value loadFinalIterVal(mlir::Value spad,
                                    mlir::affine::AffineForOp forOp,
                                    OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  return b.create<mlir::affine::AffineLoadOp>(
      forOp.getLoc(), spad, b.getConstantAffineMap(0), ValueRange());
}

static void replaceResult(int idx, mlir::affine::AffineForOp forOp,
                          mlir::Value retVal) {
  mlir::Value origRetVal = forOp.getResult(idx);
  origRetVal.replaceAllUsesWith(retVal);
}

static mlir::affine::AffineForOp
cloneAffineForWithoutIterArgs(mlir::affine::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  mlir::affine::AffineForOp newForOp = b.create<mlir::affine::AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), forOp.getStep());

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  b.setInsertionPointToStart(newForOp.getBody());
  forOp.walk([&](mlir::Operation *op) {
    if (!isa<mlir::affine::AffineYieldOp>(op) && op->getParentOp() == forOp)
      b.clone(*op, mapping);
  });

  return newForOp;
}

namespace polymer {
void demoteLoopReduction(mlir::func::FuncOp f, mlir::affine::AffineForOp forOp,
                         OpBuilder &b) {
  SmallVector<mlir::Value, 4> initVals{forOp.getInits()};
  mlir::Block *body = forOp.getBody();
  mlir::affine::AffineYieldOp yieldOp = findYieldOp(forOp);

  IterArgToMemMap iterArgToMem;
  for (auto initVal : enumerate(initVals)) {
    mlir::Value iterArg = body->getArgument(initVal.index() + 1);
    mlir::Value spad = createIterScratchpad(iterArg, iterArgToMem, forOp, b);
    storeInitValue(initVal.value(), spad, forOp, b);

    mlir::Value iterArgFromSpad = loadIterArg(iterArg, iterArgToMem, forOp, b);
    replaceIterArg(iterArg, iterArgFromSpad);
    storeIterArg(initVal.index(), spad, yieldOp, b);

    mlir::Value finalIterVal = loadFinalIterVal(spad, forOp, b);
    replaceResult(initVal.index(), forOp, finalIterVal);
  }

  cloneAffineForWithoutIterArgs(forOp, b);
  for (mlir::Value result : forOp.getResults())
    assert(result.getUses().empty());
  forOp.erase();
}

void demoteLoopReduction(mlir::func::FuncOp f, OpBuilder &b) {
  SmallVector<mlir::affine::AffineForOp, 4> forOps;
  findReductionLoops(f, forOps);

  for (mlir::affine::AffineForOp forOp : forOps)
    demoteLoopReduction(f, forOp, b);
}
} // namespace polymer

class DemoteLoopReductionPass
    : public mlir::PassWrapper<DemoteLoopReductionPass,
                               OperationPass<mlir::func::FuncOp>> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    polymer::demoteLoopReduction(f, b);
  }
};

/// ---------------- Array Expansion -------------------

static void findAllScratchpads(mlir::func::FuncOp f,
                               SmallVector<mlir::Value, 4> &spads) {
  // All scratchpads are allocated by AllocaOp.
  f.walk([&](memref::AllocaOp op) {
    if (op->hasAttr("scop.scratchpad"))
      spads.push_back(op);
  });
}

static void
getEnclosingAffineForOps(mlir::Operation *op, mlir::Operation *topOp,
                         SmallVectorImpl<mlir::Operation *> &forOps) {
  SmallVector<mlir::Operation *, 4> forAndIfOps;
  affine::getEnclosingAffineOps(*op, &forAndIfOps);

  for (mlir::Operation *forOrIfOp : forAndIfOps) {
    if (forOrIfOp == topOp)
      break;
    if (isa<mlir::affine::AffineForOp>(forOrIfOp))
      forOps.push_back(forOrIfOp);
  }
}

static void getScratchpadIterDomains(
    mlir::Value spad,
    mlir::SmallVectorImpl<affine::FlatAffineValueConstraints> &indexSets) {
  mlir::SmallPtrSet<mlir::Operation *, 4> parentVisited;
  for (mlir::Operation *user : spad.getUsers()) {
    mlir::Operation *parent = user->getParentOp();
    if (parent == nullptr || parentVisited.count(parent))
      continue;
    parentVisited.insert(parent);

    SmallVector<mlir::Operation *, 4> forOps;
    getEnclosingAffineForOps(user, spad.getParentBlock()->getParentOp(),
                             forOps);

    affine::FlatAffineValueConstraints domain;
    auto res = succeeded(getIndexSet(forOps, &domain));
    assert(res && "Cannot get the iteration domain of the given forOps");

    indexSets.push_back(domain);
  }
}

static void getNonZeroDims(ArrayRef<int64_t> coeffs,
                           const affine::FlatAffineValueConstraints &cst,
                           SmallVectorImpl<int64_t> &dims) {
  for (unsigned int i = 0; i < coeffs.size(); i++)
    if (coeffs[i] != 0 && i < cst.getNumDimVars())
      dims.push_back(i);
}

static affine::FlatAffineValueConstraints unionScratchpadIterDomains(
    mlir::ArrayRef<affine::FlatAffineValueConstraints> domains) {
  unsigned int maxDepth = 0;
  llvm::SetVector<mlir::Value> unionSymbols;
  SmallVector<mlir::Value, 4> domainSymbols;

  // Calculate the max depth and retrive all the symbols.
  for (const auto &domain : domains) {
    // depth
    maxDepth = std::max(domain.getNumDimVars(), maxDepth);
    // symbols
    domainSymbols.clear();
    domain.getValues(domain.getNumDimVars(), domain.getNumDimAndSymbolVars(),
                     &domainSymbols);
    for (mlir::Value sym : domainSymbols)
      unionSymbols.insert(sym);
  }

  // Create the union domain with maxDepth number of dims and the union of all
  // symbols appeared.
  affine::FlatAffineValueConstraints unionDomain(
      /*numDims=*/maxDepth,
      /*numSymbols=*/unionSymbols.size());
  for (auto symbol : enumerate(unionSymbols))
    unionDomain.setValue(symbol.index() + unionDomain.getNumDimVars(),
                         symbol.value());

  // Merge constraints. Only consider inequalities and those with single dim and
  // multiple symbols involved.
  for (const auto &domain : domains) {
    // TODO: Should deal with local IDs later.
    assert(domain.getNumLocalVars() == 0);

    for (unsigned int i = 0; i < domain.getNumInequalities(); i++) {
      auto mapped = llvm::map_range(domain.getInequality(i),
                                    [&](auto a) { return (int64_t)a; });
      SmallVector<int64_t> inEqVec(mapped.begin(), mapped.end());
      ArrayRef<int64_t> inEq(inEqVec.begin(), inEqVec.end());

      SmallVector<int64_t, 4> nonZeroDims;
      getNonZeroDims(inEq, domain, nonZeroDims);

      // TODO: Will cover this scenario later.
      assert(nonZeroDims.size() == 1);

      SmallVector<int64_t, 4> newInEq(unionDomain.getNumCols(), 0);
      // Merge dims
      for (unsigned int j = 0; j < unionDomain.getNumDimVars(); j++)
        if (j == nonZeroDims[0]) // only one nonzero dim is allowed.
          newInEq[j] = inEq[j];
      // Merge symbols
      for (unsigned int j = 0; j < domain.getNumSymbolVars(); j++) {
        mlir::Value symbol = domain.getValue(j + domain.getNumDimVars());
        unsigned int pos = 0;
        auto res = unionDomain.findVar(symbol, &pos);
        assert(res);
        newInEq[pos] = inEq[j + domain.getNumDimVars()];
      }
      // Merge constant
      newInEq[unionDomain.getNumCols() - 1] = inEq[domain.getNumCols() - 1];

      unionDomain.addInequality(newInEq);
    }
  }

  // Post-process
  unionDomain.removeRedundantInequalities();

  return unionDomain;
}

static void getLowerOrUpperBound(
    unsigned int dimId, bool isUpper,
    const affine::FlatAffineValueConstraints &domain, mlir::AffineMap &affMap,
    llvm::SmallVectorImpl<mlir::Value> &operands, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  SmallVector<mlir::AffineExpr, 4> exprs;
  MapVector<mlir::Value, unsigned int> symToPos;

  for (unsigned int i = 0; i < domain.getNumInequalities(); i++) {
    auto mapped = llvm::map_range(domain.getInequality(i),
                                  [&](auto a) { return (int64_t)a; });
    SmallVector<int64_t> inEqVec(mapped.begin(), mapped.end());
    ArrayRef<int64_t> inEq(inEqVec.begin(), inEqVec.end());

    SmallVector<int64_t, 4> nonZeroDims;
    getNonZeroDims(inEq, domain, nonZeroDims);
    assert(nonZeroDims.size() == 1);

    if (nonZeroDims[0] != dimId || (isUpper && inEq[dimId] > 0) ||
        (!isUpper && inEq[dimId] < 0))
      continue;

    mlir::AffineExpr expr = b.getAffineConstantExpr(0);

    for (unsigned int j = 0; j < domain.getNumSymbolVars(); j++) {
      mlir::Value symbol = domain.getValue(j + domain.getNumDimVars());
      int numSymbols = symToPos.size();
      if (!symToPos.count(symbol)) {
        symToPos[symbol] = numSymbols;
        operands.push_back(symbol);
      }

      mlir::AffineExpr term = inEq[j + domain.getNumDimVars()] *
                              b.getAffineSymbolExpr(symToPos[symbol]);
      if (isUpper)
        expr = expr + term;
      else
        expr = expr - term;
    }

    mlir::AffineExpr constExpr =
        b.getAffineConstantExpr(inEq[domain.getNumCols() - 1]);
    if (isUpper)
      expr = expr + constExpr + b.getAffineConstantExpr(1);
    else
      expr = expr - constExpr;

    exprs.push_back(expr);
  }

  // Finally setup the map.
  affMap = mlir::AffineMap::get(0, operands.size(), exprs, b.getContext());
}

static mlir::Value findInsertionPointAfter(mlir::func::FuncOp f,
                                           mlir::Value spad,
                                           ArrayRef<mlir::Value> candidates) {
  DominanceInfo dom(f);
  for (auto v1 : candidates) {
    if (v1.isa<BlockArgument>())
      continue;

    bool dominatesOthers = false;
    for (auto v2 : candidates) {
      if (v2.isa<BlockArgument>())
        continue;

      if (v1 != v2 && dom.dominates(v1.getDefiningOp(), v2.getDefiningOp())) {
        dominatesOthers = true;
        break;
      }
    }

    if (!dominatesOthers)
      return v1;
  }
  return spad;
}

static memref::AllocaOp
createScratchpadAllocaOp(mlir::func::FuncOp f, mlir::Value spad,
                         const affine::FlatAffineValueConstraints &domain,
                         OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  SmallVector<mlir::Value, 4> symbols;
  domain.getValues(domain.getNumDimVars(), domain.getNumDimAndSymbolVars(),
                   &symbols);
  b.setInsertionPointAfterValue(findInsertionPointAfter(f, spad, symbols));

  mlir::MemRefType memRefType = mlir::MemRefType::get(
      SmallVector<int64_t, 4>(domain.getNumDimVars(), -1),
      spad.getType().cast<mlir::MemRefType>().getElementType());

  llvm::SmallVector<mlir::Value, 4> memSizes(domain.getNumDimVars());
  for (unsigned int i = 0; i < domain.getNumDimVars(); i++) {
    mlir::AffineMap lbMap, ubMap;
    llvm::SmallVector<mlir::Value, 4> lbOperands, ubOperands;

    getLowerOrUpperBound(i, true, domain, ubMap, ubOperands, b);
    getLowerOrUpperBound(i, false, domain, lbMap, lbOperands, b);

    mlir::Value lb =
        b.create<mlir::affine::AffineMinOp>(spad.getLoc(), lbMap, lbOperands);
    mlir::Value ub =
        b.create<mlir::affine::AffineMaxOp>(spad.getLoc(), ubMap, ubOperands);
    mlir::Value size = b.create<mlir::affine::AffineApplyOp>(
        spad.getLoc(),
        AffineMap::get(0, 2,
                       b.getAffineSymbolExpr(0) - b.getAffineSymbolExpr(1)),
        ValueRange({ub, lb}));
    memSizes[i] = size;
  }

  return b.create<memref::AllocaOp>(spad.getLoc(), memRefType, memSizes);
}

static void resetLoadAndStoreOpsToScratchpad(mlir::func::FuncOp f,
                                             mlir::Value spad, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  for (mlir::Operation *op : spad.getUsers()) {
    if (isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp>(op)) {
      llvm::SmallVector<mlir::affine::AffineForOp, 4> forOps;
      affine::getAffineForIVs(*op, &forOps);

      mlir::MemRefType memRefType = spad.getType().cast<mlir::MemRefType>();
      SmallVector<mlir::AffineExpr, 4> indices(memRefType.getShape().size(),
                                               b.getAffineConstantExpr(0));
      SmallVector<mlir::Value, 4> operands;

      b.setInsertionPointAfter(op);
      int numOperands = 0;
      for (auto forOp : enumerate(forOps)) {
        mlir::AffineMap lbMap = forOp.value().getLowerBoundMap();
        llvm::SmallVector<mlir::Value, 4> lbOperands{
            forOp.value().getLowerBoundOperands()};
        mlir::Value lb = b.create<mlir::affine::AffineApplyOp>(
            op->getLoc(), lbMap, lbOperands);

        indices[forOp.index()] = b.getAffineDimExpr(numOperands) -
                                 b.getAffineDimExpr(numOperands + 1);

        operands.push_back(forOp.value().getInductionVar());
        operands.push_back(lb);
        numOperands += 2;
      }

      mlir::AffineMap affMap =
          mlir::AffineMap::get(numOperands, 0, indices, b.getContext());

      if (isa<mlir::affine::AffineLoadOp>(op)) {
        mlir::affine::AffineLoadOp loadOp =
            dyn_cast<mlir::affine::AffineLoadOp>(op);
        mlir::affine::AffineLoadOp newLoadOp =
            b.create<mlir::affine::AffineLoadOp>(
                op->getLoc(), loadOp.getMemRef(), affMap, operands);
        loadOp.replaceAllUsesWith(newLoadOp.getOperation());
        loadOp.erase();
      } else {
        assert(isa<mlir::affine::AffineStoreOp>(op));
        mlir::affine::AffineStoreOp storeOp =
            dyn_cast<mlir::affine::AffineStoreOp>(op);
        b.create<mlir::affine::AffineStoreOp>(
            op->getLoc(), storeOp.getValueToStore(), storeOp.getMemRef(),
            affMap, operands);
        storeOp.erase();
      }
    }
  }
}

/// Expand scratchpad based on its deepest/widest loop nest.
/// TODO: allow expansion to a specific depth.
static void expandScratchpad(mlir::func::FuncOp f, mlir::Value spad,
                             OpBuilder &b) {
  mlir::SmallVector<affine::FlatAffineValueConstraints, 4> domains;
  getScratchpadIterDomains(spad, domains);
  affine::FlatAffineValueConstraints unionDomain =
      unionScratchpadIterDomains(domains);

  mlir::Value newSpad =
      annotateScratchpad(createScratchpadAllocaOp(f, spad, unionDomain, b));
  spad.replaceAllUsesWith(newSpad);
  resetLoadAndStoreOpsToScratchpad(f, newSpad, b);
  spad.getDefiningOp()->erase();
}

static void arrayExpansion(mlir::func::FuncOp f, OpBuilder &b) {
  SmallVector<mlir::Value, 4> spads;
  findAllScratchpads(f, spads);
  for (mlir::Value spad : spads)
    expandScratchpad(f, spad, b);
}

class ArrayExpansionPass
    : public mlir::PassWrapper<ArrayExpansionPass,
                               OperationPass<mlir::func::FuncOp>> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    arrayExpansion(f, b);
  }
};

void polymer::registerRegToMemPass() {
  PassPipelineRegistration<>(
      "reg2mem", "Demote register to memref.",
      [](OpPassManager &pm) { pm.addPass(std::make_unique<RegToMemPass>()); });

  PassPipelineRegistration<>("insert-redundant-load",
                             "Insert redundant affine.load to avoid "
                             "creating unnecessary scratchpads.",
                             [](OpPassManager &pm) {
                               pm.addPass(
                                   std::make_unique<InsertRedundantLoadPass>());
                             });

  PassPipelineRegistration<>(
      "demote-loop-reduction", "Demote reduction to normal affine.for",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<DemoteLoopReductionPass>());
      });

  PassPipelineRegistration<>(
      "array-expansion",
      "Expand scratchpad to allow independant access across iteration domain.",
      [](mlir::OpPassManager &pm) {
        pm.addPass(std::make_unique<ArrayExpansionPass>());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

std::unique_ptr<Pass> polymer::createRegToMemPass() {
  return std::make_unique<RegToMemPass>();
}
