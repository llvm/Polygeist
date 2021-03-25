//===- Reg2Mem.cc - reg2mem transformation --------------------------------===//
//
// This file implements the reg2mem transformation pass.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/Reg2Mem.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

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
static void mapDefToUses(mlir::FuncOp f, DefToUsesMap &defToUses) {
  f.walk([&](mlir::Operation *useOp) {
    // Op that belongs to AffineWriteOpInterface (e.g., affine.store) or has
    // recursive side effects will be treated as .
    if (!isa<mlir::AffineWriteOpInterface>(useOp) &&
        !useOp->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
      return;
    // Should filter out for and if ops.
    if (isa<mlir::AffineForOp, mlir::AffineIfOp>(useOp))
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
        // - DimOp/AffineApplyOp: indices and bounds shouldn't be loaded from
        // memory, otherwise, it would mess up with the dependence analysis.
        if (!defOp ||
            isa<mlir::AllocOp, mlir::AllocaOp, mlir::DimOp, mlir::ConstantOp,
                mlir::AffineApplyOp, mlir::IndexCastOp>(defOp))
          continue;

        // The block that defines the value is different from the block of the
        // current op.
        if (v.getParentBlock() != op->getBlock())
          defToUses[v].insert(op);

        // No need to look at the operands of the following list of operations.
        if (!isa<mlir::AffineLoadOp>(defOp))
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
static mlir::AllocaOp annotateScratchpad(mlir::AllocaOp allocaOp) {
  allocaOp.setAttr("scop.scratchpad",
                   mlir::UnitAttr::get(allocaOp.getContext()));
  return allocaOp;
}

/// Creates a single-entry scratchpad memory that stores values from the
/// defining point and can be loaded when needed at the uses.
static mlir::AllocaOp createScratchpadAllocaOp(mlir::OpResult val,
                                               mlir::OpBuilder &b,
                                               mlir::Block *entryBlock) {
  // Sanity checks on the defining op.
  mlir::Operation *defOp = val.getOwner();

  // Set the allocation point after where the val is defined.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entryBlock);

  // The memref shape is 1 and the type is derived from val.
  return annotateScratchpad(b.create<mlir::AllocaOp>(
      defOp->getLoc(), MemRefType::get({1}, val.getType())));
}

/// Creata an AffineStoreOp for the value to be stored on the scratchpad.
static mlir::AffineStoreOp createScratchpadStoreOp(mlir::Value valToStore,
                                                   mlir::AllocaOp allocaOp,
                                                   mlir::OpBuilder &b) {
  // Create a storeOp to the memref using address 0. The new storeOp will be
  // placed right after the allocaOp, and its location is hinted by allocaOp.
  // Here we assume that allocaOp is dominated by the defining op of valToStore.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(valToStore);

  return b.create<mlir::AffineStoreOp>(
      allocaOp.getLoc(), valToStore, allocaOp.getResult(),
      b.getConstantAffineMap(0), std::vector<mlir::Value>());
}

/// Create an AffineLoadOp for the value stored in the scratchpad. The insertion
/// point will be at the beginning of the block of the useOp, such that all the
/// subsequent uses of the Value in the scratchpad can re-use the same load
/// result. Note that we don't check whether the useOp is still using the
/// original value that is stored in the scratchpad (some replacement could
/// happen already), you need to do that before calling this function to avoid
/// possible redundancy. This function won't replace uses.
static mlir::AffineLoadOp createScratchpadLoadOp(mlir::AllocaOp allocaOp,
                                                 mlir::Operation *useOp,
                                                 mlir::OpBuilder &b) {
  // The insertion point will be at the beginning of the parent block for useOp.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(useOp->getBlock());
  // The location is set to be the useOp that will finally use this newly
  // created load op. The address is set to be 0 since the memory has only one
  // element in it. You will need to replace the input to useOp outside.
  return b.create<mlir::AffineLoadOp>(useOp->getLoc(), allocaOp.getResult(),
                                      b.getConstantAffineMap(0),
                                      std::vector<mlir::Value>());
}

static void demoteRegisterToMemory(mlir::FuncOp f, OpBuilder &b) {
  if (f.getBlocks().size() == 0)
    return;

  DefToUsesMap defToUses;
  Block &entryBlock = *f.getBody().begin();

  // Get the mapping from a value to its uses that are in a
  // different block as where the value itself is defined.
  mapDefToUses(f, defToUses);
  // Make sure every def will have a single use in each block.
  filterUsesInSameBlock(defToUses);

  // Handle each def-use pair in in the current function.
  for (const auto &defUsesPair : defToUses) {
    // The value to be stored in a scratchpad.
    mlir::Value val = defUsesPair.first;

    // Create the alloca op for the scratchpad.
    mlir::AllocaOp allocaOp = createScratchpadAllocaOp(
        val.dyn_cast<mlir::OpResult>(), b, &entryBlock);

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
      mlir::AffineLoadOp loadOp = createScratchpadLoadOp(allocaOp, useOp, b);

      // Replace the uses of val in the same region as useOp (or loadOp).
      val.replaceUsesWithIf(loadOp.getResult(), [&](mlir::OpOperand &operand) {
        mlir::Operation *currUseOp = operand.getOwner();
        //  Check the equivalence of the regions.
        return currUseOp->getParentRegion() == useOp->getParentRegion();
      });
    }
  }
}

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

/// Turns affine.if with else block into two affine.if. It works iteratively:
/// one affine.if op (with else) should be handled at one time.
static void separateAffineIfBlocks(mlir::FuncOp f, OpBuilder &b) {
  // Get the first affine.if operation that has an else block.
  auto findIfWithElse = [&](auto &f) {
    Operation *opFound = nullptr;
    f.walk([&](mlir::AffineIfOp ifOp) {
      if (!opFound && ifOp.hasElse()) {
        opFound = ifOp;
        return;
      }
    });
    return opFound;
  };

  Operation *op;
  while ((op = findIfWithElse(f)) != nullptr) {
    mlir::AffineIfOp ifOp = dyn_cast<mlir::AffineIfOp>(op);
    assert(ifOp.hasElse() && "The if op found should have an else block.");

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(ifOp);

    // Here we create two new if operations, one for the then block, one for the
    // else block.
    mlir::AffineIfOp newIfThenOp =
        cast<mlir::AffineIfOp>(b.clone(*ifOp.getOperation()));
    newIfThenOp.getElseBlock()->erase(); // TODO: can we don't create it?

    b.setInsertionPointAfter(newIfThenOp);
    mlir::AffineIfOp newIfElseOp =
        cast<mlir::AffineIfOp>(b.clone(*ifOp.getOperation()));

    // Build new integer set.
    newIfElseOp.setConditional(getIntegerSetForElse(ifOp.getIntegerSet(), b),
                               ifOp.getOperands());

    Block *oldThenBlock = newIfElseOp.getThenBlock();
    newIfElseOp.getElseBlock()->moveBefore(oldThenBlock);
    oldThenBlock->erase();

    ifOp.erase();
  }
}

namespace {

class RegToMemPass
    : public mlir::PassWrapper<RegToMemPass, OperationPass<mlir::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());

    separateAffineIfBlocks(f, builder);
    demoteRegisterToMemory(f, builder);
  }
};

} // namespace

/// TODO: value analysis
static void insertRedundantLoad(mlir::FuncOp f, OpBuilder &b) {
  DominanceInfo dom(f);

  SmallVector<mlir::AffineStoreOp, 4> storeOps;
  f.walk([&storeOps](mlir::AffineStoreOp op) { storeOps.push_back(op); });

  SetVector<mlir::Operation *> storeOpsToLoad;

  for (mlir::AffineStoreOp storeOp : storeOps) {
    Value valueToStore = storeOp.getValueToStore();
    Value memref = storeOp.getMemRef();

    // TODO: deal with the complexity here.
    for (mlir::Operation *user : valueToStore.getUsers()) {
      if (user == storeOp)
        continue;
      if (user->getBlock() != storeOp->getBlock())
        continue;

      //  This user is another storeOp that dominates the current storeOp.
      if (isa<mlir::AffineStoreOp>(user) && dom.dominates(user, storeOp)) {
        // ... and there is no other storeOp that is beging dominated in
        // between.
        bool hasDominatedMemStore = false;
        for (mlir::Operation *memUser : memref.getUsers()) {
          if (memUser == storeOp || memUser == user)
            continue;
          if (isa<mlir::AffineStoreOp>(memUser) &&
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
    mlir::AffineStoreOp storeOpToLoad = cast<mlir::AffineStoreOp>(op);

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(storeOpToLoad);

    Value value = storeOpToLoad.getValueToStore();
    Value memref = storeOpToLoad.getMemRef();
    mlir::AffineLoadOp loadOp = b.create<mlir::AffineLoadOp>(
        storeOpToLoad.getLoc(), memref, storeOpToLoad.getAffineMap(),
        storeOpToLoad.getMapOperands());

    value.replaceUsesWithIf(loadOp.getResult(), [=](mlir::OpOperand &operand) {
      return operand.getOwner() != op &&
             operand.getOwner()->getBlock() == op->getBlock();
    });
  }
}

namespace {
class InsertRedundantLoadPass
    : public mlir::PassWrapper<InsertRedundantLoadPass,
                               OperationPass<mlir::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    insertRedundantLoad(f, b);
  }
};

} // namespace

namespace {
using IterArgToMemMap = llvm::MapVector<mlir::Value, mlir::Value>;
}

static void findReductionLoops(mlir::FuncOp f,
                               SmallVectorImpl<mlir::AffineForOp> &forOps) {
  f.walk([&](mlir::AffineForOp forOp) {
    if (!forOp.getIterOperands().empty())
      forOps.push_back(forOp);
  });
}

static mlir::AffineYieldOp findYieldOp(mlir::AffineForOp forOp) {
  mlir::Operation *retOp;
  forOp.walk([&](mlir::AffineYieldOp yieldOp) { retOp = yieldOp; });

  assert(retOp != nullptr);
  assert(isa<mlir::AffineYieldOp>(retOp));

  return cast<mlir::AffineYieldOp>(retOp);
}

static mlir::Value createIterScratchpad(mlir::Value iterArg,
                                        IterArgToMemMap &iterArgToMem,
                                        mlir::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(forOp);

  assert(!iterArgToMem.count(iterArg) &&
         "A scratchpad has been created for the given iterArg");

  mlir::Value spad = b.create<mlir::AllocaOp>(
      forOp.getLoc(), MemRefType::get({1}, iterArg.getType()));
  iterArgToMem[iterArg] = spad;

  return spad;
}

static void storeInitValue(mlir::Value initVal, mlir::Value spad,
                           mlir::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(spad);
  b.create<mlir::AffineStoreOp>(spad.getLoc(), initVal, spad,
                                b.getConstantAffineMap(0), llvm::None);
}

static mlir::Value loadIterArg(mlir::Value iterArg,
                               IterArgToMemMap &iterArgToMem,
                               mlir::AffineForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(forOp.getBody());

  assert(iterArgToMem.count(iterArg));

  return b.create<mlir::AffineLoadOp>(forOp.getLoc(), iterArgToMem[iterArg],
                                      b.getConstantAffineMap(0), llvm::None);
}

static void replaceIterArg(mlir::Value origIterArg, mlir::Value spadIterArg) {
  origIterArg.replaceAllUsesWith(spadIterArg);
}

static void storeIterArg(int idx, mlir::Value spad, mlir::AffineYieldOp yieldOp,
                         OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(yieldOp);
  b.create<mlir::AffineStoreOp>(yieldOp.getLoc(), yieldOp.getOperand(idx), spad,
                                b.getConstantAffineMap(0), llvm::None);
}

static mlir::Value loadFinalIterVal(mlir::Value spad, mlir::AffineForOp forOp,
                                    OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  return b.create<mlir::AffineLoadOp>(forOp.getLoc(), spad,
                                      b.getConstantAffineMap(0), llvm::None);
}

static void replaceResult(int idx, mlir::AffineForOp forOp,
                          mlir::Value retVal) {
  mlir::Value origRetVal = forOp.getResult(idx);
  origRetVal.replaceAllUsesWith(retVal);
}

static mlir::AffineForOp cloneAffineForWithoutIterArgs(mlir::AffineForOp forOp,
                                                       OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  mlir::AffineForOp newForOp = b.create<mlir::AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), forOp.getStep());

  BlockAndValueMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  b.setInsertionPointToStart(newForOp.getBody());
  forOp.walk([&](mlir::Operation *op) {
    if (!isa<mlir::AffineYieldOp>(op) && op->getParentOp() == forOp)
      b.clone(*op, mapping);
  });

  return newForOp;
}

static void demoteLoopReduction(mlir::FuncOp f, mlir::AffineForOp forOp,
                                OpBuilder &b) {
  SmallVector<mlir::Value, 4> initVals{forOp.getIterOperands()};
  mlir::Block *body = forOp.getBody();
  mlir::AffineYieldOp yieldOp = findYieldOp(forOp);

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

static void demoteLoopReduction(mlir::FuncOp f, OpBuilder &b) {
  SmallVector<mlir::AffineForOp, 4> forOps;
  findReductionLoops(f, forOps);

  for (mlir::AffineForOp forOp : forOps)
    demoteLoopReduction(f, forOp, b);
}

namespace {
class DemoteLoopReductionPass
    : public mlir::PassWrapper<DemoteLoopReductionPass,
                               OperationPass<mlir::FuncOp>> {
public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    demoteLoopReduction(f, b);
  }
};

} // namespace

void polymer::registerRegToMemPass() {
  PassRegistration<RegToMemPass>("reg2mem", "Demote register to memref.");
  PassRegistration<InsertRedundantLoadPass>(
      "insert-redundant-load", "Insert redundant affine.load to avoid "
                               "creating unnecessary scratchpads.");
  PassRegistration<DemoteLoopReductionPass>(
      "demote-loop-reduction", "Demote reduction to normal affine.for");
}
