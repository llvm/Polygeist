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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "reg2mem"

using DefToUsesMap =
    llvm::DenseMap<mlir::Value, llvm::SetVector<mlir::Operation *>>;

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
            isa<mlir::AllocOp, mlir::DimOp, mlir::AffineApplyOp>(defOp))
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

/// Creates a single-entry scratchpad memory that stores values from the
/// defining point and can be loaded when needed at the uses.
static mlir::AllocaOp createScratchpadAllocaOp(mlir::OpResult val,
                                               mlir::OpBuilder &b) {
  // Sanity checks on the defining op.
  mlir::Operation *defOp = val.getOwner();

  // Set the allocation point after where the val is defined.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(defOp);

  // The memref shape is 1 and the type is derived from val.
  return b.create<mlir::AllocaOp>(defOp->getLoc(),
                                  MemRefType::get({1}, val.getType()));
}

/// Creata an AffineStoreOp for the value to be stored on the scratchpad.
static mlir::AffineStoreOp createScratchpadStoreOp(mlir::Value valToStore,
                                                   mlir::AllocaOp allocaOp,
                                                   mlir::OpBuilder &b) {
  // Create a storeOp to the memref using address 0. The new storeOp will be
  // placed right after the allocaOp, and its location is hinted by allocaOp.
  // Here we assume that allocaOp is dominated by the defining op of valToStore.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(allocaOp);

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
  DefToUsesMap defToUses;
  // Get the mapping from a value to its uses that are in a different block as
  // where the value itself is defined.
  mapDefToUses(f, defToUses);
  // Make sure every def will have a single use in each block.
  filterUsesInSameBlock(defToUses);

  // Handle each def-use pair in in the current function.
  for (const auto &defUsesPair : defToUses) {
    // The value to be stored in a scratchpad.
    mlir::Value val = defUsesPair.first;

    // Create the alloca op for the scratchpad.
    mlir::AllocaOp allocaOp =
        createScratchpadAllocaOp(val.dyn_cast<mlir::OpResult>(), b);

    // Create the store op that stores val into the scratchpad for future uses.
    mlir::AffineStoreOp storeOp = createScratchpadStoreOp(val, allocaOp, b);

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

namespace {

class RegToMemPass
    : public mlir::PassWrapper<RegToMemPass, OperationPass<mlir::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());

    demoteRegisterToMemory(f, builder);
  }
};

} // namespace

void polymer::registerRegToMemPass() {
  PassRegistration<RegToMemPass>("reg2mem", "Demote register to memref.");
}
