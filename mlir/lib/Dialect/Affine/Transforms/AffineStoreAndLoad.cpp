#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-store-load"

using namespace mlir;

namespace {
struct AffineStoreAndLoadPass
    : public AffineStoreAndLoadBase<AffineStoreAndLoadPass> {
  void runOnFunction() override;
};
} // namespace

static bool isAffineForArg(Value index) {
  return (
      index.isa<BlockArgument>() &&
      isa<AffineForOp>(index.cast<BlockArgument>().getOwner()->getParentOp()));
}

static bool comesFromAffineOperation(Operation *op, Value curIndex,
                                     AffineExpr &expr,
                                     SmallVector<Value, 2> &newIndexes) {
  if (isAffineForArg(curIndex)) {
    OpBuilder builder(op);
    AffineApplyOp newIndex = builder.create<AffineApplyOp>(
        op->getLoc(), AffineMap::get(1, 0, expr), curIndex);
    newIndexes.push_back(newIndex);
    return true;
  }

  Operation *curOp = curIndex.getDefiningOp();
  if ((!isa<AddIOp>(curOp)) && (!isa<MulIOp>(curOp)) && (!isa<SubIOp>(curOp))) {
    return false;
  }

  Value nonCstOperand = nullptr;
  for (auto operand : curOp->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast_or_null<ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(curOp))
        expr = expr + constantIndexOp.getValue();
      if (isa<SubIOp>(curOp))
        expr = expr - constantIndexOp.getValue();
      else
        expr = expr * constantIndexOp.getValue();
    } else
      nonCstOperand = operand;
  }
  return comesFromAffineOperation(op, nonCstOperand, expr, newIndexes);
}

static bool isValidIndex(Operation *op, Value index,
                         SmallVector<Value, 2> &newIndexes) {
  if (isAffineForArg(index)) {
    newIndexes.push_back(index);
    return true;
  }
  if (index.getDefiningOp<ConstantIndexOp>()) {
    newIndexes.push_back(index);
    return true;
  }
  AffineExpr i;
  bindDims(op->getContext(), i);
  return comesFromAffineOperation(op, index, i, newIndexes);
}

static bool isValidAffineOperation(Operation *op, OperandRange indexes,
                                   SmallVector<Value, 2> &newIndexes) {
  return llvm::all_of(indexes, [&op, &newIndexes](Value index) {
    return isValidIndex(op, index, newIndexes);
  });
}

static void replaceStore(StoreOp store,
                         const SmallVector<Value, 2> &newIndexes) {
  OpBuilder builder(store);
  Location loc = store.getLoc();
  builder.create<AffineStoreOp>(loc, store.getValueToStore(), store.getMemRef(),
                                newIndexes);
  store.erase();
}

static void replaceLoad(LoadOp load, const SmallVector<Value, 2> &newIndexes) {
  OpBuilder builder(load);
  Location loc = load.getLoc();
  AffineLoadOp affineLoad =
      builder.create<AffineLoadOp>(loc, load.getMemRef(), newIndexes);
  load.getResult().replaceAllUsesWith(affineLoad.getResult());
  load.erase();
}

void AffineStoreAndLoadPass::runOnFunction() {
  getFunction().walk([](StoreOp store) {
    SmallVector<Value, 2> newIndexes;
    newIndexes.reserve(store.getIndices().size());
    if (isValidAffineOperation(store, store.getIndices(), newIndexes)) {
      LLVM_DEBUG(llvm::dbgs() << "  affine store checks -> ok\n");
      replaceStore(store, newIndexes);
    }
  });

  getFunction().walk([](LoadOp load) {
    SmallVector<Value, 2> newIndexes;
    newIndexes.reserve(load.getIndices().size());
    if (isValidAffineOperation(load, load.getIndices(), newIndexes)) {
      LLVM_DEBUG(llvm::dbgs() << "  affine load checks -> ok\n");
      replaceLoad(load, newIndexes);
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineStoreAndLoadPass() {
  return std::make_unique<AffineStoreAndLoadPass>();
}
