#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-store"

using namespace mlir;

namespace {
struct AffineStorePass : public AffineStoreBase<AffineStorePass> {
  void runOnFunction() override;
};
struct AffineLoadPass : public AffineLoadBase<AffineLoadPass> {
  void runOnFunction() override;
};
} // namespace

static bool isValidIndex(Value index) {
  if (index.isa<BlockArgument>() &&
      isa<AffineForOp>(index.cast<BlockArgument>().getOwner()->getParentOp()))
    return true;
  if (index.getDefiningOp<ConstantIndexOp>())
    return true;
  return false;
}

static bool isValidAffineOperation(OperandRange indexes) {
  return llvm::all_of(indexes, [](Value index) { return isValidIndex(index); });
}

static void replaceStore(StoreOp store) {
  OpBuilder builder(store);
  Location loc = store.getLoc();
  builder.create<AffineStoreOp>(loc, store.getValueToStore(), store.getMemRef(),
                                store.getIndices());
  store.erase();
}

static void replaceLoad(LoadOp load) {
  OpBuilder builder(load);
  Location loc = load.getLoc();
  AffineLoadOp affineLoad =
      builder.create<AffineLoadOp>(loc, load.getMemRef(), load.getIndices());
  load.getResult().replaceAllUsesWith(affineLoad.getResult());
  load.erase();
}

void AffineStorePass::runOnFunction() {
  getFunction().walk([](StoreOp store) {
    if (isValidAffineOperation(store.getIndices())) {
      LLVM_DEBUG(llvm::dbgs() << "  affine store checks -> ok\n");
      replaceStore(store);
    }
  });
}

void AffineLoadPass::runOnFunction() {
  getFunction().walk([](LoadOp load) {
    if (isValidAffineOperation(load.getIndices())) {
      LLVM_DEBUG(llvm::dbgs() << "  affine load checks -> ok\n");
      replaceLoad(load);
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineStorePass() {
  return std::make_unique<AffineStorePass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineLoadPass() {
  return std::make_unique<AffineLoadPass>();
}
