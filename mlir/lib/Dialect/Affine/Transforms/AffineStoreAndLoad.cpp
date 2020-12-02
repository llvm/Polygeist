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

void AffineStoreAndLoadPass::runOnFunction() {
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineStoreAndLoadPass() {
  return std::make_unique<AffineStoreAndLoadPass>();
}
