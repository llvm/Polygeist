#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IntegerSet.h"

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;

namespace {
struct AffineCFGPass
    : public AffineCFGBase<AffineCFGPass> {
  void runOnFunction() override;
};
} // namespace

static bool inAffine(Operation *op) {
  auto *curOp = op;
  while (auto *parentOp = curOp->getParentOp()) {
    if (isa<AffineForOp, AffineParallelOp>(parentOp))
      return true;
    curOp = parentOp;
  }
  return false;
}

void AffineCFGPass::runOnFunction() {
  getFunction().walk([](scf::IfOp ifOp) {
    if (inAffine(ifOp) && ifOp.results().size() == 0) {
        OpBuilder b(ifOp);
        std::vector<AffineExpr> exprs = {getAffineSymbolExpr(0, ifOp.getContext()) };
        bool eqflags[1] = {false};
        auto iset = IntegerSet::get(/*dim*/0, /*symbol*/1,
                        exprs,
                        eqflags);
        std::vector<mlir::Value> opvals = { ifOp.condition() };
        auto affineIfOp = b.create<AffineIfOp>(ifOp.getLoc(), iset,
                                                opvals,
                                                /*elseBlock=*/true);

        affineIfOp.thenRegion().takeBody(ifOp.thenRegion());
        affineIfOp.elseRegion().takeBody(ifOp.elseRegion());
        affineIfOp.walk([](scf::YieldOp yop) {
            OpBuilder b(yop);
            b.create<AffineYieldOp>(yop.getLoc(), yop.results());
            yop.erase();
        });
        ifOp.erase();
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}
