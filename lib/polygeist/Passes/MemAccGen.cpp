#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-access-generation"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace mlir::affine;

namespace {
struct MemAccGenPass : public MemAccGenBase<MemAccGenPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {
    
}

void MemAccGenPass::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
//   rpl.add<AffineForReductionIter>(getOperation()->getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

namespace mlir {
namespace polygeist {
    std::unique_ptr<Pass> mlir::polygeist::createMemAccGenPass() {
        return std::make_unique<MemAccGenPass>();
    }
}
} // end namespace mlir::polygeist