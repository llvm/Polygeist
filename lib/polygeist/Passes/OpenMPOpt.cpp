#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct OpenMPOpt : public OpenMPOptPassBase<OpenMPOpt> {
  void runOnOperation() override;
};
} // namespace

/// Merge any consecutive parallel's
///
///    omp.parallel {
///       codeA();
///    }
///    omp.parallel {
///       codeB();
///    }
///
///  becomes
///
///    omp.parallel {
///       codeA();
///       omp.barrier
///       codeB();
///    }
struct CombineParallel : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp nextParallel,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextParallel->getBlock();
    if (nextParallel == &parent->front())
      return failure();

    auto prevParallel = dyn_cast<omp::ParallelOp>(nextParallel->getPrevNode());
    if (!prevParallel)
      return failure();

    rewriter.setInsertionPointToEnd(&prevParallel.getRegion().front());
    rewriter.replaceOpWithNewOp<omp::BarrierOp>(
        prevParallel.getRegion().front().getTerminator(), TypeRange());
    rewriter.mergeBlocks(&nextParallel.getRegion().front(),
                         &prevParallel.getRegion().front());
    rewriter.eraseOp(nextParallel);
    return success();
  }
};

struct ParallelForInterchange : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp nextParallel,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextParallel->getBlock();
    if (parent->getOperations().size() != 2)
      return failure();

    auto prevFor = dyn_cast<scf::ForOp>(nextParallel->getParentOp());
    if (!prevFor || prevFor->getResults().size())
      return failure();

    nextParallel->moveBefore(prevFor);
    auto yield = nextParallel.getRegion().front().getTerminator();
    auto contents =
        rewriter.splitBlock(&nextParallel.getRegion().front(),
                            nextParallel.getRegion().front().begin());
    rewriter.mergeBlockBefore(contents, &prevFor.getBody()->front());
    rewriter.setInsertionPoint(prevFor.getBody()->getTerminator());
    rewriter.create<omp::BarrierOp>(nextParallel.getLoc());
    rewriter.setInsertionPointToEnd(&nextParallel.getRegion().front());
    auto newYield = rewriter.clone(*yield);
    rewriter.eraseOp(yield);
    prevFor->moveBefore(newYield);
    return success();
  }
};

void OpenMPOpt::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<CombineParallel, ParallelForInterchange>(
      getOperation()->getContext());
  GreedyRewriteConfig config;
  config.maxIterations = 47;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::createOpenMPOptPass() {
  return std::make_unique<OpenMPOpt>();
}
