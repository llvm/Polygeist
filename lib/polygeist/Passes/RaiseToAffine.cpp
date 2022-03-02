#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-affine"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct RaiseSCFToAffine : public SCFRaiseToAffineBase<RaiseSCFToAffine> {
  void runOnOperation() override;
};
} // namespace

struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // TODO: remove me or rename me.
  bool isAffine(scf::ForOp loop) const {
    // return true;
    // enforce step to be a ConstantIndexOp (maybe too restrictive).
    return isa_and_nonnull<ConstantIndexOp>(loop.getStep().getDefiningOp());
  }

  void canonicalizeLoopBounds(AffineForOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    fully2ComposeAffineMapAndOperands(&lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(&ubMap, &ubOperands);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    if (lbMap != prevLbMap)
      forOp.setLowerBound(lbOperands, lbMap);
    if (ubMap != prevUbMap)
      forOp.setUpperBound(ubOperands, ubMap);
  }

  int64_t getStep(mlir::Value value) const {
    ConstantIndexOp cstOp = value.getDefiningOp<ConstantIndexOp>();
    assert(cstOp && "expect non-null operation");
    return cstOp.value();
  }

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (isAffine(loop)) {
      OpBuilder builder(loop);

      if (!isValidIndex(loop.getLowerBound())) {
        return failure();
      }

      if (!isValidIndex(loop.getUpperBound())) {
        return failure();
      }

      AffineForOp affineLoop = rewriter.create<AffineForOp>(
          loop.getLoc(), loop.getLowerBound(), builder.getSymbolIdentityMap(),
          loop.getUpperBound(), builder.getSymbolIdentityMap(),
          getStep(loop.getStep()), loop.getIterOperands());

      canonicalizeLoopBounds(affineLoop);

      auto mergedYieldOp =
          cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

      Block &newBlock = affineLoop.region().front();

      // The terminator is added if the iterator args are not provided.
      // see the ::build method.
      if (affineLoop.getNumIterOperands() == 0) {
        auto affineYieldOp = newBlock.getTerminator();
        rewriter.eraseOp(affineYieldOp);
      }

      rewriter.updateRootInPlace(loop, [&] {
        affineLoop.region().front().getOperations().splice(
            affineLoop.region().front().getOperations().begin(),
            loop.getRegion().front().getOperations());

        for (auto pair : llvm::zip(affineLoop.region().front().getArguments(),
                                   loop.getRegion().front().getArguments())) {
          std::get<1>(pair).replaceAllUsesWith(std::get<0>(pair));
        }
      });

      rewriter.setInsertionPoint(mergedYieldOp);
      rewriter.create<AffineYieldOp>(mergedYieldOp.getLoc(),
                                     mergedYieldOp.getOperands());
      rewriter.eraseOp(mergedYieldOp);

      rewriter.replaceOp(loop, affineLoop.getResults());

      return success();
    }
    return failure();
  }
};

struct ParallelOpRaising : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  // TODO: remove me or rename me.
  bool isAffine(scf::ParallelOp loop) const {
    for (auto step : loop.getStep())
      if (!step.getDefiningOp<ConstantIndexOp>())
        return false;
    return true;
  }

  void canonicalizeLoopBounds(AffineParallelOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundsOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundsOperands());

    auto lbMap = forOp.lowerBoundsMap();
    auto ubMap = forOp.upperBoundsMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    fully2ComposeAffineMapAndOperands(&lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);

    fully2ComposeAffineMapAndOperands(&ubMap, &ubOperands);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);

    if (lbMap != prevLbMap)
      forOp.setLowerBounds(lbOperands, lbMap);
    if (ubMap != prevUbMap)
      forOp.setUpperBounds(ubOperands, ubMap);
  }

  LogicalResult matchAndRewrite(scf::ParallelOp loop,
                                PatternRewriter &rewriter) const final {
    OpBuilder builder(loop);

    if (loop.getResults().size())
      return failure();

    if (!llvm::all_of(loop.getLowerBound(), isValidIndex)) {
      return failure();
    }

    if (!llvm::all_of(loop.getUpperBound(), isValidIndex)) {
      return failure();
    }

    SmallVector<int64_t> steps;
    for (auto step : loop.getStep())
      if (auto cst = step.getDefiningOp<ConstantIndexOp>())
        steps.push_back(cst.value());
      else
        return failure();

    ArrayRef<AtomicRMWKind> reductions;
    SmallVector<AffineMap> bounds;
    for (size_t i = 0; i < loop.getLowerBound().size(); i++)
      bounds.push_back(AffineMap::get(
          /*dimCount=*/0, /*symbolCount=*/loop.getLowerBound().size(),
          builder.getAffineSymbolExpr(i)));
    AffineParallelOp affineLoop = rewriter.create<AffineParallelOp>(
        loop.getLoc(), loop.getResultTypes(), reductions, bounds,
        loop.getLowerBound(), bounds, loop.getUpperBound(),
        steps); //, loop.getInitVals());

    canonicalizeLoopBounds(affineLoop);

    auto mergedYieldOp =
        cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block &newBlock = affineLoop.region().front();

    // The terminator is added if the iterator args are not provided.
    // see the ::build method.
    if (affineLoop.getResults().size() == 0) {
      auto affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    rewriter.updateRootInPlace(loop, [&] {
      affineLoop.region().front().getOperations().splice(
          affineLoop.region().front().getOperations().begin(),
          loop.getRegion().front().getOperations());

      for (auto pair : llvm::zip(affineLoop.region().front().getArguments(),
                                 loop.getRegion().front().getArguments())) {
        std::get<1>(pair).replaceAllUsesWith(std::get<0>(pair));
      }
    });

    rewriter.setInsertionPoint(mergedYieldOp);
    rewriter.create<AffineYieldOp>(mergedYieldOp.getLoc(),
                                   mergedYieldOp.getOperands());
    rewriter.eraseOp(mergedYieldOp);

    rewriter.replaceOp(loop, affineLoop.getResults());

    return success();
  }
};

void RaiseSCFToAffine::runOnOperation() {
  ConversionTarget target(getContext());
  target
      .addLegalDialect<AffineDialect, func::FuncDialect, LLVM::LLVMDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ForOpRaising, ParallelOpRaising>(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseSCFToAffinePass() {
  return std::make_unique<RaiseSCFToAffine>();
}
} // namespace polygeist
} // namespace mlir
