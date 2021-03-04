#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-affine"

using namespace mlir;

namespace {
struct RaiseSCFToAffine : public SCFRaiseToAffineBase<RaiseSCFToAffine> {
  void runOnFunction() override;
};
} // namespace

struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  bool isAffine(scf::ForOp loop) const {
    // if we have loop-carried values do not raise for now.
    if (loop.hasIterOperands())
      return false;
    // enforce step to be a ConstantIndexOp (maybe too restrictive).
    return isa_and_nonnull<ConstantIndexOp>(loop.getStep().getDefiningOp());
  }

  bool canonicalizeLoopBounds(AffineForOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    fullyComposeAffineMapAndOperands(&lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fullyComposeAffineMapAndOperands(&ubMap, &ubOperands);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    // Any canonicalization change always leads to updated map(s).
    if (lbMap == prevLbMap && ubMap == prevUbMap)
      return false;

    if (lbMap != prevLbMap)
      forOp.setLowerBound(lbOperands, lbMap);
    if (ubMap != prevUbMap)
      forOp.setUpperBound(ubOperands, ubMap);

    return true;
  }

  int64_t getStep(mlir::Value value) const {
    ConstantIndexOp cstOp = value.getDefiningOp<ConstantIndexOp>();
    assert(cstOp && "expect non-null operation");
    return cstOp.getValue();
  }

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (isAffine(loop)) {
      OpBuilder builder(loop);

      AffineForOp affineLoop = rewriter.create<AffineForOp>(
          loop.getLoc(), loop.getLowerBound(), builder.getSymbolIdentityMap(),
          loop.getUpperBound(), builder.getSymbolIdentityMap(),
          getStep(loop.getStep()), llvm::None);

      if (!canonicalizeLoopBounds(affineLoop))
        return failure();

      Value iv = loop.getInductionVar();
      Region &region = affineLoop.getLoopBody();
      rewriter.inlineRegionBefore(loop.getRegion(), region, region.begin());
      Operation &terminator = region.front().back();
      rewriter.eraseOp(&terminator);
      rewriter.mergeBlocks(&region.back(), &region.front(), iv);
      rewriter.eraseOp(loop);
      return success();
    }
    return failure();
  }
};

void RaiseSCFToAffine::runOnFunction() {
  ConversionTarget target(getContext());
  target
      .addLegalDialect<AffineDialect, StandardOpsDialect, LLVM::LLVMDialect>();

  OwningRewritePatternList patterns;
  patterns.insert<ForOpRaising>(&getContext());

  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createRaiseSCFToAffinePass() {
  return std::make_unique<RaiseSCFToAffine>();
}
