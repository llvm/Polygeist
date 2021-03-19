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

  // TODO: remove me or rename me.
  bool isAffine(scf::ForOp loop) const {
    // return true;
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

  bool isConstantLike(Value val) const {
    Operation *op = val.getDefiningOp();
    if (!op)
      return false;
    return op->getNumOperands() == 0 && op->getNumResults() == 1 &&
           op->hasTrait<OpTrait::ConstantLike>();
  }

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (isAffine(loop)) {
      OpBuilder builder(loop);

      AffineForOp affineLoop = rewriter.create<AffineForOp>(
          loop.getLoc(), loop.getLowerBound(), builder.getSymbolIdentityMap(),
          loop.getUpperBound(), builder.getSymbolIdentityMap(),
          getStep(loop.getStep()), loop.getIterOperands());

      if (!canonicalizeLoopBounds(affineLoop))
        return failure();

      // constant should be ok too.
      if (!llvm::all_of(affineLoop.getOperands(), [this](Value operand) {
            return isValidDim(operand) || isConstantLike(operand);
          }))
        return failure();

      SmallVector<Value, 4> newBlockTransferArgs;
      newBlockTransferArgs.reserve(1 + loop.getNumIterOperands());
      Value iv = affineLoop.getInductionVar();
      newBlockTransferArgs.push_back(iv);
      llvm::append_range(newBlockTransferArgs, affineLoop.getIterOperands());

      Block &newBlock = affineLoop.region().front();

      // The terminator is added if the iterator args are not provided.
      // see the ::build method.
      if (affineLoop.getNumIterOperands() == 0) {
        auto affineYiledOp = newBlock.getTerminator();
        rewriter.eraseOp(affineYiledOp);
      }

      Block &oldBlock = loop.region().front();
      assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
             "unexpected argument size mismatch");

      auto fixYield = [&](scf::YieldOp mergedTerminator) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(mergedTerminator);
        rewriter.create<AffineYieldOp>(mergedTerminator.getLoc(),
                                       mergedTerminator.getOperands());
      };

      rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
      auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      fixYield(mergedYieldOp);
      rewriter.eraseOp(mergedYieldOp);
      rewriter.replaceOp(loop, affineLoop.getResults());
      return success();
    }
    return failure();
  }
};

void RaiseSCFToAffine::runOnFunction() {
  ConversionTarget target(getContext());
  target
      .addLegalDialect<AffineDialect, StandardOpsDialect, LLVM::LLVMDialect>();

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<ForOpRaising>(&getContext());

  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createRaiseSCFToAffinePass() {
  return std::make_unique<RaiseSCFToAffine>();
}
