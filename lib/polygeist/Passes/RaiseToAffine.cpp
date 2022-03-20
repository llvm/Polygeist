#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
    return isValidSymbol(loop.getStep());
  }

  void canonicalizeLoopBounds(PatternRewriter &rewriter,
                              AffineForOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    if (lbMap != prevLbMap)
      forOp.setLowerBound(lbOperands, lbMap);
    if (ubMap != prevUbMap)
      forOp.setUpperBound(ubOperands, ubMap);
  }

  int64_t getStep(mlir::Value value) const {
    ConstantIndexOp cstOp = value.getDefiningOp<ConstantIndexOp>();
    if (cstOp)
      return cstOp.value();
    else
      return 1;
  }

  AffineMap getMultiSymbolIdentity(Builder &B, unsigned rank) const {
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(B.getAffineSymbolExpr(i));
    return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                          B.getContext());
  }
  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (isAffine(loop)) {
      OpBuilder builder(loop);

      SmallVector<Value> lbs;
      {
        SmallVector<Value> todo = {loop.getLowerBound()};
        while (todo.size()) {
          auto cur = todo.back();
          todo.pop_back();
          if (isValidIndex(cur)) {
            lbs.push_back(cur);
            continue;
          } else if (auto selOp = cur.getDefiningOp<SelectOp>()) {
            // LB only has max of operands
            if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
              if (cmp.getLhs() == selOp.getTrueValue() &&
                  cmp.getRhs() == selOp.getFalseValue() &&
                  cmp.getPredicate() == CmpIPredicate::sge) {
                todo.push_back(cmp.getLhs());
                todo.push_back(cmp.getRhs());
                continue;
              }
            }
          }
          return failure();
        }
      }

      SmallVector<Value> ubs;
      {
        SmallVector<Value> todo = {loop.getUpperBound()};
        while (todo.size()) {
          auto cur = todo.back();
          todo.pop_back();
          if (isValidIndex(cur)) {
            ubs.push_back(cur);
            continue;
          } else if (auto selOp = cur.getDefiningOp<SelectOp>()) {
            // UB only has min of operands
            if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
              if (cmp.getLhs() == selOp.getTrueValue() &&
                  cmp.getRhs() == selOp.getFalseValue() &&
                  cmp.getPredicate() == CmpIPredicate::sle) {
                todo.push_back(cmp.getLhs());
                todo.push_back(cmp.getRhs());
                continue;
              }
            }
          }
          return failure();
        }
      }

      bool rewrittenStep = false;
      if (!loop.getStep().getDefiningOp<ConstantIndexOp>()) {
        if (ubs.size() != 1 || lbs.size() != 1)
          return failure();
        ubs[0] = rewriter.create<DivUIOp>(
            loop.getLoc(),
            rewriter.create<SubIOp>(loop.getLoc(), loop.getUpperBound(),
                                    loop.getLowerBound()),
            loop.getStep());
        lbs[0] = rewriter.create<ConstantIndexOp>(loop.getLoc(), 0);
        rewrittenStep = true;
      }

      AffineForOp affineLoop = rewriter.create<AffineForOp>(
          loop.getLoc(), lbs, getMultiSymbolIdentity(builder, lbs.size()), ubs,
          getMultiSymbolIdentity(builder, ubs.size()), getStep(loop.getStep()),
          loop.getIterOperands());

      canonicalizeLoopBounds(rewriter, affineLoop);

      auto mergedYieldOp =
          cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

      Block &newBlock = affineLoop.region().front();

      // The terminator is added if the iterator args are not provided.
      // see the ::build method.
      if (affineLoop.getNumIterOperands() == 0) {
        auto affineYieldOp = newBlock.getTerminator();
        rewriter.eraseOp(affineYieldOp);
      }

      SmallVector<Value> vals;
      rewriter.setInsertionPointToStart(&affineLoop.region().front());
      for (Value arg : affineLoop.region().front().getArguments()) {
        if (rewrittenStep && arg == affineLoop.getInductionVar()) {
          arg = rewriter.create<AddIOp>(
              loop.getLoc(), loop.getLowerBound(),
              rewriter.create<MulIOp>(loop.getLoc(), arg, loop.getStep()));
        }
        vals.push_back(arg);
      }
      assert(vals.size() == loop.getRegion().front().getNumArguments());
      rewriter.mergeBlocks(&loop.getRegion().front(),
                           &affineLoop.region().front(), vals);

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

  void canonicalizeLoopBounds(PatternRewriter &rewriter,
                              AffineParallelOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundsOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundsOperands());

    auto lbMap = forOp.lowerBoundsMap();
    auto ubMap = forOp.upperBoundsMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands);
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

    canonicalizeLoopBounds(rewriter, affineLoop);

    auto mergedYieldOp =
        cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block &newBlock = affineLoop.region().front();

    // The terminator is added if the iterator args are not provided.
    // see the ::build method.
    if (affineLoop.getResults().size() == 0) {
      auto affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    SmallVector<Value> vals;
    for (Value arg : affineLoop.region().front().getArguments()) {
      vals.push_back(arg);
    }
    rewriter.mergeBlocks(&loop.getRegion().front(),
                         &affineLoop.region().front(), vals);

    rewriter.setInsertionPoint(mergedYieldOp);
    rewriter.create<AffineYieldOp>(mergedYieldOp.getLoc(),
                                   mergedYieldOp.getOperands());
    rewriter.eraseOp(mergedYieldOp);

    rewriter.replaceOp(loop, affineLoop.getResults());

    return success();
  }
};

void RaiseSCFToAffine::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<ForOpRaising, ParallelOpRaising>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseSCFToAffinePass() {
  return std::make_unique<RaiseSCFToAffine>();
}
} // namespace polygeist
} // namespace mlir
