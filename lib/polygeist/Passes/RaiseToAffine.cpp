#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-affine"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;

namespace {
struct RaiseSCFToAffine : public SCFRaiseToAffineBase<RaiseSCFToAffine> {
  void runOnOperation() override;
};
} // namespace

/// The upper bound `min(a, b)` of a counted loop (arith.minsi, also under an
/// index_cast) raises to an affine.for with a min bound (VITIS_NOTES N153:
/// `for (u = 0; u < 32 && k + u < B; ++u)`, WhileConjunctiveBound).
static llvm::cl::opt<bool> WhileConjunctiveMinBounds(
    "raise-min-upper-bounds", llvm::cl::init(true),
    llvm::cl::desc("Raise scf.for loops whose upper bound is an arith.minsi "
                   "to affine.for with a min bound (VITIS_NOTES N153)"));

struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // TODO: remove me or rename me.
  bool isAffine(scf::ForOp loop) const {
    // return true;
    // enforce step to be a ConstantIndexOp (maybe too restrictive).
    return affine::isValidSymbol(loop.getStep());
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
          } else if (auto mn = cur.getDefiningOp<MinSIOp>();
                     mn && WhileConjunctiveMinBounds) {
            // UB min(a, b) on index (WhileConjunctiveBound's bound,
            // VITIS_NOTES N153).
            todo.push_back(mn.getLhs());
            todo.push_back(mn.getRhs());
            continue;
          } else if (auto cast = cur.getDefiningOp<IndexCastOp>();
                     cast && WhileConjunctiveMinBounds) {
            // index_cast(minsi(a, b)) = minsi(index_cast(a), index_cast(b)):
            // index_cast sign-extends, which keeps the order.
            if (auto mn = cast.getIn().getDefiningOp<MinSIOp>()) {
              OpBuilder b(loop);
              todo.push_back(b.create<IndexCastOp>(cast.getLoc(),
                                                   b.getIndexType(),
                                                   mn.getLhs()));
              todo.push_back(b.create<IndexCastOp>(cast.getLoc(),
                                                   b.getIndexType(),
                                                   mn.getRhs()));
              continue;
            }
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
        // The trip count is computed above every enclosing loop that does
        // not define the bounds or the step, where it is a valid symbol like
        // its operands: built next to the loop, inside an outer loop, the
        // affine map composition below moves it there itself and loses track
        // of the ops it moved (an assertion on a loop with a variable step
        // `m += 1 << stage` inside a loop over k, hp_fft no_StagePipeline;
        // CIRCT OPEN_ISSUES O76).
        Operation *ip = loop;
        auto definedIn = [](Value v, Operation *region) {
          Operation *def = v.getDefiningOp();
          Region *r = def ? def->getParentRegion()
                          : v.cast<BlockArgument>().getParentRegion();
          return region->isAncestor(r->getParentOp());
        };
        while (Operation *parent = ip->getParentOp()) {
          if (!isa<scf::ForOp, affine::AffineForOp, scf::IfOp, affine::AffineIfOp>(
                  parent) ||
              definedIn(loop.getStep(), parent) ||
              definedIn(loop.getUpperBound(), parent) ||
              definedIn(loop.getLowerBound(), parent))
            break;
          ip = parent;
        }
        PatternRewriter::InsertionGuard tripGuard(rewriter);
        rewriter.setInsertionPoint(ip);
        // Hoisted above the guards and loops that protect the loop, the trip
        // count also runs where the loop never would, so it must not trap:
        // a zero step divides by one, and an empty range (ub <= lb) gives 0
        // instead of a huge unsigned quotient. Where the loop itself runs,
        // the value is the plain (ub - lb + step - 1) / step (a step of 0
        // there is an endless C loop). `if (s > 0) for (i = 0; i < n; i +=
        // s)` inside another loop divided by zero (W9r7 review defect 1).
        Location loc = loop.getLoc();
        Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
        Value one = rewriter.create<ConstantIndexOp>(loc, 1);
        Value range = rewriter.create<SubIOp>(loc, loop.getUpperBound(),
                                              loop.getLowerBound());
        Value nonEmpty =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, range, zero);
        Value roundedUp = rewriter.create<AddIOp>(
            loc, rewriter.create<SubIOp>(loc, loop.getStep(), one), range);
        Value numerator =
            rewriter.create<SelectOp>(loc, nonEmpty, roundedUp, zero);
        Value stepIsZero = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                   loop.getStep(), zero);
        Value divisor =
            rewriter.create<SelectOp>(loc, stepIsZero, one, loop.getStep());
        ubs[0] = rewriter.create<DivUIOp>(loc, numerator, divisor);
        lbs[0] = rewriter.create<ConstantIndexOp>(loop.getLoc(), 0);
        rewrittenStep = true;
      }

      auto *scope = affine::getAffineScope(loop)->getParentOp();
      DominanceInfo DI(scope);

      AffineMap lbMap = getMultiSymbolIdentity(builder, lbs.size());
      {
        fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbs, DI);
        affine::canonicalizeMapAndOperands(&lbMap, &lbs);
        lbMap = removeDuplicateExprs(lbMap);
      }
      AffineMap ubMap = getMultiSymbolIdentity(builder, ubs.size());
      {
        fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubs, DI);
        affine::canonicalizeMapAndOperands(&ubMap, &ubs);
        ubMap = removeDuplicateExprs(ubMap);
      }

      affine::AffineForOp affineLoop = rewriter.create<affine::AffineForOp>(
          loop.getLoc(), lbs, lbMap, ubs, ubMap, getStep(loop.getStep()),
          loop.getInits());
      // The loop mark hls-est reads (Passes.h) moves to the affine loop.
      if (Attribute mark = loop->getAttr(kUncountedLoopAttrName))
        affineLoop->setAttr(kUncountedLoopAttrName, mark);

      auto mergedYieldOp =
          cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

      Block &newBlock = affineLoop.getRegion().front();

      // The terminator is added if the iterator args are not provided.
      // see the ::build method.
      if (affineLoop.getNumIterOperands() == 0) {
        auto *affineYieldOp = newBlock.getTerminator();
        rewriter.eraseOp(affineYieldOp);
      }

      SmallVector<Value> vals;
      rewriter.setInsertionPointToStart(&affineLoop.getRegion().front());
      for (Value arg : affineLoop.getRegion().front().getArguments()) {
        if (rewrittenStep && arg == affineLoop.getInductionVar()) {
          arg = rewriter.create<AddIOp>(
              loop.getLoc(), loop.getLowerBound(),
              rewriter.create<MulIOp>(loop.getLoc(), arg, loop.getStep()));
        }
        vals.push_back(arg);
      }
      assert(vals.size() == loop.getRegion().front().getNumArguments());
      rewriter.mergeBlocks(&loop.getRegion().front(),
                           &affineLoop.getRegion().front(), vals);

      rewriter.setInsertionPoint(mergedYieldOp);
      rewriter.create<affine::AffineYieldOp>(mergedYieldOp.getLoc(),
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

  void canonicalizeLoopBounds(PatternRewriter &rewriter,
                              affine::AffineParallelOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundsOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundsOperands());

    auto lbMap = forOp.getLowerBoundsMap();
    auto ubMap = forOp.getUpperBoundsMap();

    auto *scope = affine::getAffineScope(forOp)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands, DI);
    affine::canonicalizeMapAndOperands(&lbMap, &lbOperands);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands, DI);
    affine::canonicalizeMapAndOperands(&ubMap, &ubOperands);

    forOp.setLowerBounds(lbOperands, lbMap);
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
    affine::AffineParallelOp affineLoop =
        rewriter.create<affine::AffineParallelOp>(
            loop.getLoc(), loop.getResultTypes(), reductions, bounds,
            loop.getLowerBound(), bounds, loop.getUpperBound(),
            steps); //, loop.getInitVals());

    canonicalizeLoopBounds(rewriter, affineLoop);

    auto mergedYieldOp =
        cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block &newBlock = affineLoop.getRegion().front();

    // The terminator is added if the iterator args are not provided.
    // see the ::build method.
    if (affineLoop.getResults().size() == 0) {
      auto *affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    SmallVector<Value> vals;
    for (Value arg : affineLoop.getRegion().front().getArguments()) {
      vals.push_back(arg);
    }
    rewriter.mergeBlocks(&loop.getRegion().front(),
                         &affineLoop.getRegion().front(), vals);

    rewriter.setInsertionPoint(mergedYieldOp);
    rewriter.create<affine::AffineYieldOp>(mergedYieldOp.getLoc(),
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
