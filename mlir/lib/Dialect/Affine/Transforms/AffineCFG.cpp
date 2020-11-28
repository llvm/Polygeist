#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IntegerSet.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/PatternMatch.h"

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

namespace {
/// Fold alloc operations with no uses. Alloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplfyIntegerCastMath : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<AddIOp>()) {
      rewriter.replaceOpWithNewOp<AddIOp>(
          op,
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0),
                                       op.getType()),
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                       op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SubIOp>()) {
      rewriter.replaceOpWithNewOp<SubIOp>(
          op,
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0),
                                       op.getType()),
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                       op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<MulIOp>()) {
      rewriter.replaceOpWithNewOp<MulIOp>(
          op,
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0),
                                       op.getType()),
          rewriter.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                       op.getType()));
      return success();
    }
    return failure();
  }
};
} // namespace


struct CanonicalizeAffineApply : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {
    
    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;

    fullyComposeAffineMapAndOperands(&map, &mapOperands);
    canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, mapOperands);
    return success();
  }
};

void AffineCFGPass::runOnFunction() {
  getFunction().walk([&](scf::IfOp ifOp) {
    if (inAffine(ifOp)) {
        OpBuilder b(ifOp);
        AffineIfOp affineIfOp;
        std::vector<mlir::Type> types;
        for(auto v : ifOp.results()) {
            types.push_back(v.getType());
        }
        if (auto cmpi = ifOp.condition().getDefiningOp<CmpIOp>()) {
            AffineMap lhsmap = AffineMap::get(0, 1, getAffineSymbolExpr(0, ifOp.getContext()));
            SmallVector<Value,4> lhspack = { cmpi.lhs() };
            if (!lhspack[0].getType().isa<IndexType>()) {
                auto op = b.create<mlir::IndexCastOp>(ifOp.getLoc(), lhspack[0], mlir::IndexType::get(ifOp.getContext()));
                lhspack[0] = op;
            }

            AffineMap rhsmap = AffineMap::get(0, 1, getAffineSymbolExpr(0, ifOp.getContext()));
            SmallVector<Value,4> rhspack = {cmpi.rhs()};
            if (!rhspack[0].getType().isa<IndexType>()) {
                auto op = b.create<mlir::IndexCastOp>(ifOp.getLoc(), rhspack[0], mlir::IndexType::get(ifOp.getContext()));
                rhspack[0] = op;
            }

            Value applies[] = {
                b.create<mlir::AffineApplyOp>(cmpi.getLoc(), lhsmap, lhspack),
                b.create<mlir::AffineApplyOp>(cmpi.getLoc(), rhsmap, rhspack)
            };
            AffineExpr dims[2] = { b.getAffineDimExpr(0), b.getAffineDimExpr(1) };
            AffineExpr exprs[1];
            bool eqflags[1];
            switch(cmpi.getPredicate()) {
                case CmpIPredicate::eq:
                    exprs[0] = dims[0] - dims[1];
                    eqflags[0] = true;
                    break;

                case CmpIPredicate::sge:
                    exprs[0] = dims[0] - dims[1];
                    eqflags[0] = false;
                    break;

                case CmpIPredicate::sle:
                    exprs[0] = dims[1] - dims[0];
                    eqflags[0] = false;
                    break;

                case CmpIPredicate::sgt:
                    exprs[0] = dims[0] - dims[1] + 1;
                    eqflags[0] = false;
                    break;

                case CmpIPredicate::slt:
                    exprs[0] = dims[1] - dims[0] + 1;
                    eqflags[0] = false;
                    break;

                case CmpIPredicate::ne:
                case CmpIPredicate::ult:
                case CmpIPredicate::ule:
                case CmpIPredicate::ugt:
                case CmpIPredicate::uge:
                    llvm_unreachable("unhandled icmp");
            }
            auto iset = IntegerSet::get(/*dim*/2, /*symbol*/0,
                            exprs,
                            eqflags);
            affineIfOp = b.create<AffineIfOp>(ifOp.getLoc(), types, iset,
                                                    applies,
                                                    /*elseBlock=*/true);
        } else {
            return;
        }

        affineIfOp.thenRegion().takeBody(ifOp.thenRegion());
        affineIfOp.elseRegion().takeBody(ifOp.elseRegion());

        for (auto& blk: affineIfOp.thenRegion()) {
            if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
                OpBuilder b(yop);
                b.create<AffineYieldOp>(yop.getLoc(), yop.results());
                yop.erase();
            }
        }
        for (auto& blk: affineIfOp.elseRegion()) {
            if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
                OpBuilder b(yop);
                b.create<AffineYieldOp>(yop.getLoc(), yop.results());
                yop.erase();
            }
        }
        ifOp.replaceAllUsesWith(affineIfOp);
        ifOp.erase();

    }
  });

    getFunction().dump();
    {
        OwningRewritePatternList rpl;
        rpl.insert<SimplfyIntegerCastMath>(getFunction().getContext());
        rpl.insert<CanonicalizeAffineApply>(getFunction().getContext());
        applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(rpl));
    }
    getFunction().dump();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}
