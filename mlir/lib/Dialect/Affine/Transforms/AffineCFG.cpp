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

/*
struct CanonicalizeAffineIf : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp affineOp,
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
*/

bool isValidIndex(Value val) {
    if (mlir::isValidSymbol(val)) {
        return true;
    }
    if (auto cast = val.getDefiningOp<IndexCastOp>()) {
        return isValidIndex(cast.getOperand());
    }
    if (auto bop = val.getDefiningOp<AddIOp>()) {
        return isValidIndex(bop.getOperand(0)) &&
               isValidIndex(bop.getOperand(1));
    }
    if (auto bop = val.getDefiningOp<MulIOp>()) {
        return isValidIndex(bop.getOperand(0)) &&
               isValidIndex(bop.getOperand(1));
    }
    if (auto bop = val.getDefiningOp<SubIOp>()) {
        return isValidIndex(bop.getOperand(0)) &&
               isValidIndex(bop.getOperand(1));
    }
    if (val.getDefiningOp<ConstantIndexOp>()) {
        return true;
    }
    if (val.getDefiningOp<ConstantOp>()) {
        return true;
    }
    if (auto ba = val.dyn_cast<BlockArgument>()) {
        if (isa<AffineForOp>(ba.getOwner()->getParentOp())) {
            return true;
        }
        if (isa<AffineParallelOp>(ba.getOwner()->getParentOp())) {
            return true;
        }
        if (isa<FuncOp>(ba.getOwner()->getParentOp())) {
            return true;
        }
        //llvm::errs() << "illegal isValidIndex: " << val << " pop: " << *ba.getOwner()->getParentOp() << "\n";
    }
    //llvm::errs() << "illegal isValidIndex: " << val << "\n";
    return false;
}

// return if success
bool handle(OpBuilder& b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs, SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
    AffineMap lhsmap = AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
    if (!isValidIndex(cmpi.lhs())) {
        //llvm::errs() << "illegal lhs: " << cmpi.lhs() << " - " << cmpi << "\n";
        return false;
    }
    if (!isValidIndex(cmpi.rhs())) {
        //llvm::errs() << "illegal rhs: " << cmpi.rhs() << " - " << cmpi << "\n";
        return false;
    }
    SmallVector<Value,4> lhspack = { cmpi.lhs() };
    if (!lhspack[0].getType().isa<IndexType>()) {
        auto op = b.create<mlir::IndexCastOp>(cmpi.getLoc(), lhspack[0], mlir::IndexType::get(cmpi.getContext()));
        lhspack[0] = op;
    }

    AffineMap rhsmap = AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
    SmallVector<Value,4> rhspack = {cmpi.rhs()};
    if (!rhspack[0].getType().isa<IndexType>()) {
        auto op = b.create<mlir::IndexCastOp>(cmpi.getLoc(), rhspack[0], mlir::IndexType::get(cmpi.getContext()));
        rhspack[0] = op;
    }

    applies.push_back(b.create<mlir::AffineApplyOp>(cmpi.getLoc(), lhsmap, lhspack));
    applies.push_back(b.create<mlir::AffineApplyOp>(cmpi.getLoc(), rhsmap, rhspack));
    AffineExpr dims[2] = { b.getAffineDimExpr(2 * exprs.size() + 0), b.getAffineDimExpr(2 * exprs.size() + 1) };
    switch(cmpi.getPredicate()) {
        case CmpIPredicate::eq:
            exprs.push_back(dims[0] - dims[1]);
            eqflags.push_back(true);
            break;

        case CmpIPredicate::sge:
            exprs.push_back(dims[0] - dims[1]);
            eqflags.push_back(false);
            break;

        case CmpIPredicate::sle:
            exprs.push_back(dims[1] - dims[0]);
            eqflags.push_back(false);
            break;

        case CmpIPredicate::sgt:
            exprs.push_back(dims[0] - dims[1] + 1);
            eqflags.push_back(false);
            break;

        case CmpIPredicate::slt:
            exprs.push_back(dims[1] - dims[0] + 1);
            eqflags.push_back(false);
            break;

        case CmpIPredicate::ne:
        case CmpIPredicate::ult:
        case CmpIPredicate::ule:
        case CmpIPredicate::ugt:
        case CmpIPredicate::uge:
            llvm_unreachable("unhandled icmp");
    }
    return true;
}

void AffineCFGPass::runOnFunction() {
  getFunction().walk([&](scf::IfOp ifOp) {
    if (inAffine(ifOp)) {
        OpBuilder b(ifOp);
        AffineIfOp affineIfOp;
        std::vector<mlir::Type> types;
        for(auto v : ifOp.results()) {
            types.push_back(v.getType());
        }


        SmallVector<AffineExpr, 2> exprs;
        SmallVector<bool, 2> eqflags;
        SmallVector<Value, 4> applies;

        std::deque<Value> todo = {ifOp.condition()};
        while(todo.size()) {
            auto cur = todo.front();
            todo.pop_front();
            if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
                if (!handle(b, cmpi, exprs, eqflags, applies)) {
                    return;
                }
                continue;
            }
            if (auto andi = cur.getDefiningOp<AndOp>()) {
                todo.push_back(andi.getOperand(0));
                todo.push_back(andi.getOperand(1));
                continue;
            }
            llvm::errs() << "illegal cur: " << cur << " - " << ifOp << "\n";
            return;
        }

        auto iset = IntegerSet::get(/*dim*/2 * exprs.size(), /*symbol*/0,
                        exprs,
                        eqflags);
        affineIfOp = b.create<AffineIfOp>(ifOp.getLoc(), types, iset,
                                                applies,
                                                /*elseBlock=*/true);
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

    //getFunction().dump();
    {
        OwningRewritePatternList rpl;
        rpl.insert<SimplfyIntegerCastMath>(getFunction().getContext());
        rpl.insert<CanonicalizeAffineApply>(getFunction().getContext());
        applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(rpl));
    }
    //getFunction().dump();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}
