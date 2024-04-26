#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
struct RaiseAffineToLinalg : public SCFRaiseToAffineBase<RaiseAffineToLinalg> {
  void runOnOperation() override;
};
} // namespace

// Also want to add support for affine.for ( ) { linalg.generic } -> bigger linalg.generic
// Also probably want to try to do { linalg.generc1(); linalg.generic2(); } -> bigger linalg.generic()

/*

affine.for() {
    affine.for() {
    } 
    affine.for() {
    }
}

*/
struct Condition {
    bool ifTrue;
    AffineIfOp op;
};
struct ForOpRaising : public OpRewritePattern<affine::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const final {

    // Don't handle accumulations in registers for the moment, we can have
    // a separate pattern move them into memref's
    if (loop.getNumReductions() != 0) {
        return failure();
    }

    SmallVector<std::pair<std::vector<Condition>, AffineLoadOp> loads;
    SmallVector<std::pair<std::vector<Condition>, AffineStoreOp> stores;

    // Check that the only operations within the region are either:
    //      affine.load, affine.store, affine.if, affine.yield
    // Additionally, for each load/store, remember what conditions are
    // required for that load or store to execute.
    Walkloop result = loop->walk<WalkOrder::PreOrder>([&](Operation* op) {
        // TODO extend this, any non-memory operation is also legal here.
        // mul, add, etc (we can just check propety)
        if (isa<AffineYieldOp, AffineIfOp>(op)) {
            return WalkResult::advance();
        }
        if (isa<AffineLoadOp, AffineStoreOp>(op)) {
            Operation *cur = op->getParentOp();
            std::vector<Condition> conditions;
            while (cur != loop) {
                auto ifstmt = dyn_cast<AffineIfOp>(cur);
                if (!ifstmt) {
                    return WalkResult::interrupt();
                }
                bool ifTrue = ifstmt.getTrueRegion()->isAncestor(cur);
                conditions.push_back(ifTrue, ifstmt);
                cur = ifstmt->getParent();
            }
            if (auto load = dyn_cast<AffineLoadOp>(cur)) {
                loads.emplace_back(condition, load);
            } else {
                auto store = cast<AffineStoreOp>(cur);
                stores.emplace_back(condition, store);
            }
            return WalkResult::advance();
        }
        return WalkResult::interrupt();
    });
    
    if (result.wasInterrupted()) return failure();

    // Check that all of the stores do not alias the loaded values (otherwise we could get an incorrect result)
    // TODO we can extend this and handle things like reductions, but we're going to start easy for now
    for (auto &&[_, store] : stores) {
        for (auto &&[_, load]: loads) {
            if (mayAlias(load.getMemref(), stores.getMemref())) {
                return failure();
            }
        }
        for (auto &&[_, store2]: stores) {
            if (store == store2) continue;
            if (mayAlias(store.getMemref(), stores2.getMemref())) {
                return failure();
            }
        }
    }



    SmallVector<Value> inputs;
    SmallVector<AffineMap> affineMaps;
    for (auto [conds, load]&& : loads) {
        // Only support unconditional loads for the moment
        if (conds.size() != 0) return failure();
        inputs.push_back(load.getMemref());
        affineMaps.push_back(load.getAffineMap());
    }
    
    SmallVector<Value> outputs;
    for (auto [conds, store]&& : stores) {
        // Only support unconditional loads for the moment
        if (conds.size() != 0) return failure();
        outputs.push_back(store.getMemref());
        affineMaps.push_back(store.getAffineMap());
    }

    ArrayRef<utils::IteratorType> iteratorTypes;
    // TODO fill this with the for loop bounds

    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, TypeRange(), inputs, outputs, affineMaps, iteratorTypes,
      StringAttr(),
      StringAttr());


    auto &body = genericOp.getRegion();
    body.takeBody(loop.getRegion());

    rewriter.setInsertionPointToStart(*body.begin());

    // This index will replace the use of the affine index
    auto idx = rewriter.create<linalg::IndexOp>(rewriter.getIndexAttr(0));
    rewriter.replaceAllUsesWith(loop.getInductionVariable(), idx);

    // TODO
    // replace all loads with the corresponding block arguments we will create in the linalg generic
    //


    SmallVector<Value> toreturn;
    // push all sotred values into a linalg.yield

    // fixup the return



    // return success!


    return failure();
  }
};

void RaiseAffineToLinalg::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<ForOpRaising>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalg>();
}
} // namespace polygeist
} // namespace mlir
