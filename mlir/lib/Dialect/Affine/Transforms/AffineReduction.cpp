#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
struct AffineReductionPass : public AffineReductionBase<AffineReductionPass> {
  void runOnFunction() override;
};
} // end namespace.

namespace {

struct AffineForReductionIter : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  bool isInCurrentAffineFor(Operation *op, AffineForOp forOp) const {
    auto *parentOp = op->getParentOp();
    auto maybeParentFor = dyn_cast_or_null<AffineForOp>(parentOp);
    if (maybeParentFor && maybeParentFor == forOp)
      return true;
    return false;
  }

  bool areInSameAffineFor(AffineLoadOp load, AffineStoreOp store,
                          AffineForOp forOp) const {
    return isInCurrentAffineFor(load.getOperation(), forOp) &&
           isInCurrentAffineFor(store.getOperation(), forOp);
  }

  template <typename T>
  bool haveSameIndices(AffineLoadOp load, T storeOrLoad) const {
    static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
                  "applies to only AffineLoadOp or AffineStoreOp");
    SmallVector<Value, 4> loadIndices(load.indices());
    SmallVector<Value, 4> storeOrLoadIndices = storeOrLoad.indices();
    if (loadIndices.size() != storeOrLoadIndices.size())
      return false;
    return std::equal(loadIndices.begin(), loadIndices.end(),
                      storeOrLoadIndices.begin());
  }

  template <typename T>
  bool areCompatible(AffineLoadOp load, T store) const {
    static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
                  "applies to only AffineLoadOp or AffineStoreOp");
    if (load.getMemRef() != store.getMemRef()) {
      return false;
    }
    return haveSameIndices<T>(load, store);
  }

  bool checkDominance(AffineLoadOp load, AffineStoreOp store) const {
    Operation *loadOp = load.getOperation();
    Operation *storeOp = store.getOperation();
    DominanceInfo dom(loadOp);
    return dom.properlyDominates(loadOp, storeOp);
  }

  bool hasAllDimsReduced(ArrayRef<Value> indices, Value indVar) const {
    if (llvm::all_of(indices,
                     [indVar](Value index) { return index != indVar; }))
      return true;
    return false;
  }

  LogicalResult matchAndRewrite(AffineForOp forOp,
                                PatternRewriter &rewriter) const override {

    Block *block = forOp.getBody();
    SmallVector<std::pair<Operation *, Operation *>, 0> candidateOpsInFor;
    SmallVector<Operation *> downStreamLoads;
    block->walk([&](Operation *operation) {
      if (auto load = dyn_cast<AffineLoadOp>(operation)) {
        Value memref = load.getMemRef();
        bool foundDownStreamLoad = false;
        bool foundPairOfLoadStore = false;

        // among the users of the load memref locate
        // a possible compatible store.
        for (auto user : memref.getUsers()) {
          if (auto store = dyn_cast<AffineStoreOp>(user)) {
            SmallVector<Value, 4> indices(load.indices());
            // check for a store in the current for.
            if (!foundPairOfLoadStore &&
                // load and store must be in the same for.
                areInSameAffineFor(load, store, forOp) &&
                // must have same memref and indices.
                areCompatible<AffineStoreOp>(load, store) &&
                // load must domainte the store.
                checkDominance(load, store) &&
                // all the indices need to reduce in the current for.
                hasAllDimsReduced(indices, forOp.getInductionVar())) {
              foundPairOfLoadStore = true;
              candidateOpsInFor.push_back(
                  std::make_pair(load.getOperation(), store.getOperation()));
            }
          }
        }

        // iterate again to check for any compatible down stream load.
        for (auto user : memref.getUsers()) {
          if (auto maybeDownStreamLoad = dyn_cast<AffineLoadOp>(user)) {
            PostDominanceInfo postDom(forOp);
            // check for a load after the current for.
            if (!foundDownStreamLoad && foundPairOfLoadStore &&
                // load must post dominate the current for.
                postDom.properlyPostDominates(
                    maybeDownStreamLoad.getOperation(), forOp.getOperation()) &&
                // loads need to be compatible.
                areCompatible<AffineLoadOp>(load, maybeDownStreamLoad)) {
              foundDownStreamLoad = true;
              downStreamLoads.push_back(maybeDownStreamLoad.getOperation());
            }
          }
        }
      }
    });

    // no work to do.
    if (!candidateOpsInFor.size())
      return failure();

    llvm::errs() << "------------\n";
    llvm::errs() << "#downStreamloads: " << downStreamLoads.size() << "\n";
    llvm::errs() << "#candidateOpsInFor: " << candidateOpsInFor.size() << "\n";

    llvm::errs() << "candidateOpsInFor\n";
    for (auto pair : candidateOpsInFor) {
      std::get<0>(pair)->dump();
      std::get<1>(pair)->dump();
    }
    llvm::errs() << "downStreamLoads\n";
    for (auto l : downStreamLoads)
      l->dump();
    llvm::errs() << "-for-\n";
    forOp.dump();
    llvm::errs() << "------------\n";

    // move the load outside the loop. All the load indexes are
    // not used in the current for (see hasAllDimReduced).
    // The load result are passed to the new forOp as iter args.
    SmallVector<Value, 4> newIterArgs;
    llvm::append_range(newIterArgs, forOp.getRegionIterArgs());
    rewriter.setInsertionPoint(forOp);
    for (auto pair : candidateOpsInFor) {
      auto movedLoad = rewriter.clone(*std::get<0>(pair));
      newIterArgs.push_back(movedLoad->getResult(0));
    }

    // create the for.
    AffineForOp newForOp = rewriter.create<AffineForOp>(
        forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep(), newIterArgs);

    // remove load operation inside the for.
    size_t i = 0;
    for (auto pair : candidateOpsInFor) {
      std::get<0>(pair)->getResult(0).replaceAllUsesWith(
          newForOp.getBody()
              ->getArguments()[i + forOp.getNumRegionIterArgs() + 1]);
      rewriter.eraseOp(std::get<0>(pair));
      ++i;
    }

    Block *newBlock = newForOp.getBody();
    Block *oldBlock = forOp.getBody();
    SmallVector<Value, 4> newBlockTransferArgs;
    newBlockTransferArgs.push_back(newForOp.getInductionVar());
    for (size_t i = 0; i < forOp.getNumRegionIterArgs(); i++)
      newBlockTransferArgs.push_back(newForOp.getRegionIterArgs()[i]);
    assert(oldBlock->getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");
    rewriter.mergeBlocks(oldBlock, newBlock, newBlockTransferArgs);

    auto cloneFilteredTerminator = [&](AffineYieldOp mergedTerminator) {
      SmallVector<Value, 4> newOperands;
      llvm::append_range(newOperands, mergedTerminator.getOperands());
      // store operands are now returned.
      for (auto pair : candidateOpsInFor) {
        newOperands.push_back(std::get<1>(pair)->getOperand(0));
        // rewriter.eraseOp(std::get<1>(pair));
      }
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      rewriter.create<AffineYieldOp>(mergedTerminator.getLoc(), newOperands);
    };

    auto mergedYieldOp = cast<AffineYieldOp>(newBlock->getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);

    // prepare for new yielded value for 'replaceOp'.
    SmallVector<Value, 4> newYieldedRes;
    SmallVector<Value, 4> newRes(newForOp.getResults());
    int additionalRes =
        newForOp.getResults().size() - forOp.getResults().size();
    assert(additionalRes >= 0 && "must be >= 0");
    newRes.insert(newRes.end(), newRes.begin(), newRes.end() - additionalRes);

    // propagate results new forOp to downstream loads if any,
    // otherwise insert a store right after the for. The stored
    // element is the result of the for.
    i = 0;
    bool hasDownstreamLoads = downStreamLoads.size() != 0;
    // make sure to have a 1:1 downstream loads with a candidate load/store in
    // for.
    if (hasDownstreamLoads &&
        (downStreamLoads.size() != candidateOpsInFor.size()))
      return failure();

    for (auto pair : candidateOpsInFor) {
      if (hasDownstreamLoads) {
        assert(downStreamLoads.size() == candidateOpsInFor.size());
        downStreamLoads[i]->getResult(0).replaceAllUsesWith(
            newForOp.getResults()[forOp.getResults().size() + i]);
        rewriter.eraseOp(downStreamLoads[i]);
      } else {
        auto store = cast<AffineStoreOp>(std::get<1>(pair));
        rewriter.setInsertionPointAfter(newForOp);
        auto rank = store.getMemRef().getType().cast<MemRefType>().getRank();
        // constant. Why Mem2Reg does not remove them?
        // if (rank == 1 && store.indices().size() == 0) {
        //  ConstantIndexOp cst =
        //  rewriter.create<ConstantIndexOp>(store.getLoc(), 0);
        //  rewriter.create<AffineStoreOp>(
        //    newForOp.getLoc(),
        //    newForOp.getResults()[forOp.getResults().size() + i],
        //    store.getMemRef(), cst.getResult());
        //} else {
        rewriter.create<AffineStoreOp>(
            newForOp.getLoc(),
            newForOp.getResults()[forOp.getResults().size() + i],
            store.getMemRef(), store.getAffineMap(), store.indices());
        //}
      }
      rewriter.eraseOp(std::get<1>(pair));
      ++i;
    }

    rewriter.replaceOp(forOp, newYieldedRes);
    return success();
  }
};

} // end namespace.

void AffineReductionPass::runOnFunction() {
  // getFunction().dump();
  {
    mlir::RewritePatternSet rpl(getFunction().getContext());
    rpl.add<AffineForReductionIter>(getFunction().getContext());
    applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(rpl),
                                 /*fold*/ false);
  }
  // getFunction().dump();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::detectReductionPass() {
  return std::make_unique<AffineReductionPass>();
}
