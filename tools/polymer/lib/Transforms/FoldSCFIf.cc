//===- FoldSCFIf.cc - Fold scf.if into select --------------C++-===//

#include "polymer/Transforms/FoldSCFIf.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "fold-scf-if"

static void foldSCFIf(scf::IfOp ifOp, FuncOp f, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  LLVM_DEBUG(dbgs() << "Working on ifOp: " << ifOp << "\n\n");

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Value> thenResults, elseResults;

  auto cloneAfter = [&](Block *block, SmallVectorImpl<Value> &results) {
    BlockAndValueMapping vmap;
    for (Operation &op : block->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        for (Value result : yieldOp.getOperands())
          results.push_back(vmap.contains(result) ? vmap.lookup(result)
                                                  : result);
      else
        b.clone(op, vmap);
    }
  };

  cloneAfter(ifOp.thenBlock(), thenResults);

  // Only an if op can have results when an else block is present.
  if (ifOp.elseBlock()) {
    cloneAfter(ifOp.elseBlock(), elseResults);

    for (auto ifResult : enumerate(ifOp.getResults())) {
      Value newResult = b.create<SelectOp>(loc, ifOp.condition(),
                                           thenResults[ifResult.index()],
                                           elseResults[ifResult.index()]);
      ifResult.value().replaceAllUsesWith(newResult);
    }
  }

  ifOp.erase();
}

/// Return true if anything changed.
static bool process(FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](scf::IfOp ifOp) {
    /// TODO: add verification.
    foldSCFIf(ifOp, f, b);
    changed = true;
  });

  return changed;
}

namespace {
struct FoldSCFIfPass : PassWrapper<FoldSCFIfPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    while (process(f, b))
      ;
  }
};
} // namespace

void polymer::registerFoldSCFIfPass() {
  PassPipelineRegistration<>(
      "fold-scf-if", "Fold scf.if into select.",
      [](OpPassManager &pm) { pm.addPass(std::make_unique<FoldSCFIfPass>()); });
}
