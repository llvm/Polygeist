//===- LoopAnnotate.h - Annotate loop properties -----------------C++-===//

#include "polymer/Transforms/LoopAnnotate.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "loop-annotate"

using namespace mlir;
using namespace mlir::func;
using namespace llvm;
using namespace polymer;

/// A recursive function. Terminates when all operands are not defined by
/// affine.apply, nor loop IVs.
static void annotatePointLoops(ValueRange operands, OpBuilder &b) {
  for (mlir::Value operand : operands) {
    // If a loop IV is directly passed into the statement call.
    if (BlockArgument arg = dyn_cast<BlockArgument>(operand)) {
      mlir::affine::AffineForOp forOp =
          dyn_cast<mlir::affine::AffineForOp>(arg.getOwner()->getParentOp());
      if (forOp) {
        // An affine.for that has its indunction var used by a scop.stmt
        // caller is a point loop.
        forOp->setAttr("scop.point_loop", b.getUnitAttr());
      }
    } else {
      mlir::affine::AffineApplyOp applyOp =
          operand.getDefiningOp<mlir::affine::AffineApplyOp>();
      if (applyOp) {
        // Mark the parents of its operands, if a loop IVs, as point loops.
        annotatePointLoops(applyOp.getOperands(), b);
      }
    }
  }
}

/// Annotate loops in the dst to indicate whether they are point/tile loops.
/// Should only call this after -canonicalize.
/// TODO: Support handling index calculation, e.g., jacobi-1d.
static void annotatePointLoops(FuncOp f, OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();
  assert(m && "A FuncOp should be wrapped in a ModuleOp");

  SmallVector<mlir::func::CallOp> callers;
  f.walk([&](mlir::func::CallOp caller) {
    FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());
    assert(callee && "Callers should have its callees available.");

    // Only gather callers that calls scop.stmt
    if (callee->hasAttr("scop.stmt"))
      callers.push_back(caller);
  });

  for (mlir::func::CallOp caller : callers)
    annotatePointLoops(caller.getOperands(), b);
}

struct AnnotatePointLoopsPass
    : public mlir::PassWrapper<AnnotatePointLoopsPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    annotatePointLoops(f, b);
  }
};

void polymer::registerLoopAnnotatePasses() {
  // PassRegistration<AnnotatePointLoopsPass>();
  // "annotate-point-loops", "Annotate loops with point/tile info.");
}
