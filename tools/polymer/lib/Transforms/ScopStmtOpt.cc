//===- ScopStmtOpt.cc - Optimise SCoP statement extraction ------------C++-===//

#include "polymer/Transforms/ScopStmtOpt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"

#include <queue>
#include <utility>

#define DEBUG_TYPE "scop-stmt-opt"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {
struct OpSplitByScratchpadPass
    : public mlir::PassWrapper<OpSplitByScratchpadPass, OperationPass<FuncOp>> {
  void runOnOperation() override {}
};
} // namespace

static int annotateSplittable(FuncOp f, OpBuilder &b, int startId) {
  int id = startId;

  f.walk([&](mlir::AffineStoreOp storeOp) {
    // Breadth first search to find the splittable operations.

    // Stores operation and depth pairs.
    std::queue<std::pair<Operation *, int>> worklist;
    worklist.push(std::make_pair(storeOp, 0));

    while (!worklist.empty()) {
      std::pair<Operation *, int> front = worklist.front();
      worklist.pop();

      Operation *op = front.first;
      int depth = front.second;

      // Annotation.
      if (depth > 1) {
        op->setAttr("scop.splittable", b.getIndexAttr(id));
        id++;
      }

      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        // Filter out block arguments.
        if (defOp == nullptr || operand.isa<BlockArgument>())
          continue;
        // Filter out operations out of the current region.
        if (defOp->getParentRegion() != storeOp.getParentRegion())
          continue;
        // Filter out defining operations of specific types.
        if (isa<mlir::AffineReadOpInterface, mlir::ConstantOp>(defOp))
          continue;

        worklist.push(std::make_pair(defOp, depth + 1));
      }
    }
  });

  return id - startId;
}

namespace {

struct AnnotateSplittablePass
    : public mlir::PassWrapper<AnnotateSplittablePass,
                               OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    int numSplittable = 0;
    m.walk([&](FuncOp f) {
      numSplittable += annotateSplittable(f, b, numSplittable);
    });
  }
};

} // namespace

void polymer::registerScopStmtOptPasses() {
  PassRegistration<AnnotateSplittablePass>(
      "annotate-splittable",
      "Give operations that are splittable in its expression tree.");
}
