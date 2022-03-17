//===- LoopExtraction.cc - Extract loops ------------------------------C++-===//

#include "polymer/Transforms/LoopExtract.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "loop-extract"

using namespace mlir;
using namespace llvm;
using namespace polymer;

/// Check if the provided function has point loops in it.
static bool hasPointLoops(FuncOp f) {
  bool hasPointLoop = false;
  f.walk([&](mlir::AffineForOp forOp) {
    if (!hasPointLoop)
      hasPointLoop = forOp->hasAttr("scop.point_loop");
  });
  return hasPointLoop;
}

static bool isPointLoop(mlir::AffineForOp forOp) {
  return forOp->hasAttr("scop.point_loop");
}

static void getArgs(Operation *parentOp, llvm::SetVector<Value> &args) {
  args.clear();

  llvm::SetVector<Operation *> internalOps;
  internalOps.insert(parentOp);

  parentOp->walk([&](Operation *op) { internalOps.insert(op); });

  parentOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (!internalOps.contains(defOp))
          args.insert(operand);
      } else if (BlockArgument bArg = operand.dyn_cast<BlockArgument>()) {
        if (!internalOps.contains(bArg.getOwner()->getParentOp()))
          args.insert(operand);
      } else {
        llvm_unreachable("Operand cannot be handled.");
      }
    }
  });
}

static FuncOp createCallee(mlir::AffineForOp forOp, int id, FuncOp f,
                           OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Naming convention: <original func name>__PE<id>. <id> is maintained
  // globally.
  std::string calleeName =
      f.getName().str() + std::string("__PE") + std::to_string(id);
  FunctionType calleeType = b.getFunctionType(llvm::None, llvm::None);
  FuncOp callee = b.create<FuncOp>(forOp.getLoc(), calleeName, calleeType);
  callee.setVisibility(SymbolTable::Visibility::Private);

  Block *entry = callee.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::func::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entry);

  llvm::SetVector<Value> args;
  getArgs(forOp, args);

  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType(), arg.getLoc()));
  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));

  b.clone(*forOp.getOperation(), mapping);

  return callee;
}

static int extractPointLoops(FuncOp f, int startId, OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();

  SmallVector<Operation *, 4> callers;
  f.walk([&](mlir::func::CallOp caller) {
    FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());
    if (callee->hasAttr("scop.stmt"))
      callers.push_back(caller);
  });

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  llvm::SetVector<Operation *> extracted;

  for (Operation *caller : callers) {
    SmallVector<mlir::AffineForOp, 4> forOps;
    getLoopIVs(*caller, &forOps);

    int pointBandStart = forOps.size();
    while (pointBandStart > 0 && isPointLoop(forOps[pointBandStart - 1])) {
      pointBandStart--;
    }

    // No point loop band.
    if (static_cast<size_t>(pointBandStart) == forOps.size())
      continue;

    mlir::AffineForOp pointBandStartLoop = forOps[pointBandStart];

    // Already visited.
    if (extracted.contains(pointBandStartLoop))
      continue;
    extracted.insert(pointBandStartLoop);

    createCallee(pointBandStartLoop, startId, f, b);
    startId++;
  }

  return startId;
}

namespace {
struct ExtractPointLoopsPass
    : public mlir::PassWrapper<ExtractPointLoopsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<FuncOp, 4> fs;
    m.walk([&](FuncOp f) {
      if (hasPointLoops(f))
        fs.push_back(f);
    });

    int startId = 0;
    for (FuncOp f : fs)
      startId += extractPointLoops(f, startId, b);
  }
};
} // namespace

void polymer::registerLoopExtractPasses() {
  // PassRegistration<ExtractPointLoopsPass>();
}
