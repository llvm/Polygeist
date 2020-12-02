//===- InvariantScopTransform.cc - Invariant transform to OpenScop --------===//
//
// This file implements the transformation between MLIR and OpenScop.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/InvariantScopTransform.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Target/OpenScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

/// Insert value mapping into the given mapping object based on the provided src
/// and dst symbol tables.
static void updateValueMapping(OslSymbolTable &srcTable,
                               OslSymbolTable &dstTable,
                               BlockAndValueMapping &mapping) {
  // TODO: check the symbol compatibility between srcTable and dstTable.
  SmallVector<StringRef, 8> symbols;
  srcTable.getValueSymbols(symbols);

  for (auto sym : symbols)
    mapping.map(srcTable.getValue(sym), dstTable.getValue(sym));
}

static LogicalResult invariantScopTransform(mlir::FuncOp funcOp,
                                            OpBuilder &rewriter) {
  OslSymbolTable srcTable, dstTable;

  auto scop = createOpenScopFromFuncOp(funcOp, srcTable);
  if (!scop) {
    funcOp.emitError(
        "Cannot emit a valid OpenScop representation from the given FuncOp.");
    return failure();
  }

  // TODO: remove this line.
  // scop->print();

  auto moduleOp = dyn_cast<mlir::ModuleOp>(funcOp.getParentOp());

  // TODO: remove the root update pairs.
  createFuncOpFromOpenScop(std::move(scop), moduleOp, dstTable,
                           rewriter.getContext());

  BlockAndValueMapping mapping;
  updateValueMapping(srcTable, dstTable, mapping);

  SmallVector<StringRef, 8> stmtSymbols;
  srcTable.getOpSetSymbols(stmtSymbols);
  for (auto stmtSym : stmtSymbols) {
    // The operation to be cloned.
    auto srcOpSet = srcTable.getOpSet(stmtSym);
    // The clone destination.
    auto dstOpSet = dstTable.getOpSet(stmtSym);
    auto dstOp = dstOpSet.get(0);

    rewriter.setInsertionPoint(dstOp);

    for (unsigned i = 0, e = srcOpSet.size(); i < e; i++)
      rewriter.clone(*(srcOpSet.get(e - i - 1)), mapping);

    // rewriter.setInsertionPoint(dstOp);
    // rewriter.clone(*srcOp, mapping);
    dstOp->erase();
    dstOpSet.get(1)->erase();
  }

  // TODO: remove the callee function/update its function body.

  funcOp.erase();

  return success();
}

namespace {
class InvariantScopTransformPass
    : public mlir::PassWrapper<InvariantScopTransformPass,
                               OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    mlir::OpBuilder b(m.getContext());

    SmallVector<mlir::FuncOp, 8> funcOps;
    m.walk([&](mlir::FuncOp f) {
      if (!f.getAttr("scop.stmt"))
        funcOps.push_back(f);
    });

    for (mlir::FuncOp f : funcOps)
      if (failed(invariantScopTransform(f, b)))
        signalPassFailure();
  }
};

} // namespace

void polymer::registerInvariantScopTransformPass() {
  PassRegistration<InvariantScopTransformPass>(
      "invariant-scop", "Transform MLIR to Scop and back.");
}
