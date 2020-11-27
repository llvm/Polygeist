//===- PlutoTransform.cc - Transform MLIR code by PLUTO -------------------===//
//
// This file implements the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

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
static LogicalResult updateValueMapping(OslSymbolTable &srcTable,
                                        OslSymbolTable &dstTable,
                                        BlockAndValueMapping &mapping) {
  // TODO: check the symbol compatibility between srcTable and dstTable.
  SmallVector<StringRef, 8> symbols;
  srcTable.getValueSymbols(symbols);

  for (auto sym : symbols) {
    if (auto dstVal = dstTable.getValue(sym))
      mapping.map(srcTable.getValue(sym), dstVal);
    else {
      llvm::errs()
          << "Symbol " << sym
          << " in the source table is not found in the destination table.\n";
      return failure();
    }
  }
  return success();
}

static LogicalResult plutoTransform(mlir::FuncOp f, OpBuilder &rewriter) {
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  if (!scop)
    return success();
  if (scop->getNumStatements() == 0)
    return success();

  // for (const auto &it : *(scop->getScopStmtMap())) {
  //   llvm::outs() << it.first() << "\n";
  // }

  // Should use isldep, candl cannot work well for this case.
  // TODO: should discover why.
  context->options->isldep = 1;

  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);

  pluto_compute_dep_directions(prog);
  pluto_compute_dep_satisfaction(prog);
  pluto_tile(prog);

  pluto_populate_scop(scop->get(), prog, context);
  osl_scop_print(stdout, scop->get());

  auto moduleOp = dyn_cast<mlir::ModuleOp>(f.getParentOp());

  // TODO: remove the root update pairs.
  auto newFuncOp = createFuncOpFromOpenScop(std::move(scop), moduleOp, dstTable,
                                            rewriter.getContext());

#if 0
  BlockAndValueMapping mapping;
  // TODO: refactorize this function and the following logic.
  if (failed(updateValueMapping(srcTable, dstTable, mapping)))
    return failure();

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
    rewriter.eraseOp(dstOp);
    rewriter.eraseOp(dstOpSet.get(1));
  }

  rewriter.eraseOp(f);
#endif

  pluto_context_free(context);
  return success();
}

namespace {
/// TODO: split this into specific categories like tiling.
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
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
      if (failed(plutoTransform(f, b)))
        signalPassFailure();
  }
};
} // namespace

void polymer::registerPlutoTransformPass() {
  PassRegistration<PlutoTransformPass>("pluto-opt",
                                       "Optimization implemented by PLUTO.");
}
