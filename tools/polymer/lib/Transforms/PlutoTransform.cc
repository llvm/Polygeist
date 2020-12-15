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

/// The main function that implements the Pluto based optimization.
static mlir::FuncOp plutoTransform(mlir::FuncOp f, OpBuilder &rewriter) {
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  if (!scop)
    return nullptr;
  if (scop->getNumStatements() == 0)
    return nullptr;

  osl_scop_print(stderr, scop->get());

  // Should use isldep, candl cannot work well for this case.
  context->options->silent = 1;
  context->options->moredebug = 0;
  context->options->isldep = 1;
  context->options->readscop = 1;

  context->options->identity = 0;
// context->options->iss = 0;
#if 0
  context->options->parallel = 0;
  context->options->unrolljam = 0;
  context->options->prevector = 0;
#endif

  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);
  pluto_schedule_prog(prog);
  pluto_populate_scop(scop->get(), prog, context);
  osl_scop_print(stderr, scop->get());

  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(f.getParentOp());
  mlir::FuncOp g = cast<mlir::FuncOp>(createFuncOpFromOpenScop(
      std::move(scop), m, dstTable, rewriter.getContext(), prog));

  pluto_context_free(context);
  return g;
}

namespace {
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
                               OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    mlir::OpBuilder b(m.getContext());

    SmallVector<mlir::FuncOp, 8> funcOps;
    llvm::DenseMap<mlir::FuncOp, mlir::FuncOp> funcMap;

    m.walk([&](mlir::FuncOp f) {
      if (!f.getAttr("scop.stmt"))
        funcOps.push_back(f);
    });

    for (mlir::FuncOp f : funcOps)
      if (mlir::FuncOp g = plutoTransform(f, b)) {
        funcMap[f] = g;
        g.setPrivate();
      }

    // Replacing the original scop top-level function with the pluto transformed
    // result, such that the whole end-to-end optimization is complete.
    m.walk([&](mlir::FuncOp f) {
      for (const auto &it : funcMap) {
        mlir::FuncOp from = it.first;
        mlir::FuncOp to = it.second;
        if (f != from)
          f.walk([&](mlir::CallOp op) {
            if (op.getCallee() == from.getName())
              op.setAttr("callee", b.getSymbolRefAttr(to.getName()));
          });
      }
    });
  }
};

} // namespace

void polymer::registerPlutoTransformPass() {
  PassRegistration<PlutoTransformPass>("pluto-opt",
                                       "Optimization implemented by PLUTO.");
}
