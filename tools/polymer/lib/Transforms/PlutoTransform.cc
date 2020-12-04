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

/// The main function that implements the Pluto based optimization.
static LogicalResult plutoTransform(mlir::FuncOp f, OpBuilder &rewriter,
                                    bool useParallel) {
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  if (!scop)
    return success();
  if (scop->getNumStatements() == 0)
    return success();

  // Should use isldep, candl cannot work well for this case.
  // TODO: should discover why.
  context->options->isldep = 1;
  context->options->isldepaccesswise = 1;
  context->options->silent = 1;
  context->options->intratileopt = 1;
  context->options->diamondtile = 1;
  context->options->rar = 1;
  context->options->iss = 0;
  context->options->identity = 0;
  context->options->tile = 1;
  context->options->parallel = 1;

  if (useParallel) {
    context->options->tile = 0;
  }

  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);
  if (!context->options->silent) {
    fprintf(stderr, "[pluto] Number of statements: %d\n", prog->nstmts);
    // fprintf(stdout, "[pluto] Total number of loops: %d\n", dim_sum);
    fprintf(stderr, "[pluto] Number of deps: %d\n", prog->ndeps);
    fprintf(stderr, "[pluto] Maximum domain dimensionality: %d\n", prog->nvar);
    fprintf(stderr, "[pluto] Number of parameters: %d\n", prog->npar);
  }

  if (context->options->iss)
    pluto_iss_dep(prog);
  if (!context->options->identity)
    pluto_auto_transform(prog);
  pluto_compute_dep_directions(prog);
  pluto_compute_dep_satisfaction(prog);

  if (context->options->tile) {
    pluto_tile(prog);
  } else {
    if (context->options->intratileopt) {
      pluto_intra_tile_optimize(prog, 0);
    }
  }

  if (context->options->parallel && !context->options->tile &&
      !context->options->identity) {
    /* Obtain wavefront/pipelined parallelization by skewing if
     * necessary */
    unsigned nbands;
    Band **bands;
    pluto_compute_dep_satisfaction(prog);
    bands = pluto_get_outermost_permutable_bands(prog, &nbands);
    bool retval = pluto_create_tile_schedule(prog, bands, nbands);
    pluto_bands_free(bands, nbands);

    /* If the user hasn't supplied --tile and there is only pipelined
     * parallelism, we will warn the user */
    if (retval) {
      llvm::errs() << "[pluto] WARNING: pipelined parallelism exists and "
                      "--tile is not used.\n";
      // printf("\tUse --tile for better parallelization \n");
      //   fprintf(stdout, "[pluto] After skewing:\n");
      //   pluto_transformations_pretty_print(prog);
    }
  }

  pluto_populate_scop(scop->get(), prog, context);
  osl_scop_print(stderr, scop->get());

  auto m = dyn_cast<mlir::ModuleOp>(f.getParentOp());
  createFuncOpFromOpenScop(std::move(scop), m, dstTable, rewriter.getContext());

  pluto_context_free(context);
  return success();
}

namespace {
/// TODO: split this into specific categories like tiling.
class PlutoTileTransformPass
    : public mlir::PassWrapper<PlutoTileTransformPass,
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
      if (failed(plutoTransform(f, b, false)))
        signalPassFailure();
  }
};

/// TODO: split this into specific categories like tiling.
class PlutoParallelTransformPass
    : public mlir::PassWrapper<PlutoParallelTransformPass,
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
      if (failed(plutoTransform(f, b, true)))
        signalPassFailure();
  }
};
} // namespace

void polymer::registerPlutoTransformPass() {
  PassRegistration<PlutoTileTransformPass>(
      "pluto-opt", "Optimization implemented by PLUTO.");
  PassRegistration<PlutoParallelTransformPass>(
      "pluto-par", "Parallel optimization implemented by PLUTO.");
}
