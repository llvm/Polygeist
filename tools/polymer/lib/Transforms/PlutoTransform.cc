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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {
struct PlutoOptPipelineOptions
    : public mlir::PassPipelineOptions<PlutoOptPipelineOptions> {
  Option<std::string> dumpClastAfterPluto{
      *this, "dump-clast-after-pluto",
      llvm::cl::desc("File name for dumping the CLooG AST (clast) after Pluto "
                     "optimization.")};
  Option<bool> parallelize{
      *this, "parallelize",
      llvm::cl::desc("Enable parallelization from Pluto.")};
};

} // namespace

/// The main function that implements the Pluto based optimization.
/// TODO: transform options?
static mlir::FuncOp plutoTransform(mlir::FuncOp f, OpBuilder &rewriter,
                                   std::string dumpClastAfterPluto,
                                   bool parallelize) {

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
  context->options->parallel = parallelize;
  context->options->unrolljam = 0;
  context->options->prevector = 0;

  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);
  pluto_schedule_prog(prog);
  pluto_populate_scop(scop->get(), prog, context);
  osl_scop_print(stderr, scop->get());

  const char *dumpClastAfterPlutoStr = nullptr;
  if (!dumpClastAfterPluto.empty())
    dumpClastAfterPlutoStr = dumpClastAfterPluto.c_str();

  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(f.getParentOp());
  mlir::FuncOp g = cast<mlir::FuncOp>(createFuncOpFromOpenScop(
      std::move(scop), m, dstTable, rewriter.getContext(), prog,
      dumpClastAfterPlutoStr));

  pluto_context_free(context);
  return g;
}

namespace {
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
                               OperationPass<mlir::ModuleOp>> {
  std::string dumpClastAfterPluto = "";
  bool parallelize = false;

public:
  PlutoTransformPass() = default;
  PlutoTransformPass(const PlutoTransformPass &pass) {}
  PlutoTransformPass(const PlutoOptPipelineOptions &options)
      : dumpClastAfterPluto(options.dumpClastAfterPluto),
        parallelize(options.parallelize) {}

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
      if (mlir::FuncOp g =
              plutoTransform(f, b, dumpClastAfterPluto, parallelize)) {
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

// -------------------------- PlutoParallelizePass ----------------------------

/// Find a single affine.for with scop.parallelizable attr.
static mlir::AffineForOp findParallelizableLoop(mlir::FuncOp f) {
  mlir::AffineForOp ret = nullptr;
  f.walk([&ret](mlir::AffineForOp forOp) {
    if (!ret && forOp->hasAttr("scop.parallelizable"))
      ret = forOp;
  });
  return ret;
}

/// Turns a single affine.for with scop.parallelizable into affine.parallel. The
/// design of this function is almost the same as affineParallelize. The
/// differences are:
///
/// 1. It is not necessary to check whether the parentOp of a parallelizable
/// affine.for has the AffineScop trait.
static void plutoParallelize(mlir::AffineForOp forOp, OpBuilder b) {
  assert(forOp->hasAttr("scop.parallelizable"));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  Location loc = forOp.getLoc();

  // If a loop has a 'max' in the lower bound, emit it outside the parallel loop
  // as it does not have implicit 'max' behavior.
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  bool needsMax = lowerBoundMap.getNumResults() > 1;
  bool needsMin = upperBoundMap.getNumResults() > 1;
  AffineMap identityMap;
  if (needsMax || needsMin)
    identityMap = AffineMap::getMultiDimIdentityMap(1, loc->getContext());
  if (needsMax) {
    auto maxOp = b.create<AffineMaxOp>(loc, lowerBoundMap, lowerBoundOperands);
    lowerBoundMap = identityMap;
    lowerBoundOperands = maxOp->getResults();
  }

  // Same for the upper bound.
  if (needsMin) {
    auto minOp = b.create<AffineMinOp>(loc, upperBoundMap, upperBoundOperands);
    upperBoundMap = identityMap;
    upperBoundOperands = minOp->getResults();
  }

  // Creating empty 1-D affine.parallel op.
  AffineParallelOp newPloop = b.create<AffineParallelOp>(
      loc, llvm::None, llvm::None, lowerBoundMap, lowerBoundOperands,
      upperBoundMap, upperBoundOperands);
  // Steal the body of the old affine for op and erase it.
  newPloop.region().takeBody(forOp.region());

  for (auto user : forOp->getUsers()) {
    user->dump();
  }
  forOp.erase();
}

/// Need to check whether the bounds of the for loop are using top-level values
/// as operands. If not, then the loop cannot be directly turned into
/// affine.parallel.
static bool isBoundParallelizable(mlir::AffineForOp forOp, bool isUpper) {
  llvm::SmallVector<mlir::Value, 4> mapOperands =
      isUpper ? forOp.getUpperBoundOperands() : forOp.getLowerBoundOperands();

  for (mlir::Value operand : mapOperands)
    if (!isTopLevelValue(operand))
      return false;
  return true;
}
static bool isBoundParallelizable(mlir::AffineForOp forOp) {
  return isBoundParallelizable(forOp, true) &&
         isBoundParallelizable(forOp, false);
}

/// Iteratively replace affine.for with scop.parallelizable with
/// affine.parallel.
static void plutoParallelize(mlir::FuncOp f, OpBuilder b) {
  mlir::AffineForOp forOp = nullptr;
  while ((forOp = findParallelizableLoop(f)) != nullptr) {
    if (!isBoundParallelizable(forOp))
      llvm_unreachable(
          "Loops marked as parallelizable should have parallelizable bounds.");
    plutoParallelize(forOp, b);
  }
}

namespace {
/// Turn affine.for marked as scop.parallelizable by Pluto into actual
/// affine.parallel operation.
struct PlutoParallelizePass
    : public mlir::PassWrapper<PlutoParallelizePass,
                               OperationPass<mlir::FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    plutoParallelize(f, b);
  }
};
} // namespace

void polymer::registerPlutoTransformPass() {
  PassPipelineRegistration<PlutoOptPipelineOptions>(
      "pluto-opt", "Optimization implemented by PLUTO.",
      [](OpPassManager &pm, const PlutoOptPipelineOptions &pipelineOptions) {
        pm.addPass(std::make_unique<PlutoTransformPass>(pipelineOptions));
        pm.addPass(createCanonicalizerPass());
        if (pipelineOptions.parallelize) {
          pm.addPass(std::make_unique<PlutoParallelizePass>());
          pm.addPass(createCanonicalizerPass());
        }
      });
}
