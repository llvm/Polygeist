//===- SelectFunc.cpp - Filter and output only selected functions
//----------===//
//
// This file implements a pass to filter functions by name, removing all
// functions that don't match the specified names.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "select-func"

using namespace mlir;
using namespace polygeist;

namespace {

struct SelectFuncPass
    : public PassWrapper<SelectFuncPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectFuncPass)

  StringRef getArgument() const final { return "select-func"; }

  StringRef getDescription() const final {
    return "Filter functions by name, keeping only those specified";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    if (!pipeline.empty()) {
      OpPassManager pm(ModuleOp::getOperationName(),
                       OpPassManager::Nesting::Implicit);
      (void)parsePassPipeline(pipeline, pm, llvm::errs());
      pm.getDependentDialects(registry);
    }
  }

  SelectFuncPass() = default;
  SelectFuncPass(const SelectFuncPass &) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "SelectFunc: Filtering functions\n");

    // If no function names specified, keep all functions
    if (funcNames.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No function names specified, keeping all\n");

      // If pipeline is specified, run it on the entire module
      if (!pipeline.empty()) {
        OpPassManager pm(module.getOperationName(),
                         OpPassManager::Nesting::Implicit);
        if (failed(parsePassPipeline(pipeline, pm, llvm::errs()))) {
          signalPassFailure();
          return;
        }
        if (failed(runPipeline(pm, module))) {
          signalPassFailure();
        }
      }
      return;
    }

    // preserve-module mode applies the nested pipeline to the requested roots
    // while retaining every other top-level operation byte-for-byte.  Keep a
    // detached clone of those operations while the existing dependency-aware
    // filter provides a small, verifiable module to the nested pipeline.
    Block preservedOperations;
    if (preserveModule) {
      for (Operation &op : module.getBody()->getOperations()) {
        auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
        if (!symbolOp || !llvm::is_contained(funcNames, symbolOp.getName()))
          preservedOperations.push_back(op.clone());
      }
    }

    // Keep the requested roots and the transitive symbol dependencies they
    // reference. Previously this pass erased declarations such as `@logf`
    // while leaving calls in the selected function, producing invalid IR.
    llvm::SmallPtrSet<Operation *, 16> keep;
    llvm::SmallPtrSet<Operation *, 16> roots;
    SmallVector<Operation *> worklist;
    for (Operation &op : module.getBody()->getOperations()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
      if (symbolOp && llvm::is_contained(funcNames, symbolOp.getName()) &&
          keep.insert(&op).second) {
        roots.insert(&op);
        worklist.push_back(&op);
      }
    }
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      auto uses = SymbolTable::getSymbolUses(op);
      if (!uses)
        continue;
      for (const SymbolTable::SymbolUse &use : *uses) {
        Operation *dependency =
            SymbolTable::lookupNearestSymbolFrom(op, use.getSymbolRef());
        if (dependency && keep.insert(dependency).second)
          worklist.push_back(dependency);
      }
    }

    // Collect top-level symbols to remove.
    SmallVector<Operation *> toRemove;
    for (Operation &op : module.getBody()->getOperations()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
      if (!symbolOp)
        continue;
      if (!keep.contains(&op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Marking for removal: " << symbolOp.getName() << "\n");
        toRemove.push_back(&op);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Keeping: " << symbolOp.getName() << "\n");
      }
    }

    // Remove functions not in the filter list
    for (Operation *op : toRemove) {
      op->erase();
    }

    // A separately compiled harness can provide the transitive C helpers used
    // by the selected function.  In that mode retain their declarations (and
    // therefore the calls' exact ABI) without compiling unrelated bodies.
    // This is especially important when one of those bodies contains IR that
    // is outside the lowering scope of the selected function.
    if (externalizeDependencies || preserveModule) {
      for (Operation *op : keep) {
        if (roots.contains(op))
          continue;
        if (auto func = dyn_cast<func::FuncOp>(op); func && !func.isDeclaration()) {
          func.eraseBody();
          func.setPrivate();
        }
      }
    }

    // If pipeline is specified, run it on the filtered module
    if (!pipeline.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Running pipeline on filtered functions\n");

      OpPassManager pm(module.getOperationName(),
                       OpPassManager::Nesting::Implicit);

      if (failed(parsePassPipeline(pipeline, pm, llvm::errs()))) {
        signalPassFailure();
        return;
      }

      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }
    }

    if (preserveModule) {
      // Discard dependency declarations used only to verify the isolated
      // transformation, then restore their original definitions together
      // with every untouched function/global.  The selected function remains
      // at its original symbol and all existing callers continue to target it.
      for (Operation &op : llvm::make_early_inc_range(
               module.getBody()->getOperations())) {
        auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
        if (!symbolOp || !llvm::is_contained(funcNames, symbolOp.getName()))
          op.erase();
      }
      module.getBody()->getOperations().splice(
          module.getBody()->end(), preservedOperations.getOperations());
    }
  }

  Option<std::string> pipeline{
      *this, "pipeline",
      llvm::cl::desc("Optional pass pipeline to run on filtered functions"),
      llvm::cl::init("")};

  ListOption<std::string> funcNames{
      *this, "func-name",
      llvm::cl::desc("Function names to keep (if empty, keep all)")};

  Option<bool> externalizeDependencies{
      *this, "externalize-dependencies",
      llvm::cl::desc("Keep transitive function dependencies as declarations"),
      llvm::cl::init(false)};

  Option<bool> preserveModule{
      *this, "preserve-module",
      llvm::cl::desc("Run the pipeline only on selected roots and restore all "
                     "other top-level operations unchanged"),
      llvm::cl::init(false)};
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createSelectFuncPass() {
  return std::make_unique<SelectFuncPass>();
}
} // namespace polygeist
} // namespace mlir
