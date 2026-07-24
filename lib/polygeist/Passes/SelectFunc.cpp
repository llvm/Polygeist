//===- SelectFunc.cpp - Filter and output only selected functions
//----------===//
//
// This file implements a pass to filter functions by name, removing all
// functions that don't match the specified names.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
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

    // Keep the requested roots and the transitive symbol dependencies they
    // reference. Previously this pass erased declarations such as `@logf`
    // while leaving calls in the selected function, producing invalid IR.
    llvm::SmallPtrSet<Operation *, 16> keep;
    SmallVector<Operation *> worklist;
    for (Operation &op : module.getBody()->getOperations()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
      if (symbolOp && llvm::is_contained(funcNames, symbolOp.getName()) &&
          keep.insert(&op).second)
        worklist.push_back(&op);
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
      }
    }
  }

  Option<std::string> pipeline{
      *this, "pipeline",
      llvm::cl::desc("Optional pass pipeline to run on filtered functions"),
      llvm::cl::init("")};

  ListOption<std::string> funcNames{
      *this, "func-name",
      llvm::cl::desc("Function names to keep (if empty, keep all)")};
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createSelectFuncPass() {
  return std::make_unique<SelectFuncPass>();
}
} // namespace polygeist
} // namespace mlir
