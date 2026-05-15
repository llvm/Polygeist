//===- LowerKernelLaunch.cpp - inline kernel.defn bodies into launches ----===//
//
// Phase-2 lowering for the kernel-matcher pipeline. For each `kernel.launch
// @<name>(operands)` op, finds `kernel.defn @<name>` (in the same module or
// in a separately-loaded library file via the `kernel-library-path` option),
// clones the defn body into the launch's parent block, maps defn block args
// to launch operands, and replaces the launch's result SSA with the value
// yielded by `kernel.yield`. The kernel.launch is then erased.
//
// Phase-1 of the pipeline (kernel_match_rewrite.py --with-roundtrip-markers
// + kernel_launch_lower.py) stashes the matcher's pre-match linalg verbatim
// and restores it; that validates plumbing but not matcher labels because
// the round-trip is a no-op by construction. Phase-2 (this pass) substitutes
// a *canonical* linalg implementation from the library so that a
// wrongly-labeled kernel.launch produces different numerics from the user's
// original code and fails the e2e diff against clang.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "lower-kernel-launch"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

// Returns the DefnOp inside `module` (or `library`) named `name`, or nullptr.
static DefnOp findDefn(ModuleOp module, ModuleOp library, StringRef name) {
  if (auto d = module.lookupSymbol<DefnOp>(name))
    return d;
  if (library)
    return library.lookupSymbol<DefnOp>(name);
  return nullptr;
}

// Inline the body of `defn` in place of `launch`. The defn's block arguments
// are mapped to the launch's operands; the defn's terminating kernel.yield
// values are substituted for the launch's results.
//
// Returns success iff the substitution completed and the launch was erased.
static LogicalResult inlineDefnIntoLaunch(LaunchOp launch, DefnOp defn) {
  if (defn.isDeclaration())
    return launch.emitError("kernel.defn '") << defn.getSymName() << "' is a declaration (empty body); cannot inline";

  Block &defnBlock = defn.getBody().front();
  if (defnBlock.getNumArguments() != launch.getOperands().size())
    return launch.emitError("kernel.launch operand count (")
           << launch.getOperands().size()
           << ") does not match kernel.defn '" << defn.getSymName()
           << "' parameter count (" << defnBlock.getNumArguments() << ")";

  IRMapping mapping;
  for (auto [blockArg, operand] :
       llvm::zip(defnBlock.getArguments(), launch.getOperands())) {
    if (blockArg.getType() != operand.getType())
      return launch.emitError("operand type mismatch: kernel.defn '")
             << defn.getSymName() << "' expects " << blockArg.getType()
             << " for parameter, got " << operand.getType();
    mapping.map(blockArg, operand);
  }

  // Clone every op except the terminator into the launch's parent block,
  // immediately before the launch.
  OpBuilder builder(launch);
  YieldOp yield;
  for (Operation &op : defnBlock.without_terminator()) {
    builder.clone(op, mapping);
  }
  // Find the terminator (kernel.yield) and resolve the launch's results.
  yield = cast<YieldOp>(defnBlock.getTerminator());
  if (yield.getNumOperands() != launch.getNumResults())
    return launch.emitError("kernel.yield arity (")
           << yield.getNumOperands() << ") does not match kernel.launch result arity ("
           << launch.getNumResults() << ")";

  SmallVector<Value> remappedResults;
  for (Value y : yield.getOperands()) {
    Value mapped = mapping.lookupOrNull(y);
    if (!mapped)
      return launch.emitError("kernel.yield references value not produced by inlined body");
    remappedResults.push_back(mapped);
  }
  launch.replaceAllUsesWith(remappedResults);
  launch.erase();
  return success();
}

struct LowerKernelLaunchPass
    : public mlir::polygeist::LowerKernelLaunchBase<LowerKernelLaunchPass> {

  // Helper: parse the kernel library file (if a path was given). Returns
  // an OwningOpRef that must outlive any DefnOp lookups against the library.
  OwningOpRef<ModuleOp> loadLibrary(MLIRContext *ctx) {
    if (kernelLibraryPath.empty())
      return OwningOpRef<ModuleOp>();
    std::string err;
    auto fileOrErr = openInputFile(kernelLibraryPath, &err);
    if (!fileOrErr) {
      getOperation().emitError(
          "lower-kernel-launch: cannot open kernel-library-path '")
          << kernelLibraryPath << "': " << err;
      return OwningOpRef<ModuleOp>();
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(fileOrErr), llvm::SMLoc());
    auto parsed = parseSourceFile<ModuleOp>(sourceMgr, ctx);
    if (!parsed) {
      getOperation().emitError(
          "lower-kernel-launch: failed to parse kernel library at '")
          << kernelLibraryPath << "'";
    }
    return parsed;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> libraryHolder = loadLibrary(module.getContext());
    ModuleOp library = libraryHolder ? libraryHolder.get() : ModuleOp();

    // Collect the launches up front; we'll erase them as we go.
    SmallVector<LaunchOp> launches;
    module.walk([&](LaunchOp op) { launches.push_back(op); });

    for (LaunchOp launch : launches) {
      auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!sym) {
        launch.emitError("kernel.launch missing 'kernel' symbol ref");
        signalPassFailure();
        return;
      }
      DefnOp defn = findDefn(module, library, sym.getLeafReference().getValue());
      if (!defn) {
        launch.emitError("lower-kernel-launch: no kernel.defn @")
            << sym.getLeafReference().getValue()
            << " found in input module or library";
        signalPassFailure();
        return;
      }
      if (failed(inlineDefnIntoLaunch(launch, defn))) {
        signalPassFailure();
        return;
      }
    }

    // After inlining, any kernel.defn ops in the *input* module that have no
    // remaining uses are dead — they were just symbol carriers. Don't touch
    // the library module (it's separately owned).
    SmallVector<DefnOp> deadDefns;
    module.walk([&](DefnOp d) {
      if (SymbolTable::symbolKnownUseEmpty(d, module))
        deadDefns.push_back(d);
    });
    for (DefnOp d : deadDefns)
      d.erase();
  }
};

} // anonymous namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerKernelLaunchPass() {
  return std::make_unique<LowerKernelLaunchPass>();
}
} // namespace polygeist
} // namespace mlir
