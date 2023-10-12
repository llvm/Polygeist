#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "polygeist/Tools/MergeHostDeviceGPUModules.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

// Parse and verify the input MLIR file. Returns null on error.
OwningOpRef<Operation *> loadModule(MLIRContext &context,
                                    StringRef inputFilename,
                                    bool insertImplictModule) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFileForTool(sourceMgr, &context, insertImplictModule);
}

namespace mlir {
namespace polygeist {
static LogicalResult polygeistMergeGPUModulesMain(int argc, char **argv,
                                                  MLIRContext &context) {
  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory polygeistMergeOpts(
      "polygeist-merge-gpu-modules options");

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output merged MLIR module name"),
      llvm::cl::init("-"), llvm::cl::cat(polygeistMergeOpts));
  static llvm::cl::opt<std::string> deviceModuleFilename(
      "device", llvm::cl::desc("Input device MLIR module"),
      llvm::cl::cat(polygeistMergeOpts));
  static llvm::cl::opt<std::string> hostModuleFilename(
      "host", llvm::cl::desc("Input host MLIR module"),
      llvm::cl::cat(polygeistMergeOpts));

  llvm::cl::HideUnrelatedOptions(polygeistMergeOpts);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Polygeist tool for mergin gpu host and device modules.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  std::string errorMessage;

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    return failure();

  OwningOpRef<Operation *> hostOpRef =
      loadModule(context, hostModuleFilename, false);
  if (!hostOpRef)
    return failure();

  OwningOpRef<Operation *> deviceOpRef =
      loadModule(context, deviceModuleFilename, false);
  if (!deviceOpRef)
    return failure();

  if (polygeist::mergeDeviceIntoHost(cast<ModuleOp>(hostOpRef.get()),
                                     cast<ModuleOp>(deviceOpRef.get()))
          .failed())
    return failure();

  OwningOpRef<Operation *> op = hostOpRef.get()->clone();

  op.get()->print(output->os());
  output->keep();

  return success();
}
} // namespace polygeist
} // namespace mlir

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  // TODO put this in a register function for the mergeDeviceIntoHost func
  MLIRContext context(registry);
  context.loadDialect<mlir::gpu::GPUDialect>();

  return failed(
      mlir::polygeist::polygeistMergeGPUModulesMain(argc, argv, context));
}
