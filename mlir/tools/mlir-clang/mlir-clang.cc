#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <fstream>

using namespace llvm;

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<Specify input file>"),
                                          cl::Required, cl::cat(toolOptions));

static cl::opt<std::string> cfunction(cl::Positional,
                                      cl::desc("<Specify function>"),
                                      cl::Required, cl::cat(toolOptions));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

#include "Lib/clang-mlir.cc"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
int main(int argc, char **argv) {

  using namespace mlir;

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream inputFile(inputFileName);
  if (!inputFile.good()) {
    outs() << "Not able to open file: " << inputFileName << "\n";
    return -1;
  }

  // registerDialect<AffineDialect>();
  // registerDialect<StandardOpsDialect>();
  MLIRContext context;

  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  // MLIRContext context;

  if (showDialects) {
    outs() << "Registered Dialects:\n";
    for (Dialect *dialect : context.getLoadedDialects()) {
      outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }
  auto module =
      mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc());

  parseMLIR(inputFileName.c_str(), cfunction, includeDirs, module);
  mlir::PassManager pm(&context);

  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createMemRefDataFlowOptPass());
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  if (CudaLower)
    optPM.addPass(mlir::createParallelLowerPass());

  if (mlir::failed(pm.run(module)))
    return 4;

  module.print(outs());
  return 0;
}