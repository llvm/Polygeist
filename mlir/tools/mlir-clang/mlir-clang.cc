#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <fstream>

using namespace llvm;

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::init(false),
                              cl::desc("Emit llvm"));

static cl::opt<std::string> Standard("std", cl::init(""),
                                     cl::desc("C/C++ std"));

static cl::opt<std::string> Output("o", cl::init("-"),
                              cl::desc("Output file"));

static cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                           cl::desc("<Specify input file>"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> cfunction("function",
                                      cl::desc("<Specify function>"),
                                      cl::init("main"), cl::cat(toolOptions));

static cl::opt<bool> FOpenMP("fopenmp", cl::init(false),
                              cl::desc("Enable OpenMP"));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::list<std::string> defines("D", cl::desc("defines"),
                                     cl::cat(toolOptions));

#include "Lib/clang-mlir.cc"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
int main(int argc, char **argv) {

  using namespace mlir;

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);
  assert(inputFileName.size());
  for (auto inp : inputFileName) {
    std::ifstream inputFile(inp);
    if (!inputFile.good()) {
      outs() << "Not able to open file: " << inp << "\n";
      return -1;
    }
  }

  // registerDialect<AffineDialect>();
  // registerDialect<StandardOpsDialect>();
  MLIRContext context;
  context.disableMultithreading();
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

  llvm::Triple triple;
  llvm::DataLayout DL("");
  parseMLIR(argv[0], inputFileName, cfunction, includeDirs, defines, module, triple, DL);
  mlir::PassManager pm(&context);

  llvm::errs() << "<immediate: mlir>\n";
  module.dump();
  llvm::errs() << "</immediate: mlir>\n";
  pm.enableVerifier(false);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (true) {
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createMem2RegPass());
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createMem2RegPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createLoopRestructurePass());
  // optPM.addPass(mlir::createAffineLoopInvariantCodeMotionPass());
  optPM.addPass(mlir::createRaiseSCFToAffinePass());
  // optPM.addPass(mlir::replaceAffineCFGPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createMemRefDataFlowOptPass());
  if (mlir::failed(pm.run(module))) {
    module.dump();
    return 4;
  }
  if (mlir::failed(mlir::verify(module))) {
    module.dump();
    return 5;
  }

#define optPM optPM2
#define pm pm2
    mlir::PassManager pm(&context);
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (CudaLower)
      optPM.addPass(mlir::createParallelLowerPass());

    if (EmitLLVM) {
      pm.addPass(mlir::createLowerAffinePass());
      pm.addPass(mlir::createLowerToCFGPass());
      LowerToLLVMOptions options;
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      options.dataLayout = DL;
      pm.addPass(mlir::createLowerToLLVMPass(options));
    }

    if (mlir::failed(pm.run(module)))
      return 4;
    // module.dump();
    if (mlir::failed(mlir::verify(module))) {
      return 5;
    }
  }

  if (EmitLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }
    llvmModule->setDataLayout(DL);
    llvmModule->setTargetTriple(triple.getTriple());
    if (Output == "-")
      llvm::outs() << *llvmModule << "\n";
    else {
      std::error_code EC;
     	llvm::raw_fd_ostream out(Output, EC);
      out << *llvmModule << "\n";
    }
 	
  } else {
    if (Output == "-")
      module.print(outs());
    else {
      std::error_code EC;
     	llvm::raw_fd_ostream out(Output, EC);
      module.print(outs());
    }
  }
  return 0;
}
