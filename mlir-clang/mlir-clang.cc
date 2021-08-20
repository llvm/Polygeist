#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <fstream>

#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

using namespace llvm;

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::init(false),
                              cl::desc("Emit llvm"));

static cl::opt<bool> SCFOpenMP("scf-openmp", cl::init(true),
                               cl::desc("Emit llvm"));

static cl::opt<bool> ShowAST("show-ast", cl::init(false), cl::desc("Show AST"));

static cl::opt<bool> ImmediateMLIR("immediate", cl::init(false),
                                   cl::desc("Emit immediate mlir"));

static cl::opt<bool> RaiseToAffine("raise-scf-to-affine", cl::init(false),
                                   cl::desc("Raise SCF to Affine"));

static cl::opt<bool> ScalarReplacement("scal-rep", cl::init(true),
                                       cl::desc("Raise SCF to Affine"));

static cl::opt<bool>
    DetectReduction("detect-reduction", cl::init(false),
                    cl::desc("Detect reduction in inner most loop"));

static cl::opt<std::string> Standard("std", cl::init(""),
                                     cl::desc("C/C++ std"));

static cl::opt<std::string> CUDAGPUArch("cuda-gpu-arch", cl::init(""),
                                        cl::desc("CUDA GPU arch"));

static cl::opt<std::string> Output("o", cl::init("-"), cl::desc("Output file"));

static cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                           cl::desc("<Specify input file>"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> cfunction("function",
                                      cl::desc("<Specify function>"),
                                      cl::init("main"), cl::cat(toolOptions));

static cl::opt<bool> FOpenMP("fopenmp", cl::init(false),
                             cl::desc("Enable OpenMP"));

static cl::opt<bool> ToCPU("cpuify", cl::init(false),
                           cl::desc("Convert to cpu"));

static cl::opt<std::string> MArch("march", cl::init(""),
                                  cl::desc("Architecture"));

static cl::opt<std::string> ResourceDir("resource-dir", cl::init(""),
                                        cl::desc("Resource-dir"));

static cl::opt<bool> Verbose("v", cl::init(false), cl::desc("Verbose"));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::list<std::string> defines("D", cl::desc("defines"),
                                     cl::cat(toolOptions));

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

#include "Lib/clang-mlir.cc"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
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
  mlir::DialectRegistry registry;
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  context.disableMultithreading();
  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();
  // MLIRContext context;

  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);

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
  parseMLIR(argv[0], inputFileName, cfunction, includeDirs, defines, module,
            triple, DL);
  mlir::PassManager pm(&context);

  if (ImmediateMLIR) {
    llvm::errs() << "<immediate: mlir>\n";
    module.dump();
    llvm::errs() << "</immediate: mlir>\n";
  }
  pm.enableVerifier(false);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (true) {
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createRemoveTrivialUsePass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createLoopRestructurePass());
    if (!CudaLower)
      optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    if (ScalarReplacement)
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    if (RaiseToAffine && !CudaLower) {
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
      optPM.addPass(polygeist::createRaiseSCFToAffinePass());
      optPM.addPass(polygeist::replaceAffineCFGPass());
      if (ScalarReplacement)
        optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
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

    if (DetectReduction)
      optPM.addPass(polygeist::detectReductionPass());

    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (CudaLower) {
      optPM.addPass(polygeist::createParallelLowerPass());
      optPM.addPass(polygeist::replaceAffineCFGPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(polygeist::createMem2RegPass());
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::createCanonicalizerPass());

      if (RaiseToAffine) {
        optPM.addPass(polygeist::createCanonicalizeForPass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
        optPM.addPass(polygeist::createRaiseSCFToAffinePass());
        optPM.addPass(polygeist::replaceAffineCFGPass());
        if (ScalarReplacement)
          optPM.addPass(mlir::createAffineScalarReplacementPass());
      }
      if (ToCPU)
        optPM.addPass(polygeist::createCPUifyPass());
    }

    if (EmitLLVM) {
      pm.addPass(mlir::createLowerAffinePass());
      pm.nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
      if (mlir::failed(pm.run(module))) {
        module.dump();
        return 4;
      }
      mlir::PassManager pm2(&context);
      if (SCFOpenMP)
        pm2.nest<mlir::FuncOp>().addPass(createConvertSCFToOpenMPPass());
      if (mlir::failed(pm2.run(module))) {
        module.dump();
        return 4;
      }
      mlir::PassManager pm3(&context);
      pm3.addPass(mlir::createLowerToCFGPass());
      pm3.addPass(createConvertOpenMPToLLVMPass());
      LowerToLLVMOptions options(&context);
      options.dataLayout = DL;
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      pm3.addPass(mlir::createLowerToLLVMPass(options));
      if (mlir::failed(pm3.run(module))) {
        module.dump();
        return 4;
      }
    } else {

      if (mlir::failed(pm.run(module))) {
        module.dump();
        return 4;
      }
    }
    // module.dump();
    if (mlir::failed(mlir::verify(module))) {
      module.dump();
      return 5;
    }
  }

  if (EmitLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      module.dump();
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
      module.print(out);
    }
  }
  return 0;
}
