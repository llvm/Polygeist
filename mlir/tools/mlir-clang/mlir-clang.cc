#include "Lib/clang-mlir.h"

#include <fstream>

#include "clang/AST/Attr.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Pragma.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/Debug.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace llvm::opt;
using namespace clang;
using namespace clang;
using namespace clang::driver;
using namespace mlir;

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::init(false),
                              cl::desc("Emit llvm"));

static cl::opt<bool> ImmediateMLIR("immediate", cl::init(false),
                                   cl::desc("Emit immediate mlir"));

static cl::opt<bool> RaiseToAffine("raise-scf-to-affine", cl::init(false),
                                   cl::desc("Raise SCF to Affine"));

static cl::opt<bool>
    DetectReduction("detect-reduction", cl::init(false),
                    cl::desc("Detect reduction in inner most loop"));

static cl::opt<std::string> Standard("std", cl::init(""),
                                     cl::desc("C/C++ std"));

static cl::opt<std::string> Output("o", cl::init("-"), cl::desc("Output file"));

static cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                           cl::desc("<Specify input file>"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> cfunction("function",
                                      cl::desc("<Specify function>"),
                                      cl::init("main"), cl::cat(toolOptions));

static cl::opt<bool> FOpenMP("fopenmp", cl::init(false),
                             cl::desc("Enable OpenMP"));

static cl::opt<std::string> MArch("march", cl::init(""),
                                  cl::desc("Architecture"));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::list<std::string> defines("D", cl::desc("defines"),
                                     cl::cat(toolOptions));

static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines, mlir::ModuleOp &module,
                      llvm::Triple &triple, llvm::DataLayout &DL) {

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  bool Success;
  //{
  const char *binary = Argv0; // CudaLower ? "clang++" : "clang";
  const std::unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  std::vector<const char *> Argv;
  Argv.push_back(binary);
  for (auto a : filenames) {
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (CudaLower)
    Argv.push_back("--cuda-gpu-arch=sm_35");
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (Standard != "") {
    auto a = "-std=" + Standard;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (MArch != "") {
    auto a = "-march=" + MArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : includeDirs) {
    Argv.push_back("-I");
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : defines) {
    char *chars = (char *)malloc(a.length() + 3);
    chars[0] = '-';
    chars[1] = 'D';
    memcpy(chars + 2, a.data(), a.length());
    chars[2 + a.length()] = 0;
    Argv.push_back(chars);
  }

  Argv.push_back("-emit-ast");

  const std::unique_ptr<Compilation> compilation(
      driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
  JobList &Jobs = compilation->getJobs();
  if (Jobs.size() < 1)
    return false;

  MLIRAction Act(fn, module, RaiseToAffine);

  for (auto &job : Jobs) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Command *cmd = cast<Command>(&job);
    if (strcmp(cmd->getCreator().getName(), "clang"))
      return false;

    const ArgStringList *args = &cmd->getArguments();

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
      Clang->getHeaderSearchOpts().ResourceDir =
          CompilerInvocation::GetResourcesPath(Argv0, GetExecutablePathVP);

    //}
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
    Clang->setTarget(TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
    if (!Clang->hasTarget())
      return false;

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((Clang->getLangOpts().CUDA || Clang->getLangOpts().OpenMPIsDevice) &&
        !Clang->getFrontendOpts().AuxTriple.empty()) {
      auto TO = std::make_shared<clang::TargetOptions>();
      TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
      TO->HostTriple = Clang->getTarget().getTriple().str();
      Clang->setAuxTarget(
          TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getLangOpts());

    // Adjust target options based on codegen options.
    Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                           Clang->getTargetOpts());

    module->setAttr(
        LLVM::LLVMDialect::getDataLayoutAttrName(),
        StringAttr::get(
            module.getContext(),
            Clang->getTarget().getDataLayout().getStringRepresentation()));
    module->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(module.getContext(),
                        Clang->getTarget().getTriple().getTriple()));

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {

        llvm::Error err = Act.Execute();
        if (err) {
          llvm::errs() << "saw error: " << err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }
    DL = Clang->getTarget().getDataLayout();
    triple = Clang->getTarget().getTriple();
  }
  return true;
}

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
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  context.disableMultithreading();
  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
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
    optPM.addPass(mlir::createMem2RegPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createLoopRestructurePass());
    optPM.addPass(mlir::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
    optPM.addPass(mlir::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    if (RaiseToAffine) {
      optPM.addPass(mlir::createCanonicalizeForPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
      optPM.addPass(mlir::createRaiseSCFToAffinePass());
      optPM.addPass(mlir::replaceAffineCFGPass());
    }
    if (DetectReduction)
      optPM.addPass(mlir::detectReductionPass());
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
      LowerToLLVMOptions options(&context);
      options.dataLayout = DL;
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      pm.addPass(mlir::createLowerToLLVMPass(options));
    }

    if (mlir::failed(pm.run(module))) {
      module.dump();
      return 4;
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
