#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"

#if POLYGEIST_ENABLE_ROCM
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Transforms/IPO/Internalize.h"

#include <mutex>

#define DEBUG_TYPE "polygeist-serialize-to-hsaco"

using namespace mlir;

namespace {
class SerializeToHsacoPass
    : public PassWrapper<SerializeToHsacoPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToHsacoPass)

  SerializeToHsacoPass(StringRef arch = "", StringRef features = "",
                       int llvmOptLevel = 3, int hsaOptLevel = 3,
                       std::string rocmPath = "/opt/rocm",
                       bool outputIntermediate = false);

  StringRef getArgument() const override { return "polygeist-gpu-to-hsaco"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to HSACO binary annotations";
  }

protected:
  LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                             llvm::TargetMachine &targetMachine) override;
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext) override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Loads LLVM bitcode libraries
  std::optional<SmallVector<std::unique_ptr<llvm::Module>, 3>>
  loadLibraries(SmallVectorImpl<char> &path,
                SmallVectorImpl<StringRef> &libraries,
                llvm::LLVMContext &context);

  std::unique_ptr<SmallVectorImpl<char>> assembleIsa(const std::string &isa);
  std::unique_ptr<std::vector<char>>
  createHsaco(const SmallVectorImpl<char> &isaBinary);

  // Serializes PTX to HSACO.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;

  int llvmOptLevel;
  int hsaOptLevel;
  bool outputIntermediate;
  std::string rocmPath;
};
} // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option, StringRef value) {
  if (!option.hasValue())
    option = value.str();
}

SerializeToHsacoPass::SerializeToHsacoPass(StringRef arch, StringRef features,
                                           int llvmOptLevel, int hsaOptLevel,
                                           std::string rocmPath,
                                           bool outputIntermediate) {
  maybeSetOption(this->chip, arch);
  maybeSetOption(this->features, features);
  this->llvmOptLevel = llvmOptLevel;
  this->hsaOptLevel = hsaOptLevel;
  this->outputIntermediate = outputIntermediate;
  this->rocmPath = rocmPath;
}

LogicalResult
SerializeToHsacoPass::optimizeLlvm(llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine) {
  if (llvmOptLevel < 0 || llvmOptLevel > 3)
    return getOperation().emitError()
           << "Invalid serizalize to gpu blob optimization level"
           << llvmOptLevel << "\n";

  targetMachine.setOptLevel(static_cast<llvm::CodeGenOptLevel>(llvmOptLevel));

  auto transformer =
      makeOptimizingTransformer(llvmOptLevel, /*sizeLevel=*/0, &targetMachine);
  auto error = transformer(&llvmModule);
  if (error) {
    InFlightDiagnostic mlirError = getOperation()->emitError();
    llvm::handleAllErrors(
        std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
          mlirError << "Could not optimize LLVM IR: " << ei.message() << "\n";
        });
    return mlirError;
  }

  for (auto &F : llvmModule) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto g = dyn_cast<llvm::GetElementPtrInst>(&I))
          g->setIsInBounds(true);
        if (isa<llvm::FPMathOperator>(&I))
          I.setHasAllowContract(true);
      }
    }
  }

  StripDebugInfo(llvmModule);

  if (outputIntermediate) {
    llvm::outs() << "Optimized GPU LLVM module for: "
                 << getOperation().getNameAttr() << "\n"
                 << llvmModule << "\n";
    llvm::outs().flush();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Optimized GPU LLVM module for: "
                 << getOperation().getNameAttr() << "\n";
    llvm::dbgs() << llvmModule << "\n";
    llvm::dbgs().flush();
  });

  return success();
}

std::unique_ptr<llvm::Module>
SerializeToHsacoPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  gpu::GPUModuleOp gpum = getOperation();

  mlir::ModuleOp m = gpum->getParentOfType<mlir::ModuleOp>();

  mlir::ModuleOp tmpModule(
      mlir::ModuleOp::create(mlir::OpBuilder(m->getContext()).getUnknownLoc()));
  // Prepare DL, triple attributes
  auto triple = m->getAttrOfType<StringAttr>(
      StringRef("polygeist.gpu_module." +
                LLVM::LLVMDialect::getTargetTripleAttrName().str()));
  this->triple = std::string(triple.getValue());

  // TODO This is the CUDA data layout - we need to somehow get the correct one
  // for our target AMDGPU - the modules linked in below contain the correct one
  auto DL = m->getAttrOfType<mlir::StringAttr>(
                 StringRef("polygeist.gpu_module." +
                           LLVM::LLVMDialect::getDataLayoutAttrName().str()))
                .getValue();
  tmpModule->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(), triple);
  tmpModule->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                     StringAttr::get(tmpModule->getContext(), DL));
  tmpModule->setAttr(
      ("dlti." + DataLayoutSpecAttr::kAttrKeyword).str(),
      translateDataLayout(llvm::DataLayout(DL), tmpModule->getContext()));

  tmpModule->getRegion(0).front().erase();
  IRMapping mapping;
  gpum->getRegion(0).cloneInto(&tmpModule->getRegion(0), mapping);

  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(tmpModule, llvmContext, gpum.getNameAttr());
  tmpModule->erase();

  if (!llvmModule)
    return llvmModule;

  if (outputIntermediate) {
    llvm::outs() << "Unoptimized GPU LLVM module for: "
                 << getOperation().getNameAttr() << "\n"
                 << *llvmModule << "\n";
    llvm::outs().flush();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Unoptimized GPU LLVM module for: "
                 << getOperation().getNameAttr() << "\n";
    llvm::dbgs() << *llvmModule << "\n";
    llvm::dbgs().flush();
  });

  // Set correct attributes for global __constant__ and __device__ variables
  for (auto &GV : llvmModule->globals()) {
    auto AS = GV.getAddressSpace();
    if (AS == 4 || AS == 1) {
      GV.setVisibility(
          llvm::GlobalVariable::VisibilityTypes::ProtectedVisibility);
      GV.setExternallyInitialized(true);
    }
  }

  auto ret = std::move(llvmModule);

  // Walk the LLVM module in order to determine if we need to link in device
  // libs
  bool needOpenCl = false;
  bool needOckl = false;
  bool needOcml = false;
  for (llvm::Function &f : ret->functions()) {
    if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
      StringRef funcName = f.getName();
      if ("printf" == funcName)
        needOpenCl = true;
      if (funcName.startswith("__ockl_"))
        needOckl = true;
      if (funcName.startswith("__ocml_"))
        needOcml = true;
    }
  }

  if (needOpenCl)
    needOcml = needOckl = true;

  // No libraries needed (the typical case)
  if (!(needOpenCl || needOcml || needOckl))
    return ret;

  // Define one of the control constants the ROCm device libraries expect to be
  // present These constants can either be defined in the module or can be
  // imported by linking in bitcode that defines the constant. To simplify our
  // logic, we define the constants into the module we are compiling
  auto addControlConstant = [&module = *ret](StringRef name, uint32_t value,
                                             uint32_t bitwidth) {
    using llvm::GlobalVariable;
    if (module.getNamedGlobal(name)) {
      return;
    }
    llvm::IntegerType *type =
        llvm::IntegerType::getIntNTy(module.getContext(), bitwidth);
    auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
    auto *constant = new GlobalVariable(
        module, type,
        /*isConstant=*/true, GlobalVariable::LinkageTypes::LinkOnceODRLinkage,
        initializer, name,
        /*before=*/nullptr,
        /*threadLocalMode=*/GlobalVariable::ThreadLocalMode::NotThreadLocal,
        /*addressSpace=*/4);
    constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
    constant->setVisibility(
        GlobalVariable::VisibilityTypes::ProtectedVisibility);
    constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  };

  // Set up control variables in the module instead of linking in tiny bitcode
  if (needOcml) {
    // TODO(kdrewnia): Enable math optimizations once we have support for
    // `-ffast-math`-like options
    addControlConstant("__oclc_finite_only_opt", 0, 8);
    addControlConstant("__oclc_daz_opt", 0, 8);
    addControlConstant("__oclc_correctly_rounded_sqrt32", 1, 8);
    addControlConstant("__oclc_unsafe_math_opt", 0, 8);
  }
  if (needOcml || needOckl) {
    addControlConstant("__oclc_wavefrontsize64", 1, 8);
    StringRef chipSet = this->chip.getValue();
    if (chipSet.startswith("gfx"))
      chipSet = chipSet.substr(3);
    uint32_t minor =
        llvm::APInt(32, chipSet.substr(chipSet.size() - 2), 16).getZExtValue();
    uint32_t major = llvm::APInt(32, chipSet.substr(0, chipSet.size() - 2), 10)
                         .getZExtValue();
    uint32_t isaNumber = minor + 1000 * major;
    addControlConstant("__oclc_ISA_version", isaNumber, 32);

    // This constant must always match the default code object ABI version
    // of the AMDGPU backend.
    addControlConstant("__oclc_ABI_version", 400, 32);
  }

  // Determine libraries we need to link - order matters due to dependencies
  llvm::SmallVector<StringRef, 4> libraries;
  if (needOpenCl)
    libraries.push_back("opencl.bc");
  if (needOcml)
    libraries.push_back("ocml.bc");
  if (needOckl)
    libraries.push_back("ockl.bc");

  std::optional<SmallVector<std::unique_ptr<llvm::Module>, 3>> mbModules;
  std::string theRocmPath = rocmPath;
  llvm::SmallString<32> bitcodePath(theRocmPath);
  llvm::sys::path::append(bitcodePath, "amdgcn", "bitcode");
  mbModules = loadLibraries(bitcodePath, libraries, llvmContext);

  if (!mbModules) {
    getOperation()
            .emitWarning("Could not load required device libraries")
            .attachNote()
        << "This will probably cause link-time or run-time failures";
    return ret; // We can still abort here
  }

  llvm::Linker linker(*ret);
  for (std::unique_ptr<llvm::Module> &libModule : *mbModules) {
    // This bitcode linking code is substantially similar to what is used in
    // hip-clang It imports the library functions into the module, allowing LLVM
    // optimization passes (which must run after linking) to optimize across the
    // libraries and the module's code. We also only import symbols if they are
    // referenced by the module or a previous library since there will be no
    // other source of references to those symbols in this compilation and since
    // we don't want to bloat the resulting code object.
    bool err = linker.linkInModule(
        std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &m, const StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });
    // True is linker failure
    if (err) {
      getOperation().emitError(
          "Unrecoverable failure during device library linking.");
      // We have no guaranties about the state of `ret`, so bail
      return nullptr;
    }
  }

  return ret;
}

std::unique_ptr<SmallVectorImpl<char>>
SerializeToHsacoPass::assembleIsa(const std::string &isa) {
  auto loc = getOperation().getLoc();

  SmallVector<char, 0> result;
  llvm::raw_svector_ostream os(result);

  llvm::Triple triple(llvm::Triple::normalize(this->triple));
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return {};
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa), SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(this->triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, this->triple, mcOptions));
  mai->setRelaxELFRelocations(true);
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(this->triple, this->chip, this->features));

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(target->createMCObjectStreamer(
      triple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti, mcOptions.MCRelaxAll, mcOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  mcStreamer->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    emitError(loc, "assembler initialization error");
    return {};
  }

  parser->setTargetParser(*tap);
  parser->Run(false);

  return std::make_unique<SmallVector<char, 0>>(std::move(result));
}

std::unique_ptr<std::vector<char>>
SerializeToHsacoPass::createHsaco(const SmallVectorImpl<char> &isaBinary) {
  auto loc = getOperation().getLoc();

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    emitError(loc, "temporary file for ISA binary creation error");
    return {};
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << StringRef(isaBinary.data(), isaBinary.size());
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    emitError(loc, "temporary file for HSA code object creation error");
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  std::string theRocmPath = rocmPath;
  llvm::SmallString<32> lldPath(theRocmPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath,
      {"ld.lld", "-shared", tempIsaBinaryFilename, "-o", tempHsacoFilename});
  if (lldResult != 0) {
    emitError(loc, "lld invocation error");
    return {};
  }

  // Load the HSA code object.
  auto hsacoFile = openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    emitError(loc, "read HSA code object from temp file error");
    return {};
  }

  StringRef buffer = hsacoFile->getBuffer();
  return std::make_unique<std::vector<char>>(buffer.begin(), buffer.end());
}

std::unique_ptr<std::vector<char>>
SerializeToHsacoPass::serializeISA(const std::string &isa) {
  if (outputIntermediate) {
    llvm::outs() << "AMD isa for: " << getOperation().getNameAttr() << "\n"
                 << isa << "\n";
    llvm::outs().flush();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "AMD isa for: " << getOperation().getNameAttr() << "\n"
                 << isa << "\n";
    llvm::dbgs().flush();
  });

  auto isaBinary = assembleIsa(isa);
  if (!isaBinary)
    return {};
  return createHsaco(*isaBinary);
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>, 3>>
SerializeToHsacoPass::loadLibraries(SmallVectorImpl<char> &path,
                                    SmallVectorImpl<StringRef> &libraries,
                                    llvm::LLVMContext &context) {
  SmallVector<std::unique_ptr<llvm::Module>, 3> ret;
  size_t dirLength = path.size();

  if (!llvm::sys::fs::is_directory(path)) {
    getOperation().emitRemark() << "Bitcode path: " << path
                                << " does not exist or is not a directory\n";
    return {};
  }

  for (const StringRef file : libraries) {
    llvm::SMDiagnostic error;
    llvm::sys::path::append(path, file);
    llvm::StringRef pathRef(path.data(), path.size());
    std::unique_ptr<llvm::Module> library =
        llvm::getLazyIRFileModule(pathRef, error, context);
    path.truncate(dirLength);
    if (!library) {
      getOperation().emitError() << "Failed to load library " << file
                                 << " from " << path << error.getMessage();
      return {};
    }
    // Some ROCM builds don't strip this like they should
    if (auto *openclVersion = library->getNamedMetadata("opencl.ocl.version"))
      library->eraseNamedMetadata(openclVersion);
    // Stop spamming us with clang version numbers
    if (auto *ident = library->getNamedMetadata("llvm.ident"))
      library->eraseNamedMetadata(ident);
    ret.push_back(std::move(library));
  }

  return ret;
}

void SerializeToHsacoPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerROCDLDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

namespace mlir::polygeist {

// Register pass to serialize GPU kernel functions to a HSACO binary annotation.
void registerGpuSerializeToHsacoPass() {
  PassRegistration<SerializeToHsacoPass> registerSerializeToHsaco([] {
    // Initialize LLVM AMDGPU backend.
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();

    return std::make_unique<SerializeToHsacoPass>();
  });
}

std::unique_ptr<Pass>
createGpuSerializeToHsacoPass(StringRef arch, StringRef features,
                              int llvmOptLevel, int hsaOptLevel,
                              std::string rocmPath, bool outputIntermediate) {
  return std::make_unique<SerializeToHsacoPass>(
      arch, features, llvmOptLevel, hsaOptLevel, rocmPath, outputIntermediate);
}

} // namespace mlir::polygeist

#else
namespace mlir::polygeist {
void registerGpuSerializeToHsacoPass() {}
} // namespace mlir::polygeist
#endif
