//===- LowerGPUToCUBIN.cpp - Convert GPU kernel to CUBIN blob -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that serializes a gpu module into CUBIN blob and
// adds that blob as a string attribute of the module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"


#if POLYGEIST_ENABLE_CUDA
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DebugInfo.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cuda.h>
// TODO use this library if possible
#include <nvPTXCompiler.h>

using namespace mlir;

static void emitCudaError(const llvm::Twine &expr, const char *buffer,
                          CUresult result, Location loc) {
  const char *error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                                             \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitCudaError(#expr, jitErrorBuffer, status, loc);                       \
      return {};                                                               \
    }                                                                          \
  } while (false)

#define RETURN_ON_NVPTX_ERROR(x)                                        \
  do {                                                                  \
    nvPTXCompileResult result = x;                                      \
    if (result != NVPTXCOMPILE_SUCCESS) {                               \
      emitError(loc, llvm::Twine("error: ").concat(#x).concat(" failed with error code ").concat(std::to_string(result))); \
      return {};                                                        \
    }                                                                   \
  } while(0)


namespace {
class SerializeToCubinPass
    : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)

  SerializeToCubinPass(StringRef triple = "nvptx64-nvidia-cuda", StringRef chip = "sm_35",
                       StringRef features = "+ptx60", int optLevel = 3,
                       std::string ptxasExe = "",
                       std::string libDeviceFile = "");

  StringRef getArgument() const override { return "gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }

protected:
  LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                             llvm::TargetMachine &targetMachine) override;
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext) override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes PTX to CUBIN.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;

  std::string ptxasExe;
  std::string libDeviceFile;
  int optLevel;

};
} // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option, StringRef value) {
  if (!option.hasValue())
    option = value.str();
}

SerializeToCubinPass::SerializeToCubinPass(StringRef triple,
                                           StringRef chip, StringRef features,
                                           int optLevel,
                                           std::string ptxasExe,
                                           std::string libDeviceFile) {
  maybeSetOption(this->triple, triple);
  maybeSetOption(this->chip, chip);
  maybeSetOption(this->features, features);
  this->optLevel = optLevel;
  this->ptxasExe = ptxasExe;
  this->libDeviceFile = libDeviceFile;
}

void SerializeToCubinPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<llvm::Module>
SerializeToCubinPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {

  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(getOperation(), llvmContext,
                                 "LLVMDialectModule");
  if (!llvmModule)
    return llvmModule;

  #ifndef NDEBUG
  llvm::errs() << "GPULLVM\n";
  llvmModule->dump();
  llvm::errs() << "GPULLVM\n";
  #endif

  // TODO get libdevice path
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> libDevice = llvm::parseIRFile(libDeviceFile, err, llvmContext);
  if (!libDevice || llvm::verifyModule(*libDevice, &llvm::errs())) {
    err.print("in serialize-to-blob", llvm::errs());
    // TODO what should the
    //unsigned diagID = ci.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "Could not parse IR");
    //ci.getDiagnostics().Report(diagID);
    return llvmModule;
  }
  // TODO do we need any flags?
  // TODO Internalize all but the public kernel function (https://llvm.org/docs/NVPTXUsage.html)
  llvm::Linker::linkModules(*llvmModule, std::move(libDevice));
  #ifndef NDEBUG
  llvm::errs() << "GPULLVM\n";
  llvmModule->dump();
  llvm::errs() << "GPULLVM\n";
  #endif

  return llvmModule;
}

LogicalResult
SerializeToCubinPass::optimizeLlvm(llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine) {
  int optLevel = this->optLevel;
  // TODO check cuda opt levels
  if (optLevel < 0 || optLevel > 3)
    return getOperation().emitError()
           << "Invalid serizalize to gpu blob optimization level" << optLevel << "\n";

  targetMachine.setOptLevel(static_cast<llvm::CodeGenOpt::Level>(optLevel));

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, &targetMachine);
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

  #ifndef NDEBUG
  llvm::errs() << "OPTGPULLVM\n";
  llvmModule.dump();
  llvm::errs() << "OPTGPULLVM\n";
  #endif
  StripDebugInfo(llvmModule);


  return success();
}

std::unique_ptr<std::vector<char>>
SerializeToCubinPass::serializeISA(const std::string &isa) {
  Location loc = getOperation().getLoc();

  llvm::errs() << isa << "\n";

  auto tmpInput =
    llvm::sys::fs::TempFile::create("/tmp/isainput%%%%%%%.s");
  if (!tmpInput) {
    llvm::errs() << "Failed to create temp file\n";
    return {};
  }
  {
    llvm::raw_fd_ostream out(tmpInput->FD, /*shouldClose*/ false);
    out << isa << "\n";
    out.flush();
  }
  auto tmpOutput =
    llvm::sys::fs::TempFile::create("/tmp/cubinoutput%%%%%%%.cubin");
  if (!tmpOutput) {
    llvm::errs() << "Failed to create temp file\n";
    return {};
  }

  std::vector<StringRef> Argv;
  Argv.push_back(ptxasExe);
  Argv.push_back(llvm::Triple(triple).isArch64Bit() ? "-m64" : "-m32");
  Argv.push_back("--gpu-name");
  Argv.push_back(chip.c_str());
  Argv.push_back("--opt-level");
  Argv.push_back(std::to_string(3));
  Argv.push_back("--verbose");
  Argv.push_back("--output-file");
  Argv.push_back(tmpOutput->TmpName);
  Argv.push_back(tmpInput->TmpName);

  llvm::sys::ExecuteAndWait(ptxasExe.c_str(), Argv);

  auto MB = llvm::MemoryBuffer::getFile(tmpOutput->TmpName, false, false, false);
  if (MB.getError()) {
    llvm::errs() << loc << "MemoryBuffer getFile failed";
    return {};
  }
  auto membuf = std::move(*MB);

  // TODO if one failed
  if (tmpOutput->discard())
    llvm::errs() << "Failed to erase temp file\n";
  if (tmpInput->discard())
    llvm::errs() << "Failed to erase temp file\n";

  size_t cubinSize = membuf->getBufferSize();
  auto result = std::make_unique<std::vector<char>>(cubinSize);
  memcpy(&(*result)[0], membuf->getBufferStart(), cubinSize);

  return result;
}

namespace mlir::polygeist {

// Register pass to serialize GPU kernel functions to a CUBIN binary annotation.
void registerGpuSerializeToCubinPass() {
  PassRegistration<SerializeToCubinPass> registerSerializeToCubin([] {
    // Initialize LLVM NVPTX backend.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    return std::make_unique<SerializeToCubinPass>();
  });
}

std::unique_ptr<Pass> createGpuSerializeToCubinPass(StringRef triple,
                                                    StringRef arch,
                                                    StringRef features,
                                                    int optLevel,
                                                    std::string ptxasExe,
                                                    std::string libDeviceFile) {
  return std::make_unique<SerializeToCubinPass>(triple, arch, features, optLevel, ptxasExe, libDeviceFile);
}

}

#else  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
namespace mlir::polygeist {
void registerGpuSerializeToCubinPass() {}
}
#endif // MLIR_GPU_TO_CUBIN_PASS_ENABLE
