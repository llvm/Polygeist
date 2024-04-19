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
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/raw_ostream.h"

#if POLYGEIST_ENABLE_CUDA
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <cuda.h>
// TODO use this library if possible, crashes for some reason
#include <nvPTXCompiler.h>

#define DEBUG_TYPE "polygeist-serialize-to-cubin"

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

#define RETURN_ON_NVPTX_ERROR(x)                                               \
  do {                                                                         \
    nvPTXCompileResult result = x;                                             \
    if (result != NVPTXCOMPILE_SUCCESS) {                                      \
      emitError(loc, llvm::Twine("error: ")                                    \
                         .concat(#x)                                           \
                         .concat(" failed with error code ")                   \
                         .concat(std::to_string(result)));                     \
      return {};                                                               \
    }                                                                          \
  } while (0)

namespace {
class SerializeToCubinPass
    : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)

  SerializeToCubinPass(StringRef chip = "sm_35", StringRef features = "+ptx60",
                       int llvmOptLevel = 3, int ptxasOptLevel = 3,
                       std::string ptxasPath = "",
                       std::string libDevicePath = "",
                       bool outputIntermediate = false);

  StringRef getArgument() const override { return "gpu-to-cubin-polygeist"; }
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

  std::string ptxasPath;
  std::string libDevicePath;
  int llvmOptLevel;
  int ptxasOptLevel;
  bool outputIntermediate;
};
} // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option, StringRef value) {
  if (!option.hasValue())
    option = value.str();
}

SerializeToCubinPass::SerializeToCubinPass(StringRef chip, StringRef features,
                                           int llvmOptLevel, int ptxasOptLevel,
                                           std::string ptxasPath,
                                           std::string libDevicePath,
                                           bool outputIntermediate) {
  maybeSetOption(this->chip, chip);
  maybeSetOption(this->features, features);
  this->llvmOptLevel = llvmOptLevel;
  this->ptxasOptLevel = ptxasOptLevel;
  this->ptxasPath = ptxasPath;
  this->libDevicePath = libDevicePath;
  this->outputIntermediate = outputIntermediate;
}

void SerializeToCubinPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<llvm::Module>
SerializeToCubinPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  gpu::GPUModuleOp gpum = getOperation();

  mlir::ModuleOp m = gpum->getParentOfType<mlir::ModuleOp>();

  mlir::ModuleOp tmpModule(
      mlir::ModuleOp::create(mlir::OpBuilder(m->getContext()).getUnknownLoc()));
  // Prepare DL, triple attributes
  auto triple = m->getAttrOfType<StringAttr>(
      StringRef("polygeist.gpu_module." +
                LLVM::LLVMDialect::getTargetTripleAttrName().str()));
  this->triple = std::string(triple.getValue());
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
      GV.setDSOLocal(true);
      GV.setExternallyInitialized(true);
    }
  }

  // Link libdevice
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> libDevice =
      llvm::parseIRFile(libDevicePath, err, llvmContext);
  if (!libDevice || llvm::verifyModule(*libDevice, &llvm::errs())) {
    err.print("in serialize-to-cubin: Could not parse IR", llvm::errs());
    return llvmModule;
  }

  llvm::Linker::linkModules(*llvmModule, std::move(libDevice));

  // Internalize all but the public kernel function
  // (https://llvm.org/docs/NVPTXUsage.html)
  llvm::NamedMDNode *MD =
      llvmModule->getOrInsertNamedMetadata("nvvm.annotations");
  if (!MD) {
    // TODO what is the correct course of action here?
    assert(0);
  }
  llvm::internalizeModule(
      *llvmModule, [&](const llvm::GlobalValue &GV) -> bool {
        for (auto *Op : MD->operands()) {
          llvm::MDString *KindID = dyn_cast<llvm::MDString>(Op->getOperand(1));
          if (!KindID || KindID->getString() == "kernel") {
            llvm::GlobalValue *KernelFn =
                llvm::mdconst::dyn_extract_or_null<llvm::Function>(
                    Op->getOperand(0));
            if (KernelFn == &GV)
              return true;
          }
        }
        return false;
      });

  // Convert some intrinsic functions to call to libdevice
  SmallVector<llvm::IntrinsicInst *> toConvert;
  for (auto &F : *llvmModule) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto II = dyn_cast<llvm::IntrinsicInst>(&I)) {
          toConvert.push_back(II);
        }
      }
    }
  }
  llvm::for_each(toConvert, [&](llvm::IntrinsicInst *II) {
    if (II->getIntrinsicID() == llvm::Intrinsic::powi) {
      StringRef fname;
      if (II->getArgOperand(0)->getType()->isFloatTy())
        fname = "__nv_powif";
      else if (II->getArgOperand(0)->getType()->isDoubleTy())
        fname = "__nv_powi";
      else
        llvm_unreachable("unhandled float type in powi call");
      auto *CI = llvm::CallInst::Create(
          II->getFunctionType(), llvmModule->getFunction(fname),
          llvm::ArrayRef<llvm::Value *>(
              {II->getArgOperand(0), II->getArgOperand(1)}),
          fname, II);
      II->replaceAllUsesWith(CI);
      II->eraseFromParent();
    }
  });

  return llvmModule;
}

LogicalResult
SerializeToCubinPass::optimizeLlvm(llvm::Module &llvmModule,
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

std::unique_ptr<std::vector<char>>
SerializeToCubinPass::serializeISA(const std::string &isa) {
  Location loc = getOperation().getLoc();

  if (outputIntermediate) {
    llvm::outs() << "PTX module for: " << getOperation().getNameAttr() << "\n"
                 << isa << "\n";
    llvm::outs().flush();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "PTX module for: " << getOperation().getNameAttr() << "\n";
    llvm::dbgs() << isa << "\n";
    llvm::dbgs().flush();
  });

  llvm::SmallString<64> tmpInput;
  int tmpInputFD;
  llvm::SmallString<64> tmpOutput;
  int tmpOutputFD;
  llvm::sys::fs::createTemporaryFile("isainput", "s", tmpInputFD, tmpInput);
  llvm::FileRemover tmpInputRemover(tmpInput.c_str());
  llvm::sys::fs::createTemporaryFile("cubinoutput", "cubin", tmpOutputFD,
                                     tmpOutput);
  llvm::FileRemover tmpOutputRemover(tmpOutput.c_str());
  {
    llvm::raw_fd_ostream out(tmpInputFD, /*shouldClose*/ false);
    out << isa << "\n";
    out.flush();
  }

  std::vector<StringRef> Argv;
  Argv.push_back(ptxasPath.c_str());
  Argv.push_back(llvm::Triple(triple).isArch64Bit() ? "-m64" : "-m32");
  Argv.push_back("--gpu-name");
  Argv.push_back(chip.c_str());
  Argv.push_back("--opt-level");
  Argv.push_back(std::to_string(ptxasOptLevel));
  Argv.push_back("--verbose");
  Argv.push_back("--output-file");
  Argv.push_back(tmpOutput.c_str());
  Argv.push_back(tmpInput.c_str());

  llvm::sys::ExecuteAndWait(ptxasPath.c_str(), Argv);

  auto MB = llvm::MemoryBuffer::getFile(tmpOutput, false, false, false);
  if (MB.getError()) {
    llvm::errs() << loc << "MemoryBuffer getFile failed";
    return {};
  }
  auto membuf = std::move(*MB);

  size_t cubinSize = membuf->getBufferSize();
  auto result = std::make_unique<std::vector<char>>(cubinSize);
  memcpy(result->data(), membuf->getBufferStart(), cubinSize);

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

std::unique_ptr<Pass> createGpuSerializeToCubinPass(
    StringRef arch, StringRef features, int llvmOptLevel, int ptxasOptLevel,
    std::string ptxasPath, std::string libDevicePath, bool outputIntermediate) {
  return std::make_unique<SerializeToCubinPass>(
      arch, features, llvmOptLevel, ptxasOptLevel, ptxasPath, libDevicePath,
      outputIntermediate);
}

} // namespace mlir::polygeist

#else
namespace mlir::polygeist {
void registerGpuSerializeToCubinPass() {}
} // namespace mlir::polygeist
#endif
