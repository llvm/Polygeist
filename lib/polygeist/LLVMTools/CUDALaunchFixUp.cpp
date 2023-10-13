#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";

void fixup(Module &M) {
  auto LaunchKernelFunc =
      dyn_cast_or_null<Function>(M.getGlobalVariable(cudaLaunchSymbolName));
  if (!LaunchKernelFunc)
    return;

  for (auto User : LaunchKernelFunc->users()) {
    if (auto CI = dyn_cast<CallInst>(User)) {
      SmallVector<Type *> ArgTypes = {
          // function ptr
          PointerType::get(M.getContext(), 0),
          // stream
          PointerType::get(M.getContext(), 0),
          // dyn shmem size
          Type::getInt64Ty(M.getContext()),
          // grid size
          Type::getInt64Ty(M.getContext()),
          Type::getInt64Ty(M.getContext()),
          Type::getInt64Ty(M.getContext()),
          // block size
          Type::getInt64Ty(M.getContext()),
          Type::getInt64Ty(M.getContext()),
          Type::getInt64Ty(M.getContext()),
      };
      auto StubFunc = cast<Function>(CI->getArgOperand(0));
      for (auto ArgTy : StubFunc->getFunctionType()->params())
        ArgTypes.push_back(ArgTy);
      auto PolygeistLaunchFunc = Function::Create(
          FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                            /*isVarAtg=*/false),
          llvm::GlobalValue::InternalLinkage,
          "__polygeist_launch_kernel_" + StubFunc->getName(), M);
    }
  }
}

struct CUDALaunchFixUp : PassInfoMixin<CUDALaunchFixUp> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    fixup(M);
    return PreservedAnalyses::none();
  }
};

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getCUDALaunchFixUpPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CUDALaunchFixUp", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager &PM, OptimizationLevel Level) {
                  PM.addPass(CUDALaunchFixUp());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::ModulePassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "cuda-launch-fixup") {
                    PM.addPass(CUDALaunchFixUp());
                    return true;
                  }
                  return false;
                });
          }};
}
