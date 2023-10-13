#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"

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
          // grid size
          Type::getInt32Ty(M.getContext()),
          Type::getInt32Ty(M.getContext()),
          Type::getInt32Ty(M.getContext()),
          // block siz3
          Type::getInt32Ty(M.getContext()),
          Type::getInt32Ty(M.getContext()),
          Type::getInt32Ty(M.getContext()),
          // dyn shmem size
          Type::getInt32Ty(M.getContext()),
          // stream
          PointerType::get(M.getContext(), 0),
      };
      auto StubFunc = cast<Function>(CI->getArgOperand(0));
      for (auto ArgTy : StubFunc->getFunctionType()->params())
        ArgTypes.push_back(ArgTy);
      auto PolygeistLaunchFunc = Function::Create(
          FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                            /*isVarAtg=*/false),
          llvm::GlobalValue::InternalLinkage,
          "__polygeist_launch_kernel_" + StubFunc->getName(), M);

      IRBuilder<> Builder(CI);
      auto FuncPtr = CI->getArgOperand(0);
      auto GridDim1 = CI->getArgOperand(1);
      auto GridDim2 = CI->getArgOperand(2);
      auto GridDimX = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
      auto GridDimY = Builder.CreateLShr(
          GridDim1, ConstantInt::get(Builder.getInt32Ty(), 32));
      GridDimY = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
      auto GridDimZ = GridDim2;
      auto BlockDim1 = CI->getArgOperand(3);
      auto BlockDim2 = CI->getArgOperand(4);
      auto BlockDimX = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
      auto BlockDimY = Builder.CreateLShr(
          BlockDim1, ConstantInt::get(Builder.getInt32Ty(), 32));
      BlockDimY = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
      auto BlockDimZ = BlockDim2;
      auto SharedMemSize = CI->getArgOperand(6);
      auto StreamPtr = CI->getArgOperand(7);
      SmallVector<Value *> Args = {
          FuncPtr,   GridDimX,  GridDimY,      GridDimZ,  BlockDimX,
          BlockDimY, BlockDimZ, SharedMemSize, StreamPtr,
      };
      Builder.CreateCall(PolygeistLaunchFunc, Args);
      CI->eraseFromParent();
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
