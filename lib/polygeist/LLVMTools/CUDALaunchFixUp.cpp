#include "llvm/ADT/STLExtras.h"
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

// TODO Force inline all kernel stubs and delete their bodies
//
namespace {

constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";
constexpr char kernelPrefix[] = "__polygeist_launch_kernel_";

void fixup(Module &M) {
  auto LaunchKernelFunc = M.getFunction(cudaLaunchSymbolName);
  if (!LaunchKernelFunc)
    return;

  SmallVector<CallInst *> ToHandle;
  for (auto User : LaunchKernelFunc->users()) {
    if (auto CI = dyn_cast<CallInst>(User)) {
      ToHandle.push_back(CI);
    }
  }
  for (CallInst *CI : ToHandle) {
    IRBuilder<> Builder(CI);
    auto FuncPtr = CI->getArgOperand(0);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto GridDimX = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
    auto GridDimY = Builder.CreateLShr(
        GridDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    GridDimY = Builder.CreateTrunc(GridDimY, Builder.getInt32Ty());
    auto GridDimZ = GridDim2;
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto BlockDimX = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
    auto BlockDimY = Builder.CreateLShr(
        BlockDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    BlockDimY = Builder.CreateTrunc(BlockDimY, Builder.getInt32Ty());
    auto BlockDimZ = BlockDim2;
    auto SharedMemSize = CI->getArgOperand(6);
    auto StreamPtr = CI->getArgOperand(7);
    SmallVector<Value *> Args = {
        FuncPtr,   GridDimX,  GridDimY,      GridDimZ,  BlockDimX,
        BlockDimY, BlockDimZ, SharedMemSize, StreamPtr,
    };
    auto StubFunc = cast<Function>(CI->getArgOperand(0));
    for (auto &Arg : StubFunc->args())
      Args.push_back(&Arg);
    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    auto PolygeistLaunchFunc = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                          /*isVarAtg=*/false),
        llvm::GlobalValue::ExternalLinkage, kernelPrefix + StubFunc->getName(),
        M);

    Builder.CreateCall(PolygeistLaunchFunc, Args);
    CI->eraseFromParent();
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

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getCUDALaunchFixUpPluginInfo();
}
