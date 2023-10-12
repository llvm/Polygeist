#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

bool runBye(Function &F) { return false; }

struct CUDALaunchFixUp : PassInfoMixin<CUDALaunchFixUp> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (!runBye(F))
      return PreservedAnalyses::all();
    return PreservedAnalyses::none();
  }
};

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getByePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CUDALaunchFixUp", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerVectorizerStartEPCallback(
                [](llvm::FunctionPassManager &PM, OptimizationLevel Level) {
                  PM.addPass(CUDALaunchFixUp());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "goodbye") {
                    PM.addPass(CUDALaunchFixUp());
                    return true;
                  }
                  return false;
                });
          }};
}
