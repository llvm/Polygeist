//===- WrapKernelLaunchPipeline.cpp - runtime pipeline scopes -------------===//
//
// Inserts begin/end calls around functions that contain matched kernel
// dispatches. The runtime can use this explicit scope to keep CUDA mappings,
// temporary allocations, descriptors, streams, and future device-resident
// values alive across a sequence of lowered library calls.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "KernelLaunchLoweringUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

static bool isRuntimePipelineCall(func::CallOp call, StringRef beginSymbol,
                                  StringRef endSymbol) {
  StringRef callee = call.getCallee();
  return callee == beginSymbol || callee == endSymbol;
}

static bool isCudaShimCall(func::CallOp call) {
  StringRef callee = call.getCallee();
  if (!callee.startswith("polygeist_"))
    return false;
  if (callee.startswith("polygeist_cublas_pipeline_"))
    return false;
  return callee.startswith("polygeist_cublas_") ||
         callee.startswith("polygeist_cudnn_") ||
         callee.startswith("polygeist_cuda_") ||
         callee.startswith("polygeist_rmsnorm_") ||
         callee.startswith("polygeist_whisper_");
}

static bool containsRawKernelLaunch(func::FuncOp func) {
  bool found = false;
  func.walk([&](LaunchOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool containsCudaShimCall(func::FuncOp func, StringRef beginSymbol,
                                 StringRef endSymbol) {
  bool found = false;
  func.walk([&](func::CallOp call) {
    if (isRuntimePipelineCall(call, beginSymbol, endSymbol))
      return WalkResult::advance();
    if (isCudaShimCall(call)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool alreadyWrapped(func::FuncOp func, StringRef beginSymbol,
                           StringRef endSymbol) {
  bool sawBegin = false;
  bool sawEnd = false;
  func.walk([&](func::CallOp call) {
    StringRef callee = call.getCallee();
    sawBegin |= callee == beginSymbol;
    sawEnd |= callee == endSymbol;
  });
  return sawBegin || sawEnd;
}

struct WrapKernelLaunchPipelinePass
    : public mlir::polygeist::WrapKernelLaunchPipelineBase<
          WrapKernelLaunchPipelinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder moduleBuilder(ctx);

    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp func) { funcs.push_back(func); });

    bool needsDeclarations = false;
    for (func::FuncOp func : funcs) {
      if (func.isDeclaration())
        continue;
      if (alreadyWrapped(func, beginSymbol, endSymbol))
        continue;
      if (containsCudaShimCall(func, beginSymbol, endSymbol) ||
          containsRawKernelLaunch(func)) {
        needsDeclarations = true;
        break;
      }
    }

    if (!needsDeclarations)
      return;

    ensureShimDecl(module, beginSymbol, TypeRange{}, moduleBuilder);
    ensureShimDecl(module, endSymbol, TypeRange{}, moduleBuilder);

    for (func::FuncOp func : funcs) {
      if (func.isDeclaration())
        continue;
      if (alreadyWrapped(func, beginSymbol, endSymbol))
        continue;
      if (!containsCudaShimCall(func, beginSymbol, endSymbol) &&
          !containsRawKernelLaunch(func))
        continue;

      Block &entry = func.getBody().front();
      OpBuilder entryBuilder(ctx);
      entryBuilder.setInsertionPointToStart(&entry);
      entryBuilder.create<func::CallOp>(func.getLoc(), beginSymbol,
                                        TypeRange{}, ValueRange{});

      SmallVector<func::ReturnOp> returns;
      func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
      for (func::ReturnOp ret : returns) {
        OpBuilder retBuilder(ret);
        retBuilder.create<func::CallOp>(ret.getLoc(), endSymbol, TypeRange{},
                                        ValueRange{});
      }
    }
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createWrapKernelLaunchPipelinePass() {
  return std::make_unique<WrapKernelLaunchPipelinePass>();
}
} // namespace polygeist
} // namespace mlir
