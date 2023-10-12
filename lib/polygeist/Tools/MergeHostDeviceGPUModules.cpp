#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <polygeist/Tools/MergeHostDeviceGPUModules.h>

using namespace mlir;

namespace {
constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";
constexpr char gpuModuleName[] = "polygeist_gpu_module";
} // namespace

LogicalResult mlir::polygeist::mergeDeviceIntoHost(ModuleOp hostModule,
                                                   ModuleOp deviceModule) {
  if (hostModule->walk([](gpu::GPUModuleOp) { return WalkResult::interrupt(); })
          .wasInterrupted()) {
    return failure();
  }

  auto ctx = hostModule.getContext();

  auto moduleBuilder = OpBuilder::atBlockBegin(hostModule.getBody());
  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(
      deviceModule->getLoc(), gpuModuleName);
  gpuModule.getRegion().takeBody(deviceModule.getRegion());

  auto gpuModuleBuilder = OpBuilder::atBlockEnd(gpuModule.getBody());
  gpuModuleBuilder.create<gpu::ModuleEndOp>(gpuModule->getLoc());

  auto launchFunc =
      dyn_cast<LLVM::LLVMFuncOp>(hostModule.lookupSymbol(cudaLaunchSymbolName));
  if (!launchFunc) {
    return success();
  }
  auto launchFuncUses = launchFunc.getSymbolUses(hostModule);
  for (auto use : *launchFuncUses) {
    if (auto callOp = dyn_cast<LLVM::CallOp>(use.getUser())) {
      auto loc = callOp->getLoc();
      StringRef callee = cast<LLVM::AddressOfOp>(
                             callOp.getCalleeOperands().front().getDefiningOp())
                             .getGlobalName();
      if (callee.consume_front("_Z")) {
        int tmp;
        callee.consumeInteger(/*radix=*/10, tmp);
      }
      callee.consume_front("__device_stub");

      // LLVM::LLVMFuncOp gpuFuncOp =
      // cast<LLVM::LLVMFuncOp>(deviceModule.lookupSymbol(callee));
      SymbolRefAttr gpuFuncSymbol = SymbolRefAttr::get(
          StringAttr::get(ctx, gpuModuleName),
          {SymbolRefAttr::get(StringAttr::get(ctx, callee))});
      OpBuilder builder(callOp);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      llvm::SmallVector<Value> args = {one};
      Value dynShMemSize = callOp.getArgOperands()[6];
      builder.create<gpu::LaunchFuncOp>(
          loc, gpuFuncSymbol, gpu::KernelDim3({one, one, one}),
          gpu::KernelDim3({one, one, one}), dynShMemSize,
          ValueRange(args)); // , /*asyncObject=*/nullptr); //,
                             // /*asyncDependencies=*/{});
      callOp->erase();
    }
  }

  return success();
}
