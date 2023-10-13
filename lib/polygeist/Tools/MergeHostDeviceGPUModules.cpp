#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include <cstring>
#include <polygeist/Tools/MergeHostDeviceGPUModules.h>

using namespace mlir;

namespace {
constexpr char gpuModuleName[] = "__polygeist_gpu_module";
constexpr char kernelPrefix[] = "__polygeist_launch_kernel_";
} // namespace

LogicalResult mlir::polygeist::mergeDeviceIntoHost(ModuleOp hostModule,
                                                   ModuleOp deviceModule) {
  if (hostModule->walk([](gpu::GPUModuleOp) { return WalkResult::interrupt(); })
          .wasInterrupted()) {
    return failure();
  }
  llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
  hostModule->walk([&](LLVM::LLVMFuncOp funcOp) {
    auto symName = funcOp.getName();
    if (symName.startswith(kernelPrefix))
      launchFuncs.push_back(funcOp);
  });

  auto ctx = hostModule.getContext();

  auto moduleBuilder = OpBuilder::atBlockBegin(hostModule.getBody());
  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(
      deviceModule->getLoc(), gpuModuleName);
  gpuModule.getRegion().takeBody(deviceModule.getRegion());

  auto gpuModuleBuilder = OpBuilder::atBlockEnd(gpuModule.getBody());
  gpuModuleBuilder.create<gpu::ModuleEndOp>(gpuModule->getLoc());

  for (auto launchFunc : launchFuncs) {
    auto launchFuncUses = launchFunc.getSymbolUses(hostModule);
    for (auto use : *launchFuncUses) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(use.getUser())) {
        auto loc = callOp->getLoc();
        StringRef callee =
            cast<LLVM::AddressOfOp>(
                callOp.getCalleeOperands().front().getDefiningOp())
                .getGlobalName();
        int symbolLength = 0;
        if (callee.consume_front("_Z"))
          callee.consumeInteger(/*radix=*/10, symbolLength);
        const char stubPrefix[] = "__device_stub__";
        callee.consume_front(stubPrefix);

        // LLVM::LLVMFuncOp gpuFuncOp =
        // cast<LLVM::LLVMFuncOp>(deviceModule.lookupSymbol(callee));
        Twine deviceSymbol =
            "_Z" + std::to_string(symbolLength - strlen(stubPrefix)) + callee;
        SymbolRefAttr gpuFuncSymbol = SymbolRefAttr::get(
            StringAttr::get(ctx, gpuModuleName),
            {SymbolRefAttr::get(StringAttr::get(ctx, deviceSymbol.str()))});
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
