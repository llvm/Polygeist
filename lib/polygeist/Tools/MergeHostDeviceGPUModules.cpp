#include <polygeist/Tools/MergeHostDeviceGPUModules.h>

using namespace mlir;

LogicalResult polygeist::mergeDeviceIntoHost(ModuleOp hostModule,
                                             ModuleOp deviceModule) {
  if (hostModule->walk([](gpu::GPUModuleOp) {}).wasInterrupted()) {
    return failure();
  }

  auto moduleBuilder = OpBuilder::atBlockBegin(hostModule.getBody());

  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>();

  moduleBuilder.inlineBlockBefore(deviceModule.getBody(),
                                  deviceModule.getBody());

  return success();
}
