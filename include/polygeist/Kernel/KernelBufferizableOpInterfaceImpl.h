//===- KernelBufferizableOpInterfaceImpl.h ---------------------*- C++ -*-===//

#ifndef POLYGEIST_KERNEL_KERNELBUFFERIZABLEOPINTERFACEIMPL_H
#define POLYGEIST_KERNEL_KERNELBUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::polygeist::kernel {

/// Register One-Shot Bufferize semantics for kernel.launch. Tensor results
/// alias the launch operands yielded by the referenced kernel.defn.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::polygeist::kernel

#endif
