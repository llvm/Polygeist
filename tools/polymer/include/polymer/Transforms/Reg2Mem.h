//===- Reg2Mem.cc - reg2mem transformation --------------------------------===//
//
// This file declares the registration for the reg2mem transformation pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

namespace polymer {

void registerRegToMemPass();
std::unique_ptr<mlir::Pass> createRegToMemPass();
} // namespace polymer
