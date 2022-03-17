//===- Passes.h - Include Tblgen pass defs ------------C++-===//
#ifndef POLYMER_TRANSFORMS_PASSES_H
#define POLYMER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace polymer {

std::unique_ptr<mlir::Pass> createAnnotateScopPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "polymer/Transforms/Passes.h.inc"

} // namespace polymer

#endif
