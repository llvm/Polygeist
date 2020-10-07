//===- InvariantScopTransform.h - Invariant transform to OpenScop ---------===//
//
// This file declares the transformation between MLIR and OpenScop.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace polymer {

void registerInvariantScopTransformPass();

} // namespace polymer
