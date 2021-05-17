//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// This file declares some generic utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_UTILS_H
#define POLYMER_SUPPORT_UTILS_H

#include "llvm/ADT/SetVector.h"

namespace mlir {
class Block;
class Value;
} // namespace mlir

namespace polymer {

/// Find all the values that should be the arguments of the given block if it is
/// not nested in the current scope.
void inferBlockArgs(mlir::Block *block, llvm::SetVector<mlir::Value> &args);

} // namespace polymer

#endif
