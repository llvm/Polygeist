//===- EmitOpenScop.h -------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_EMITOPENSCOP_H
#define POLYMER_EMITOPENSCOP_H

#include "mlir/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace polymer {

mlir::LogicalResult emitOpenScop(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerOpenScopEmitterTranslation();

} // namespace polymer

#endif
