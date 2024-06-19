//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TARGET_OPENSCOP_H
#define POLYMER_TARGET_OPENSCOP_H

#include <memory>

#include "pluto/internal/pluto.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
template <typename OpTy> class OwningOpRef;
class MLIRContext;
class ModuleOp;
namespace func {
class FuncOp;
}
struct LogicalResult;
class Operation;
class Value;
} // namespace mlir

namespace polymer {

class OslScop;
class PolymerSymbolTable;

std::unique_ptr<OslScop> createOpenScopFromFuncOp(mlir::func::FuncOp funcOp,
                                                  PolymerSymbolTable &symTable);

/// Create a function (FuncOp) from the given OpenScop object in the given
/// module (ModuleOp).
mlir::Operation *
createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, mlir::ModuleOp module,
                         PolymerSymbolTable &symTable,
                         mlir::MLIRContext *context, PlutoProg *prog = nullptr,
                         const char *dumpClastAfterPluto = nullptr);

mlir::OwningOpRef<mlir::ModuleOp>
translateOpenScopToModule(std::unique_ptr<OslScop> scop,
                          mlir::MLIRContext *context);

mlir::LogicalResult translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os);

void registerToOpenScopTranslation();
void registerFromOpenScopTranslation();

} // namespace polymer

#endif
