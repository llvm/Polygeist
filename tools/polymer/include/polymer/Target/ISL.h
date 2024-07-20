//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TARGET_ISL_H
#define POLYMER_TARGET_ISL_H

#include <memory>

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

struct isl_schedule;

#define __isl_give

namespace polymer {

class IslScop;
class PolymerSymbolTable;

std::unique_ptr<IslScop> createIslFromFuncOp(mlir::func::FuncOp funcOp);

/// Create a function (FuncOp) from the given OpenScop object in the given
/// module (ModuleOp).
mlir::func::FuncOp createFuncOpFromIsl(std::unique_ptr<IslScop> scop,
                                       mlir::func::FuncOp f,
                                       __isl_give isl_schedule *newSchedule);

} // namespace polymer

#endif
