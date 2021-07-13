/// Utility functions shared among mlir-clang library sources.

#ifndef MLIR_TOOLS_MLIRCLANG_UTILS_H
#define MLIR_TOOLS_MLIRCLANG_UTILS_H

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;
class FuncOp;
class Value;
class OpBuilder;
class AbstractOperation;
class Type;
} // namespace mlir

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlirclang {

/// Replace the given function by the operation with the given name, and use the
/// same argument list. For example, if the function is @foo(%a, %b) and opName
/// is "bar.baz", we will create an operator baz of the bar dialect, with
/// operands %a and %b. The new op will be inserted at where the insertion point
/// of the provided OpBuilder is.
mlir::Operation *replaceFuncByOperation(mlir::FuncOp f, llvm::StringRef opName,
                                        llvm::ArrayRef<mlir::Value> operands,
                                        mlir::OpBuilder &b);
mlir::Operation *buildLinalgOp(const mlir::AbstractOperation *op,
                               llvm::ArrayRef<mlir::Value> operands,
                               llvm::ArrayRef<mlir::Type> results,
                               mlir::OpBuilder &b);

} // namespace mlirclang

#endif
