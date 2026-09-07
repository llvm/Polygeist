//===- KernelLibraryConstantPropagation.h ----------------------*- C++ -*-===//

#ifndef POLYGEIST_KERNEL_LIBRARY_CONSTANT_PROPAGATION_H
#define POLYGEIST_KERNEL_LIBRARY_CONSTANT_PROPAGATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "polygeist/Kernel/KernelOps.h"

#include <optional>

namespace mlir::polygeist {

/// Attribute containing a dictionary whose `operand_N` entries describe
/// tensor operands that are known to contain one uniform scalar value.
inline constexpr llvm::StringLiteral kUniformOperandConstantsAttr =
    "polygeist.uniform_operand_constants";

/// Infer program-level uniform tensor constants and annotate kernel.launch
/// consumers.  The analysis is independent of the eventual CPU/GPU runtime.
void propagateKernelLibraryConstants(ModuleOp module);

/// Return the inferred uniform scalar value for one launch operand.
std::optional<TypedAttr>
getUniformOperandConstant(kernel::LaunchOp launch, unsigned operandNumber);

} // namespace mlir::polygeist

#endif
