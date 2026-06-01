//===- KernelLaunchLoweringUtils.h - shared kernel.launch helpers --*- C++ -*-===//
//
// Helpers shared by the kernel.launch → runtime-shim ABI lowering passes:
//   - LowerKernelLaunchToCuBLAS  (most matched library ops)
//   - LowerKernelLaunchToPVA     (int8/int16 conv2d → PVA Solutions)
//
// All three helpers are backend-agnostic — they take the target shim symbol
// (and arg types) as arguments. Per-backend passes own the libSym → shim
// symbol mapping and the top-level dispatch.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_POLYGEIST_TRANSFORMS_KERNEL_LAUNCH_LOWERING_UTILS_H
#define DIALECT_POLYGEIST_TRANSFORMS_KERNEL_LAUNCH_LOWERING_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "polygeist/Kernel/KernelOps.h"

namespace mlir {
namespace polygeist {

// Get-or-create a `func.func private @<shim>(<argTypes>)` declaration at
// module scope. Idempotent.
func::FuncOp ensureShimDecl(ModuleOp module, StringRef shimSym,
                            TypeRange argTypes, OpBuilder &builder);

// Extract a raw `!llvm.ptr` to the FIRST DATA ELEMENT of a memref:
// aligned_ptr (as index) + offset*sizeof(elt) → !llvm.ptr.
Value memrefBasePtr(OpBuilder &b, Location loc, Value m);

// Lower a kernel.launch carrying the matcher's 9-tap conv shape to a
// func.call against the supplied shim symbol. Backend-agnostic: the caller
// picks `shimSymbol` based on element type / target accelerator. Handles
// both the new 19-operand form (M, N + 9 input subviews + 1 output + 9
// weights) and the legacy 10-operand f64 form (hardcoded polybench
// weights inside the shim).
LogicalResult lowerCudnnConv2D9tap(kernel::LaunchOp launch, ModuleOp module,
                                    StringRef shimSymbol);

// Same convention as lowerCudnnConv2D9tap, but for 5x5 / 25-tap stencils.
// The launch has 25 input subviews, one output subview, then 25 scalar weights.
LogicalResult lowerCudnnConv2D25tap(kernel::LaunchOp launch, ModuleOp module,
                                     StringRef shimSymbol);

// Lower a kernel.launch carrying a "uniform-weight K×K image filter" shape
// (1 input subview + 1 output subview, no scalar weights) to a func.call
// whose signature is `(M, N, A_ptr, B_ptr)`. Used by the PVA pass for
// pvaBoxFilter-style ops where the kernel coefficients are implicit.
LogicalResult lowerImageFilter2Operand(kernel::LaunchOp launch,
                                        ModuleOp module,
                                        StringRef shimSymbol);

} // namespace polygeist
} // namespace mlir

#endif // DIALECT_POLYGEIST_TRANSFORMS_KERNEL_LAUNCH_LOWERING_UTILS_H
