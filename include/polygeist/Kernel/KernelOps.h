//===- KernelOps.h - Kernel dialect operations ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_KERNEL_KERNELOPS_H
#define POLYGEIST_KERNEL_KERNELOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "polygeist/Kernel/KernelDialect.h"

namespace mlir {
namespace polygeist {
namespace kernel {

} // namespace kernel
} // namespace polygeist
} // namespace mlir

#define GET_OP_CLASSES
#include "polygeist/Kernel/KernelOps.h.inc"

#endif // POLYGEIST_KERNEL_KERNELOPS_H 