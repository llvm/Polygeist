//===- KernelDialect.h - Kernel dialect declaration -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_KERNEL_KERNELDIALECT_H
#define POLYGEIST_KERNEL_KERNELDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace polygeist {
namespace kernel {

} // namespace kernel
} // namespace polygeist
} // namespace mlir

#include "polygeist/Kernel/KernelOpsDialect.h.inc"

#endif // POLYGEIST_KERNEL_KERNELDIALECT_H 