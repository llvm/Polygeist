//===- KernelDialect.cpp - Kernel dialect implementation --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

#include "polygeist/Kernel/KernelOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Kernel dialect initialization
//===----------------------------------------------------------------------===//

void KernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/Kernel/KernelOps.cpp.inc"
      >();
} 