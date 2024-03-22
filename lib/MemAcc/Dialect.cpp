//===- MemAccDialect.cpp - MemAcc dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemAcc/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "MemAcc/Ops.h"

using namespace mlir;
using namespace mlir::MemAcc;

//===----------------------------------------------------------------------===//
// MemAcc dialect.
//===----------------------------------------------------------------------===//

void MemAccDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MemAcc/MemAccOps.cpp.inc"
      >();
}

#include "MemAcc/MemAccOpsDialect.cpp.inc"
