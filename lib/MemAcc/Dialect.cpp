//===- MemAccDialect.cpp - MemAcc dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemAcc/Dialect.h"
#include "MemAcc/Ops.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::MemAcc;

#include "MemAcc/MemAccOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MemAcc dialect.
//===----------------------------------------------------------------------===//

void MemAccDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MemAcc/MemAccOps.cpp.inc"
      >();
}


