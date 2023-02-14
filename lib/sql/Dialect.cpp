//===- SQLDialect.cpp - SQL dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "sql/SQLDialect.h"
#include "sql/SQLOps.h"

using namespace mlir;
using namespace mlir::sql;

//===----------------------------------------------------------------------===//
// SQL dialect.
//===----------------------------------------------------------------------===//

void SQLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sql/SQLOps.cpp.inc"
      >();
}

#include "sql/SQLOpsDialect.cpp.inc"
