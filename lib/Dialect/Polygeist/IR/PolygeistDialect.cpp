//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/IR/PolygeistDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.cpp.inc"
      >();
}

#include "mlir/Dialect/Polygeist/IR/PolygeistOpsDialect.cpp.inc"
