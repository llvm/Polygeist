//===- SQLTypes.cpp - SQL dialect types ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sql/SQLDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "sql/SQLTypes.h"


#define DEBUG_TYPE "sql"

using namespace mlir::sql;

#define GET_TYPEDEF_CLASSES
#include "sql/SQLOpsTypes.cpp.inc"


void SQLDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "sql/SQLOpsTypes.cpp.inc"
    >();
}