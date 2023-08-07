//===- Parser.h - SQL dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//




#ifndef SQLPARSER_H
#define SQLPARSER_H

#include "mlir/IR/Dialect.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"

#include "sql/SQLDialect.h"
#include "sql/SQLOps.h"
#include "sql/SQLTypes.h"

mlir::Value parseSQL(mlir::Location loc, mlir::OpBuilder& builder, std::string str);

#endif // SQLPARSER_H