//===- SQLTypes.h - SQL dialect types --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SQL_SQLTYPES_H
#define SQL_SQLTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "sql/SQLOpsTypes.h.inc"


#endif // SQL_SQLTYPES_H