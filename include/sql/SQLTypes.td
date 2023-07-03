//===- SQLTypes.td - SQL dialect types ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SQL_TYPES
#define SQL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "SQLDialect.td"

def SQLExprType : SQL_Type<"Expr", "expr"> {
  let summary = "SQL expression type";
}


#endif // SQL_TYPES