//===- SQLOps.td - SQL dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SQL_OPS
#define SQL_OPS

include "mlir/IR/AttrTypeBase.td"
include "SQLDialect.td"

def SelectOp : SQL_Op<"select", [Pure]> {
  let summary = "select";

  // TODO: limit (optional), where clauses, join, etc
  let arguments = (ins StrArrayAttr:$column, StrAttr:$table);
  let results = (outs Index : $result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def ExecuteOp : SQL_Op<"execute", []> {
  let summary = "execute query";

  let arguments = (ins Index:$handle);
  let results = (outs Index:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def NumResultsOp : SQL_Op<"num_results", [Pure]> {
  let summary = "number of results";

  let arguments = (ins Index:$handle);
  let results = (outs Index:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}  

def ResultOp : SQL_Op<"get_result", [Pure]> {
  let summary = "get results of execution";

  let arguments = (ins Index:$handle, StrAttr:$column, Index:$row);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}  
 
#endif // SQL_OPS