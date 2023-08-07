//===- SQLOps.td - SQL dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SQL_OPS
#define SQL_OPS

include "mlir/IR/OpBase.td"
include "SQLDialect.td"
include "SQLTypes.td"


class SQL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SQL_Dialect, mnemonic, traits>; 


def IntOp : SQL_Op<"int", [Pure]> {
  let summary = "int op";

  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def ColumnOp : SQL_Op<"column", [Pure]> {
  let summary = "column op";

  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def TableOp : SQL_Op<"table", [Pure]> {
  let summary = "table";

  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def EmptyTableOp : SQL_Op<"empty_table", [Pure]> {
  let summary = "empty_table";
  // i need to specify the size of a Variadic? 
  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

// def VectorNotEmpty : AttrConstraint<CPred<"$_self.size() > 0">,
//                                      "VectorNotEmpty">; 

                                    
def SelectOp : SQL_Op<"select", [Pure]> {
  let summary = "select";
  // i need to specify the size of a Variadic? 
  let arguments = (ins Variadic<SQLExprType>:$columns, SQLExprType:$table);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def UnparsedOp : SQL_Op<"unparsed", [Pure]> {
  let summary = "unparsed sql op";

  let arguments = (ins AnyType:$input);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 1;
} 

def ExecuteOp : SQL_Op<"execute", []> {
  let summary = "execute query";

  let arguments = (ins Index:$conn, SQLExprType:$command);
  let results = (outs Index:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}


def NumResultsOp : SQL_Op<"num_results", [Pure]> {
  let summary = "number of results";

  let arguments = (ins Index:$handle);
  let results = (outs Index:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 1;
}  

def GetValueOp : SQL_Op<"get_value", [Pure]> {
  let summary = "get value of execution";

  let arguments = (ins Index:$handle, Index:$column, Index:$row);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 1;
}  

// def SelectOp : SQL_Op<"select", [Pure]>{
//   let summary = "select";

//   let arguments = (ins StrArrayAttr:$columns, 
//                        Optional<AnyMemRef>:$from);
//                       //  optional<list<clauses>>:$where,
//                       //  optional<int>:$limit, 
//                       //  optional<int>:$order);
//   let results = (outs AnyType:$result);

//   let hasFolder = 0;
//   let hasCanonicalizer = 0;
// }
#endif // SQL_OPS