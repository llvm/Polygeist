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

def WhereOp: SQL_Op<"where", [Pure]> {
  let summary = "where op";

  let arguments = (ins SQLBoolType:$expr);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
} 

def BoolArithOp: SQL_Op<"bool_arith", [Pure]> {
  let summary = "bool_arith op";

  let arguments = (ins SQLBoolType:$left, SQLBoolType:$right, StrAttr:$op);
  let results = (outs SQLBoolType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def CalcBoolOp: SQL_Op<"calc_bool", [Pure]> {
  let summary = "calc_bool op";

  let arguments = (ins StrAttr:$expr);
  let results = (outs SQLBoolType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}




def ArithOp: SQL_Op<"arith", [Pure]> {
  let summary = "arith op";

  let arguments = (ins SQLExprType:$left, SQLExprType:$right, StrAttr:$op);
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def AndOp: SQL_Op<"and", [Pure]> {
  let summary = "and op";

  let arguments = (ins SQLBoolType:$left, SQLBoolType:$right);
  let results = (outs SQLBoolType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def OrOp: SQL_Op<"or", [Pure]> {
  let summary = "or op";

  let arguments = (ins SQLBoolType:$left, SQLBoolType:$right);
  let results = (outs SQLBoolType:$result);

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


def SQLConstantStringOp : SQL_Op<"str_constant", [Pure]> {
  let summary = "str_constant";

  let arguments = (ins StrAttr:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def SQLToStringOp : SQL_Op<"to_string", [Pure]> {
  let summary = "to_string";

  let arguments = (ins SQLExprType:$expr);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}    


def SQLBoolToStringOp : SQL_Op<"bool_to_string", [Pure]> {
  let summary = "bool_to_string";

  let arguments = (ins SQLBoolType:$expr);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}    


def SQLStringConcatOp : SQL_Op<"string_concat", [Pure]> {
  let summary = "string_concat";

  let arguments = (ins Variadic<AnyType>:$expr);
  let results = (outs AnyType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 1;
}                       

def ConstantBoolOp : SQL_Op <"constant_bool", [Pure]> {
  let summary = "constant_bool";
  let results = (outs SQLBoolType:$result);
}


def SelectOp : SQL_Op<"select", [Pure]> {
  let summary = "select";
  // i need to specify the size of a Variadic? 
  let arguments = (ins Variadic<SQLExprType>:$columns,
                    SQLExprType:$table,
                    SQLExprType:$where,
                    SI64Attr:$limit); 
                    // attribute limit<int> if >= 0 then its the real thing, otherwise its infinity
  let results = (outs SQLExprType:$result);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
}

def AllColumnsOp : SQL_Op<"all_columns", [Pure]> {
  let summary = "all_columns";
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