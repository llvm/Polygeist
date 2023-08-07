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

class SQL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SQL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}



def SQLExprType : SQL_Type<"Expr", "expr"> {
  let summary = "SQL expression type";
  let description = "Custom attr or value type in sql dialect";

  // placeholder params 
  // let parameters = (ins StringRefParameter<"the custom value">:$value);
  // let assemblyFormat = "`<` $value `>`";
}

#endif // SQL_TYPES