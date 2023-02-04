//===- SQLDialect.td - SQL dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SQL_DIALECT
#define SQL_DIALECT


include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"


def SQL_Dialect : Dialect {
  let summary = "A dialect for SQL languages in MLIR.";
  let description = [{
    TBD
  }];
  let name = "sql";
  let cppNamespace = "::mlir::sql";
}


//===----------------------------------------------------------------------===//
// SQL Operations
//===----------------------------------------------------------------------===//

class SQL_Op<string mnemonic, list<Trait> traits = []>
    : Op<SQL_Dialect, mnemonic, traits>; 

#endif // SQL_DIALECT 





