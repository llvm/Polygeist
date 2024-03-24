//===- MemAccDialect.td - MemAcc dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MemAcc_DIALECT
#define MemAcc_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MemAcc dialect definition.
//===----------------------------------------------------------------------===//

def MemAcc_Dialect : Dialect {
    let name = "memacc";
    let summary = "A MemAcc out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::MemAcc";
}

//===----------------------------------------------------------------------===//
// Base MemAcc operation definition.
//===----------------------------------------------------------------------===//

class MemAcc_Op<string mnemonic, list<Trait> traits = []> :
        Op<MemAcc_Dialect, mnemonic, traits>;

#endif // MemAcc_DIALECT