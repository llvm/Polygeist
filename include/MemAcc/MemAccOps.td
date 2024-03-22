//===- MemAccOps.td - MemAcc dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MemAcc_OPS
#define MemAcc_OPS

include "MemAccDialect.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MemAcc_GenericAccessOp : MemAcc_Op<"generic_access"> {
  let summary = "Defines a generic memory access pattern.";
  let description = [{
    The `MemAcc.generic_access` operation is designed to encapsulate any
    arbitrary memory access pattern, including stride, indirect, and other
    patterns. It takes a region of code as input, which should conclude with
    a memref load or store operation to signify the end of the memory access
    pattern.

    Example:

    ```mlir
    memref.generic_access {
      // Define memory access pattern here
      %0 = memref.load %addr[%offset] : memref<...>
    } : <type>
    ```
  }];

  let regions = (region AnyRegion:$body);
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{
    `region` $body attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MemAcc_OPS