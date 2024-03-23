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
include "mlir/IR/BuiltinAttributes.td"  // This includes the definition for IntegerAttr and others

def MemAcc_GenericStoreOp : MemAcc_Op<"generic_store"> {
  let summary = "Defines a generic memory access pattern.";
  let description = [{
    The `MemAcc.generic_store` operation is designed to encapsulate any
    arbitrary memory access pattern, including stride, indirect, and other
    patterns. It takes a region of code as input, which should conclude with
    a memref store operation to signify the end of the memory access
    pattern.

    Example:

    ```mlir
    memref.generic_store {
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

def MemAcc_GenericLoadOp : MemAcc_Op<"generic_load"> {
  let summary = "Defines a generic memory access pattern.";
  let description = [{
    The `MemAcc.generic_load` operation encapsulates any arbitrary memory 
    access pattern, including stride, indirect, and other patterns. It takes 
    a region of code as input, which should conclude with a memref load 
    operation to signify the end of the memory access pattern. The type of 
    the last load operation in the region determines the result type of the 
    `MemAcc.generic_load` operation. Additionally, it supports an attribute 
    `indirection-level` to specify the level of indirection used in the 
    memory access pattern.

    Example:

    ```mlir
    %result = memacc.generic_load indirection-level = 2 {
      %loaded_value = memref.load %addr[%offset] : memref<...>
      memacc.yield %loaded_value : type_of(<...>)
    } : type_of(<...>)
    ```
  }];

  let regions = (region AnyRegion:$body);
  let arguments = (ins OptionalAttr<I64Attr>:$indirectionLevel);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `region` $body `indirection_level` `=` $indirectionLevel attr-dict `:` functional-type(operands, results)
  }];

    // Correct way to add an optional integer attribute

  // let attributes = (ins Optional<I64Attr>:$indirectionLevel);
}


//===----------------------------------------------------------------------===//
// MemAcc_YieldOp
//===----------------------------------------------------------------------===//

def MemAcc_YieldOp : MemAcc_Op<"yield", [Terminator, Pure]> {
  let summary = "MemAcc dialect yield operation";
  let description = [{
    The `MemAcc.yield` operation serves as a terminator for blocks within MemAcc
    operations, indicating the end of a block's execution path. It can optionally
    return a memref to the enclosing operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$operands attr-dict `:` functional-type(operands, results)";
}

#endif // MemAcc_OPS