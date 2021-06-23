//===- BFVOps.td - BFV dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_OPS
#define POLYGEIST_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"


// HasParent<"ParallelOp">,
def BarrierOp : Polygeist_Op<"barrier",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "barrier for parallel loops";
  let description = [{}];
}

//===----------------------------------------------------------------------===//
// SubIndexOp
//===----------------------------------------------------------------------===//

def SubIndexOp : Polygeist_Op<
    "subindex", [DeclareOpInterfaceMethods<ViewLikeOpInterface>, NoSideEffect] >  {
  let summary = "memref subview operation";

  let arguments = (ins
    AnyMemRef:$source,
    Index:$index
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 0;
}

#endif // POLYGEIST_OPS
