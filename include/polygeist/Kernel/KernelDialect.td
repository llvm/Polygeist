//===- KernelDialect.td - Kernel dialect definition -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_DIALECT
#define KERNEL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Kernel dialect definition
//===----------------------------------------------------------------------===//

def Kernel_Dialect : Dialect {
  let name = "kernel";
  let cppNamespace = "::mlir::polygeist::kernel";
  let description = [{
    The kernel dialect provides operations for NVIDIA kernel matrix multiplication
    routines, including standard and batched GEMM operations. This dialect enables
    representation and optimization of high-performance linear algebra kernels
    within the Polygeist infrastructure.
  }];
}

//===----------------------------------------------------------------------===//
// Base class for kernel dialect operations
//===----------------------------------------------------------------------===//

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, traits>;

#endif // KERNEL_DIALECT 