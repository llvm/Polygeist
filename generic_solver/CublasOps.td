//===- KernelOps.td - kernel dialect operation definitions ---*- tablegen -*-===//
//
// This file defines the kernel operation definitions in TableGen format.
//
//===----------------------------------------------------------------------===//

#ifndef kernel_OPS
#define kernel_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"

//===----------------------------------------------------------------------===//
// kernel dialect definition
//===----------------------------------------------------------------------===//

def Kernel_Dialect : Dialect {
  let name = "kernel";
  let cppNamespace = "::mlir::kernel";
  let description = [{
    The kernel dialect provides operations for NVIDIA kernel matrix multiplication
    routines, including standard and batched GEMM operations.
  }];
}

//===----------------------------------------------------------------------===//
// Base class for kernel dialect operations
//===----------------------------------------------------------------------===//

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// kernel ops instantiation collection
//===----------------------------------------------------------------------===//

def Opinst_DefnCollection : Op<Kernel_Dialect, "defn_collection", []> {
  let summary = "Collection of operation definitions";
  let description = [{
    A collection of operation definitions that can be referenced elsewhere.
    This operation serves as a container for multiple operation definitions.
  }];
  
  let regions = (region SizedRegion<1>:$defns);
  
  let assemblyFormat = [{
    $defns attr-dict
  }];
}

def Opinst_Defn : Op<Kernel_Dialect, "defn", []> {
  let summary = "Definition of an operation";
  let description = [{
    A definition of an operation with inputs and arbitrary body code.
    Can contain either literal code or a linalg.generic representation.
  }];
  
  let arguments = (ins
    StrAttr:$name,
    Variadic<AnyType>:$inputs
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $name `(` $inputs `)` $body attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// Example pattern representation
//===----------------------------------------------------------------------===//

// Patterns for gemm and batched_gemm expressed in a mathematical notation.
// These are informational and would be used by pattern matchers.

// Standard GEMM pattern: C(i,k) += alpha * A(i,j) * B(j,k)
// Batched GEMM pattern: C(N, i,k) += alpha * A(N, i,j) * B(N, j,k)

// Index of max absolute value pattern: result = argmax_i |x_i|
// Index of min absolute value pattern: result = argmin_i |x_i|
// Sum of absolute values pattern: result = sum_i |x_i|

#endif // kernel_OPS 