//===- KernelOps.td - Kernel dialect operation definitions -*-- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_OPS
#define KERNEL_OPS

include "polygeist/Kernel/KernelDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Kernel operation definitions
//===----------------------------------------------------------------------===//

def Kernel_DefnCollectionOp : Kernel_Op<"defn_collection", [NoTerminator]> {
  let summary = "Collection of kernel operation definitions";
  let description = [{
    A collection of operation definitions that can be referenced elsewhere.
    This operation serves as a container for multiple kernel operation definitions,
    enabling modular organization of kernel implementations.
  }];
  
  let regions = (region SizedRegion<1>:$defns);
  
  let assemblyFormat = [{
    $defns attr-dict
  }];
}

def Kernel_DefnOp : Kernel_Op<"defn", [
  AffineScope,
  AutomaticAllocationScope,
  IsolatedFromAbove, 
  FunctionOpInterface, 
  Symbol
]> {
  let summary = "Definition of a kernel operation";
  let description = [{
    A definition of a kernel operation with inputs and arbitrary body code.
    Can contain either literal CUDA/HIP code or a linalg.generic representation
    for high-performance linear algebra operations.
    
    This operation is particularly useful for defining custom GEMM variants,
    batched operations, and other specialized linear algebra kernels.
    
    Example:
    ```mlir
    kernel.defn @custom_gemm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, 
                             %C: memref<?x?xf32>, %alpha: f32) -> tensor<f32> {
      // Kernel implementation
      kernel.yield %some_result : tensor<f32>
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  
  let regions = (region AnyRegion:$body);
  
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  
  let hasCustomAssemblyFormat = 1;
  
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the argument types of this kernel.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this kernel.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return getBody().empty(); }
  }];
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

def Kernel_LaunchOp : Kernel_Op<"launch",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "kernel launch operation";
  let description = [{
    The `kernel.launch` operation represents a launch of a kernel that is
    within the same symbol scope as the launch. The operands and result types of
    the launch must match the specified kernel type. The kernel is encoded as a
    symbol reference attribute named "kernel".

    Example:

    ```mlir
    %result = kernel.launch @custom_gemm(%A, %B, %C, %alpha) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, f32) -> tensor<f32>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$kernel, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "DefnOp":$kernel, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("kernel", SymbolRefAttr::get(kernel));
      $_state.addTypes(kernel.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$kernel, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("kernel", kernel);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$kernel, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(kernel), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$kernel, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), kernel),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getKernelType();

    /// Get the argument operands to the launched kernel.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the kernel of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("kernel");
    }

    /// Set the kernel for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("kernel", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $kernel `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Kernel_YieldOp : Kernel_Op<"yield", [Pure, HasParent<"DefnOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Terminator for kernel.defn operation";
  let description = [{
    The `kernel.yield` operation terminates regions within kernel operations.
    It optionally returns values from the kernel definition.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, std::nullopt);
    }]>
  ];

  let hasVerifier = 1;
}

#endif // KERNEL_OPS 