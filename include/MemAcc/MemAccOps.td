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
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
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
  let results = (outs Variadic<AnyType>:$result);

  let assemblyFormat = [{
    `region` $body `indirection_level` `=` $indirectionLevel attr-dict `:` functional-type(operands, results)
  }];

    // Correct way to add an optional integer attribute

  // let attributes = (ins Optional<I64Attr>:$indirectionLevel);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def MemAcc_LoadOp : MemAcc_Op<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).

    In an `affine.if` or `affine.for` body, the indices of a load are restricted
    to SSA values bound to surrounding loop induction variables,
    [symbols](Affine.md/#dimensions-and-symbols), results of a
    constant operations, or the result of an
    `affine.apply` operation that can in turn take as arguments all of the
    aforementioned SSA values or the recursively result of such an
    `affine.apply` operation.

    Example:

    ```mlir
    %1 = affine.apply affine_map<(d0, d1) -> (3*d0)> (%i, %j)
    %2 = affine.apply affine_map<(d0, d1) -> (d1+1)> (%i, %j)
    %12 = memacc.load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

    // Example of an indirect load (treated as non-affine)
    %3 = affine.apply affine_map<(d0) -> (2*d0 + 1)>(%12)
    %13 = memref.load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in affine
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def MemAcc_StoreOp : MemAcc_Op<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.

    In an affine context, the indices of a store are restricted to SSA values
    bound to surrounding loop induction variables,
    [symbols](Affine.md/#restrictions-on-dimensions-and-symbols), results of a
    `constant` operation, or the result of an
    [`affine.apply`](Affine.md/#affineapply-affineapplyop) operation that can in
    turn take as arguments all of the aforementioned SSA values or the
    recursively result of such an `affine.apply` operation.

    Example:

    ```mlir
    memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in polyhedral
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return ::llvm::cast<MemRefType>(getMemRef().getType());
      }
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
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

// Base class for integer and floating point arithmetic ops. All ops have one
// result, require operands and results to be of the same type, and can accept
// tensors or vectors of integers or floats.
class MemAcc_ArithOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_Op<mnemonic, traits # [SameOperandsAndResultType, NoMemoryEffect] #
    ElementwiseMappable.traits>;

// Base class for unary arithmetic operations.
class MemAcc_UnaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_ArithOp<mnemonic, traits # [Pure]> {
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary arithmetic operations.
class MemAcc_BinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_ArithOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for integer binary operations.
class MemAcc_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_BinaryOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>,
    Results<(outs SignlessIntegerLike:$result)>;

// Base class for integer binary operations without undefined behavior.
class MemAcc_TotalIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_IntBinaryOp<mnemonic, traits # [Pure]>;

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def MemAcc_AddIOp : MemAcc_TotalIntBinaryOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Performs N-bit addition on the operands. The operands are interpreted as 
    unsigned bitvectors. The result is represented by a bitvector containing the 
    mathematical value of the addition modulo 2^n, where `n` is the bitwidth. 
    Because `arith` integers use a two's complement representation, this operation 
    is applicable on both signed and unsigned integer operands.

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type, 
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar addition.
    %a = arith.addi %b, %c : i64

    // Scalar addition with overflow flags.
    %a = arith.addi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise addition.
    %f = arith.addi %g, %h : vector<4xi32>

    // Tensor element-wise addition.
    %x = arith.addi %y, %z : tensor<4x?xi8>
    ```
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def MemAcc_SubIOp : MemAcc_TotalIntBinaryOp<"subi"> {
  let summary = [{
    Integer subtraction operation.
  }];
  let description = [{
    Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
    bitvectors. The result is represented by a bitvector containing the mathematical
    value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
    integers use a two's complement representation, this operation is applicable on
    both signed and unsigned integer operands.

    The `subi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = arith.subi %b, %c : i64

    // Scalar subtraction with overflow flags.
    %a = arith.subi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise subtraction.
    %f = arith.subi %g, %h : vector<4xi32>

    // Tensor element-wise subtraction.
    %x = arith.subi %y, %z : tensor<4x?xi8>
    ```
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def MemAcc_MulIOp : MemAcc_TotalIntBinaryOp<"muli", [Commutative]> {
  let summary = [{
    Integer multiplication operation.
  }];
  let description = [{
    Performs N-bit multiplication on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation is
    applicable on both signed and unsigned integer operands.

    The `muli` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = arith.muli %b, %c : i64

    // Scalar multiplication with overflow flags.
    %a = arith.muli %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise multiplication.
    %f = arith.muli %g, %h : vector<4xi32>

    // Tensor element-wise multiplication.
    %x = arith.muli %y, %z : tensor<4x?xi8>
    ```
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast can convert between memrefs of signless integers and indices too.
def IndexCastTypeConstraint : TypeConstraint<Or<[
        SignlessIntegerLike.predicate,
        MemRefOf<[AnySignlessInteger, Index]>.predicate]>,
    "signless-integer-like or memref of signless-integer">;

// Base class for arithmetic cast operations. Requires a single operand and
// result. If either is a shaped type, then the other must be of the same shape.
class MemAcc_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<Trait> traits = []> :
    MemAcc_Op<mnemonic, traits # [Pure, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def MemAcc_IndexCastOp
  : MemAcc_CastOp<"index_cast", IndexCastTypeConstraint, IndexCastTypeConstraint> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is sign-extended. If casting to a narrower
    integer, the value is truncated.
  }];

  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

#endif // MemAcc_OPS