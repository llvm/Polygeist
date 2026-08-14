//===- KernelOps.cpp - Kernel dialect operations ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Kernel/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

//===----------------------------------------------------------------------===//
// DefnOp
//===----------------------------------------------------------------------===//

LogicalResult DefnOp::verify() {
  // Check that the body region has exactly one block
  if (!getBody().hasOneBlock())
    return emitOpError("body region must have exactly one block");
  
  // The block can have any number of arguments
  // No special verification needed for block arguments
  
  return success();
}

ParseResult DefnOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { 
    return builder.getFunctionType(argTypes, results); 
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DefnOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  auto defnOp = cast<DefnOp>((*this)->getParentOp());

  // The operand number and types must match the kernel signature.
  const auto &results = defnOp.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing kernel (@"
           << defnOp.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of yield operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match kernel result type ("
                         << results[i] << ")"
                         << " in kernel @" << defnOp.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

FunctionType LaunchOp::getKernelType() {
  // Get the kernel symbol reference
  auto kernelAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("kernel");
  if (!kernelAttr)
    return nullptr;
  
  // Look up the kernel DefnOp in the symbol table
  auto *symbolTableOp = (*this)->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp)
    return nullptr;
    
  auto kernelOp = dyn_cast_or_null<DefnOp>(
      SymbolTable::lookupSymbolIn(symbolTableOp, kernelAttr));
  if (!kernelOp)
    return nullptr;
    
  return kernelOp.getFunctionType();
}

LogicalResult LaunchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the kernel attribute was specified.
  auto kernelAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("kernel");
  if (!kernelAttr)
    return emitOpError("requires a 'kernel' symbol reference attribute");

  // Check that the kernel symbol exists and is a DefnOp.
  auto kernelOp = symbolTable.lookupNearestSymbolFrom<DefnOp>(*this, kernelAttr);
  if (!kernelOp)
    return emitOpError() << "'" << kernelAttr.getValue()
                         << "' does not reference a valid kernel";

  // Verify that the operand and result types match the kernel signature.  A
  // bufferized launch keeps the scalar operands unchanged, replaces tensor
  // operands with equivalent memrefs, and writes the kernel results into the
  // destination operands recorded by polygeist.result_destinations.
  auto kernelType = kernelOp.getFunctionType();
  bool isBufferized = (*this)->hasAttr("polygeist.bufferized");
  if (kernelType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for kernel");

  auto areBufferCompatible = [](Type expected, Type actual) {
    if (expected == actual)
      return true;
    auto expectedTensor = dyn_cast<RankedTensorType>(expected);
    auto actualMemref = dyn_cast<MemRefType>(actual);
    if (!expectedTensor || !actualMemref ||
        expectedTensor.getElementType() != actualMemref.getElementType() ||
        expectedTensor.getRank() != actualMemref.getRank())
      return false;
    for (auto [expectedDim, actualDim] :
         llvm::zip(expectedTensor.getShape(), actualMemref.getShape()))
      if (!ShapedType::isDynamic(expectedDim) &&
          !ShapedType::isDynamic(actualDim) && expectedDim != actualDim)
        return false;
    return true;
  };

  for (unsigned i = 0, e = kernelType.getNumInputs(); i != e; ++i)
    if ((!isBufferized && getOperand(i).getType() != kernelType.getInput(i)) ||
        (isBufferized &&
         !areBufferCompatible(kernelType.getInput(i),
                              getOperand(i).getType())))
      return emitOpError("operand type mismatch: expected operand type ")
             << kernelType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (isBufferized) {
    if (getNumResults() != 0)
      return emitOpError("bufferized launch must not have SSA results");
    auto destinations = (*this)->getAttrOfType<DenseI64ArrayAttr>(
        "polygeist.result_destinations");
    if (!destinations || destinations.size() != kernelType.getNumResults())
      return emitOpError("bufferized launch requires one destination operand "
                         "index for every kernel result");
    for (int64_t destination : destinations.asArrayRef())
      if (destination < 0 || destination >= (int64_t)getNumOperands())
        return emitOpError("bufferized launch destination operand index is "
                           "out of range");
    return success();
  }

  if (kernelType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for kernel");

  for (unsigned i = 0, e = kernelType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != kernelType.getResult(i))
      return emitOpError("result type mismatch: expected result type ")
             << kernelType.getResult(i) << ", but provided "
             << getResult(i).getType() << " for result number " << i;
  
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "polygeist/Kernel/KernelOps.cpp.inc"
