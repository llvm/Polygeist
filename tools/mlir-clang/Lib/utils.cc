//===- utils.cc -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "clang-mlir.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "clang/AST/Expr.h"

using namespace mlir;
using namespace llvm;
using namespace clang;

Operation *mlirclang::buildLinalgOp(const AbstractOperation *op, OpBuilder &b,
                                    SmallVectorImpl<mlir::Value> &input,
                                    SmallVectorImpl<mlir::Value> &output) {
  StringRef name = op->name;
  if (name.compare("linalg.copy") == 0) {
    assert(input.size() == 1 && "linalg::copyOp requires 1 input");
    assert(output.size() == 1 && "linalg::CopyOp requires 1 output");
    return b.create<linalg::CopyOp>(b.getUnknownLoc(), input[0], output[0]);
  } else {
    llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + name);
    return nullptr;
  }
}

Operation *
mlirclang::replaceFuncByOperation(FuncOp f, StringRef opName, OpBuilder &b,
                                  SmallVectorImpl<mlir::Value> &input,
                                  SmallVectorImpl<mlir::Value> &output) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  const AbstractOperation *op = AbstractOperation::lookup(opName, ctx);

  if (opName.startswith("linalg"))
    return buildLinalgOp(op, b, input, output);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), op->name, input,
                         f.getCallableResults(), {});
  return b.createOperation(opState);
}
