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
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "clang/AST/Expr.h"

using namespace mlir;
using namespace llvm;
using namespace clang;

Operation *buildLinalgOp(StringRef name, OpBuilder &b,
                         SmallVectorImpl<mlir::Value> &input,
                         SmallVectorImpl<mlir::Value> &output) {
  if (name.compare("memref.copy") == 0) {
    assert(input.size() == 1 && "memref::copyOp requires 1 input");
    assert(output.size() == 1 && "memref::CopyOp requires 1 output");
    return b.create<memref::CopyOp>(b.getUnknownLoc(), input[0], output[0]);
  } else {
    llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + name);
    return nullptr;
  }
}

Operation *mlirclang::replaceFuncByOperation(
    func::FuncOp f, StringRef opName, OpBuilder &b,
    SmallVectorImpl<mlir::Value> &input, SmallVectorImpl<mlir::Value> &output) {
  MLIRContext *ctx = f->getContext();
  if (!ctx->isOperationRegistered(opName)) {
    ctx->allowUnregisteredDialects();
    llvm::errs() << " warning unregistered dialect op: " << opName << "\n";
  }

  if (opName.startswith("memref"))
    return buildLinalgOp(opName, b, input, output);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), opName, input, f.getResultTypes(),
                         {});
  return b.create(opState);
}

mlir::Value mlirclang::castInteger(mlir::OpBuilder &builder,
                                   mlir::Location &loc, mlir::Value v,
                                   mlir::Type postTy_) {
  auto prevTy = v.getType().cast<mlir::IntegerType>();
  auto postTy = postTy_.cast<mlir::IntegerType>();
  if (prevTy.getWidth() < postTy.getWidth())
    return builder.create<mlir::arith::ExtUIOp>(loc, postTy, v);
  else if (prevTy.getWidth() > postTy.getWidth())
    return builder.create<mlir::arith::TruncIOp>(loc, postTy, v);
  else
    return v;
}
