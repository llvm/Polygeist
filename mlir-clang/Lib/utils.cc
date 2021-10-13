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

/// TODO: rename

void mlirclang::initializeValueByInitListExpr(mlir::Value toInit, Expr *expr,
                                              MLIRScanner *scanner) {
  auto initListExpr = cast<InitListExpr>(expr);
  assert(toInit.getType().isa<MemRefType>() &&
         "The value initialized by an InitListExpr should be a MemRef.");

  // The initialization values will be translated into individual
  // memref.store operations. This requires that the memref value should
  // have static shape.
  MemRefType memTy = toInit.getType().cast<MemRefType>();
  assert(memTy.hasStaticShape() &&
         "The memref to be initialized by InitListExpr should have static "
         "shape.");

  auto shape = memTy.getShape();
  // `offsets` is being mutable during the recursive function call to helper.
  SmallVector<int64_t> offsets;

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<void(Expr *)> helper = [&](Expr *expr) {
    Location loc = toInit.getLoc();

    OpBuilder &b = scanner->getBuilder();
    OpBuilder::InsertionGuard guard(b);

    InitListExpr *initListExpr = dyn_cast<InitListExpr>(expr);

    // All the addresses have been instantiated, can generate the store
    // operation.
    if (offsets.size() == shape.size()) {
      // Generate the constant addresses.
      SmallVector<mlir::Value> addr;
      transform(offsets, std::back_inserter(addr), [&](const int64_t &offset) {
        return scanner->getConstantIndex(offset);
      });

      // Resolve the value to be stored.
      Expr *toVisit = (initListExpr && initListExpr->hasArrayFiller())
                          ? initListExpr->getInit(0)
                          : expr;
      assert(!isa<InitListExpr>(toVisit) &&
             "The expr to visit and resolve shouldn't still be a "
             "InitListExpr - "
             "it should be an actual value.");

      auto visitor = scanner->Visit(toVisit);
      auto valueToStore = visitor.getValue(b);

      b.create<memref::StoreOp>(loc, valueToStore, toInit, addr);
    } else {
      assert(initListExpr &&
             "The passed in expr should be an InitListExpr since we're still "
             "iterating all the values to be stored.");

      unsigned nextDim = offsets.size();
      offsets.push_back(0);
      for (unsigned i = 0, e = shape[nextDim]; i < e; ++i) {
        offsets[nextDim] = i;

        // If the current expr is an array filler, we will pass it all the
        // way down until we reach to the last dimension. Otherwise, there
        // should be a corresponding InitListExpr for the current dim, and
        // we pass that down.
        helper(initListExpr->hasArrayFiller() ? expr
                                              : initListExpr->getInit(i));
      }
      offsets.pop_back();
    }
  };

  helper(initListExpr);
}
