//===- SQLLower.cpp - Lower sql ops to mlir ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic SQL for representation
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <mutex>

#define DEBUG_TYPE "sql-opt"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace sql;

namespace {
struct SQLLower : public SQLLowerBase<SQLLower> {
  void runOnOperation() override;
};

} // end anonymous namespace

struct NumResultsOpLowering : public OpRewritePattern<sql::NumResultsOp> {
  using OpRewritePattern<sql::NumResultsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::NumResultsOp loop,
                                PatternRewriter &rewriter) const final {
    auto module = loop->getParentOfType<ModuleOp>();

    // 1) make sure the postgres_getresult function is declared
    auto rowsfn = dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
        module, builder.getStringAttr("PQcmdTuples")));

    auto atoifn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, builder.getStringAttr("atoi")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = loop.getHandle();
    arg = rewriter.create<arith::IndexCastOp>(loop.getLoc(),
                                              rewriter.getIntTy(64), arg);
    arg = rewriter.create<LLVM::IntToPtrOp>(
        loop.getLoc(), LLVM::LLVMPointerType::get(builder.getInt8Ty()), arg);

    // 3) call and replace
    Value args[] = {arg} Value res =
        rewriter.create<mlir::func::CallOp>(loop.getLoc(), rowsfn, args)
            ->getResult(0);

    Value args2[] = {res} Value res2 =
        rewriter.create<mlir::func::CallOp>(loop.getLoc(), atoifn, args2)
            ->getResult(0);

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        loop, rewriter.getIndexType(), res2);

    // 4) done
    return success();
  }
};

void SQLLower::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  if (!dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("PQcmdTuples")))) {
    mlir::Type argtypes[] = {LLVM::LLVMPointerType::get(builder.getInt8Ty())};
    mlir::Type rettypes[] = {LLVM::LLVMPointerType::get(builder.getInt8Ty())};

    auto fn =
        builder.create<func::FuncOp>(module.getLoc(), "PQcmdTuples",
                                     builder.getFunctionType(argtys, rettys));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Private);
  }
  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("atoi")))) {
    mlir::Type argtypes[] = {LLVM::LLVMPointerType::get(builder.getInt8Ty())};

    // todo use data layout
    mlir::Type rettypes[] = {builder.getIntTy(sizeof(int))};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "atoi", builder.getFunctionType(argtys, rettys));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Private);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<NumResultsOpLowering>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createSQLLowerPass() {
  return std::make_unique<SQLLower>();
}
} // namespace polygeist
} // namespace mlir
