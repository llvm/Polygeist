//===- SQLLower.cpp - Lower PostgreSQL to sql mlir ops ------ -*-===//
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
#include "sql/SQLOps.h"
#include "sql/Passes/Passes.h"
#include <algorithm>
#include <mutex>

#define DEBUG_TYPE "sql-lower-opt"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::sql;

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

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(loop);

    // 1) make sure the postgres_getresult function is declared
    auto rowsfn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("PQcmdTuples")));

    auto atoifn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("atoi")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = loop.getHandle();
    arg = rewriter.create<arith::IndexCastOp>(loop.getLoc(),
                                              rewriter.getI64Type(), arg);
    arg = rewriter.create<LLVM::IntToPtrOp>(
        loop.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), arg);

    // 3) call and replace
    Value args[] = {arg}; 
    
    Value res =
        rewriter.create<mlir::func::CallOp>(loop.getLoc(), rowsfn, args)
            ->getResult(0);

    Value args2[] = {res}; 
    
    Value res2 =
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

  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(module);
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  if (!dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("PQcmdTuples")))) {
    mlir::Type argtypes[] = {LLVM::LLVMPointerType::get(builder.getI8Type())};
    mlir::Type rettypes[] = {LLVM::LLVMPointerType::get(builder.getI8Type())};

    auto fn =
        builder.create<func::FuncOp>(module.getLoc(), "PQcmdTuples",
                                     builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }
  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("atoi")))) {
    mlir::Type argtypes[] = {LLVM::LLVMPointerType::get(builder.getI8Type())};

    // todo use data layout
    mlir::Type rettypes[] = {builder.getI64Type()};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "atoi", builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<NumResultsOpLowering>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace sql {
std::unique_ptr<Pass> createSQLLowerPass() {
  return std::make_unique<SQLLower>();
}
} // namespace polygeist
} // namespace mlir
