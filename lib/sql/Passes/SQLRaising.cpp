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

#define DEBUG_TYPE "sql-raising-opt"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::sql;

namespace {
struct SQLRaising : public SQLRaisingBase<SQLRaising> {
  void runOnOperation() override;
};

} // end anonymous namespace

struct PQcmdTuplesRaising : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const final {                                
    if (call.getCallee() != "PQcmdTuples") {
        return failure();
    }
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(call);
    auto module = call->getParentOfType<ModuleOp>();

    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = call.getArgOperands()[0];
    arg = rewriter.create<LLVM::PtrToIntOp>(
        call.getLoc(), rewriter.getIntegerType(64), arg);

    arg = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getIndexType(), arg);

    Value res = rewriter.create<sql::NumResultsOp>(call.getLoc(), rewriter.getIndexType(), arg);

    res = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getI64Type(), res);

    auto itoafn = dyn_cast_or_null<func::FuncOp>(
      symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("itoa")));

    Value args2[] = {res};

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(call, itoafn, args2);

    // 4) done
    return success();
  }
};

void SQLRaising::runOnOperation() {
  auto module = getOperation();
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(module);
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("itoa")))) {
    mlir::Type rettypes[] = {LLVM::LLVMPointerType::get(builder.getI8Type())};

    // todo use data layout
    mlir::Type argtypes[] = {builder.getI64Type()};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "itoa", builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<PQcmdTuplesRaising>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace sql {
std::unique_ptr<Pass> createSQLRaisingPass() {
  return std::make_unique<SQLRaising>();
}
} // namespace polygeist
} // namespace mlir
