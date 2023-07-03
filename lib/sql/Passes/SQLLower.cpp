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
#include "polygeist/Ops.h"
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

  LogicalResult matchAndRewrite(sql::NumResultsOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 1) make sure the postgres_getresult function is declared
    auto rowsfn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("PQntuples")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = op.getHandle();
    arg = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              rewriter.getI8Type(), arg);

    arg = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), arg);
    
    arg = rewriter.create<polygeist::Pointer2MemrefOp>(op.getLoc(), 
                        rowsfn.getFunctionType().getInput(0), arg);

    // 3) call and replace
    Value args[] = {arg}; 
    
    Value res =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), rowsfn, args)
            ->getResult(0);

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        op, rewriter.getIndexType(), res);

    return success();
  }
};


struct GetValueOpLowering : public OpRewritePattern<sql::GetValueOp> {
  using OpRewritePattern<sql::GetValueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::GetValueOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 1) make sure the postgres_getresult function is declared
    auto valuefn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("PQgetvalue")));

    auto atoifn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("atoi")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value handle = op.getHandle();
    handle = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              rewriter.getI64Type(), handle);
    handle = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), handle);

    handle = rewriter.create<polygeist::Pointer2MemrefOp>(op.getLoc(), 
                        valuefn.getFunctionType().getInput(0), handle);

    Value row = op.getRow();
    row = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              valuefn.getFunctionType().getInput(1), row);
    Value column = op.getColumn(); 
    column = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              valuefn.getFunctionType().getInput(2), column);

    Value args[] = {handle, row, column}; 
    
    Value res =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), valuefn, args)
            ->getResult(0);

    Value args2[] = {res}; 
    
    Value res2 =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), atoifn, args2)
            ->getResult(0);

    if (op.getType() != res2.getType()) {
        if (op.getType().isa<IndexType>())
            res2 = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                op.getType(), res2);
        else if (auto IT = op.getType().dyn_cast<IntegerType>()) {
            auto IT2 = res2.getType().dyn_cast<IntegerType>();
            if (IT.getWidth() < IT2.getWidth()) {
                res2 = rewriter.create<arith::TruncIOp>(op.getLoc(),
                                                op.getType(), res2);
            } else if (IT.getWidth() > IT2.getWidth()) {
                res2 = rewriter.create<arith::ExtUIOp>(op.getLoc(),
                                                op.getType(), res2);
            } else assert(0 && "illegal integer type conversion");
        } else {
            assert(0 && "illegal type conversion");
        }
    }
    rewriter.replaceOp(op, res2);

    return success();
  }
};

struct ExecuteOpLowering : public OpRewritePattern<sql::ExecuteOp> {
  using OpRewritePattern<sql::ExecuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::ExecuteOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 1) make sure the postgres_getresult function is declared
    auto executefn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("PQexec")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value conn = op.getConn();
    conn = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              rewriter.getI8Type(), conn);
    conn = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), conn);
    conn = rewriter.create<polygeist::Pointer2MemrefOp>(op.getLoc(), 
                        executefn.getFunctionType().getInput(0), conn);

    Value command = op.getCommand(); 
    command = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              rewriter.getI8Type(), command);
    command = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), command);
    StringRef strname = command.getDefiningOp<LLVM::AddressOfOp>().getGlobalName();
    Attribute strattr = dyn_cast_or_null<LLVM::GlobalOp>(
            symbolTable.lookupSymbolIn(module, rewriter.getStringAttr(strname))).getValueAttr();
    auto str = strattr.cast<StringAttr>().getValue();
    llvm::errs() << str << "\n";
    command = rewriter.create<polygeist::Pointer2MemrefOp>(op.getLoc(), 
                        executefn.getFunctionType().getInput(1), command);


    // 3) call and replace
    Value args[] = {conn, command}; 
    
    Value res =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), executefn, args)
            ->getResult(0);

    rewriter.replaceOp(op, res);

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
          module, builder.getStringAttr("PQntuples")))) {
    mlir::Type argtypes[] = {MemRefType::get({-1}, builder.getI8Type())};
    mlir::Type rettypes[] = {builder.getI64Type()};

    auto fn =
        builder.create<func::FuncOp>(module.getLoc(), "PQntuples",
                                     builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  if (!dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("PQgetvalue")))) {
    mlir::Type argtypes[] = {
                MemRefType::get({-1}, builder.getI8Type()), 
                builder.getI64Type(), 
                builder.getI64Type()};
    mlir::Type rettypes[] = {MemRefType::get({-1}, builder.getI8Type())};

    auto fn =
        builder.create<func::FuncOp>(module.getLoc(), "PQgetvalue",
                                     builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("PQexec")))) {
    mlir::Type argtypes[] = {MemRefType::get({-1}, builder.getI8Type()),
                             MemRefType::get({-1}, builder.getI8Type())};
    mlir::Type rettypes[] = {MemRefType::get({-1}, builder.getI8Type())};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "PQexec", builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("atoi")))) {
    mlir::Type argtypes[] = {MemRefType::get({-1}, builder.getI8Type())};
    // mlir::Type argtypes[] = {LLVM::LLVMPointerType::get(builder.getI64Type())};

    // todo use data layout
    mlir::Type rettypes[] = {builder.getI64Type()};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "atoi", builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<NumResultsOpLowering>(&getContext());
  patterns.insert<GetValueOpLowering>(&getContext());
  patterns.insert<ExecuteOpLowering>(&getContext());

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
