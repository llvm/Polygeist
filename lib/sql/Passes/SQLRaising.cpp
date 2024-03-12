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
#include "sql/Parser.h"
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

struct PQntuplesRaising : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const final {                                
    if (call.getCallee() != "PQntuples") {
        return failure();
    }
    auto module = call->getParentOfType<ModuleOp>();
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);


    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = call.getArgOperands()[0];
    arg = rewriter.create<polygeist::Memref2PointerOp>(
              call.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), arg);
    arg = rewriter.create<LLVM::PtrToIntOp>(call.getLoc(), rewriter.getIntegerType(64), arg);
    arg = rewriter.create<arith::IndexCastOp>(call.getLoc(), rewriter.getIndexType(), arg);


    Value res = rewriter.create<sql::NumResultsOp>(call.getLoc(), rewriter.getIndexType(), arg);
    res = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getI64Type(), res);
    rewriter.replaceOp(call, res);
    return success();
  }
};

struct PQgetvalueRaising : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const final {                                
    if (call.getCallee() != "PQgetvalue") {
        return failure();
    }
    auto module = call->getParentOfType<ModuleOp>();
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 2) convert the args to valid args to postgres_getresult abi
    Value handle = call.getArgOperands()[0];
    handle = rewriter.create<polygeist::Memref2PointerOp>(
              call.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), handle);
    handle = rewriter.create<LLVM::PtrToIntOp>(call.getLoc(), rewriter.getIntegerType(64), handle);
    handle = rewriter.create<arith::IndexCastOp>(call.getLoc(), rewriter.getIndexType(), handle);

    Value row = call.getArgOperands()[1];
    row = rewriter.create<arith::IndexCastOp>(call.getLoc(), rewriter.getIndexType(), row);
    Value column = call.getArgOperands()[2];
    column = rewriter.create<arith::IndexCastOp>(call.getLoc(), rewriter.getIndexType(), column);
    
    Value res = rewriter.create<sql::GetValueOp>(call.getLoc(), rewriter.getIndexType(), handle, row, column);

    res = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getI64Type(), res);

    Value args2[] = {res};


    auto itoafn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("itoa")));

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(call, itoafn, args2); 

    return success();
  }
};


struct PQexecRaising : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const final {                                
    if (call.getCallee() != "PQexec") {
        return failure();
    }

    // 2) convert the args to valid args to postgres_getresult abi
    Value conn = call.getArgOperands()[0];
    if (!conn.getType().isa<LLVM::LLVMPointerType>()) {
        conn = rewriter.create<polygeist::Memref2PointerOp>(
              call.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), conn);
    }
    conn = rewriter.create<LLVM::PtrToIntOp>(
        call.getLoc(), rewriter.getIntegerType(64), conn);

    conn = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getIndexType(), conn);

    Value command = call.getArgOperands()[1];

    command = rewriter.create<sql::UnparsedOp>(call.getLoc(), 
                                                  ExprType::get(rewriter.getContext()), command);

    Value res = rewriter.create<sql::ExecuteOp>(call.getLoc(), rewriter.getIndexType(), conn, command);

    res = rewriter.create<arith::IndexCastOp>(call.getLoc(),
                                              rewriter.getI64Type(), res);
    res = rewriter.create<LLVM::IntToPtrOp>(call.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), res);
    res = rewriter.create<polygeist::Pointer2MemrefOp>(call.getLoc(), call.getResultTypes()[0], res);

    
    assert(call.getResultTypes()[0] == res.getType());
    rewriter.replaceOp(call, res);

    return success();
  }
};


class UnparsedOpParseFix final : public OpRewritePattern<UnparsedOp> {
public:
  using OpRewritePattern<UnparsedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnparsedOp op,
                                PatternRewriter &rewriter) const override {

    Value input = op->getOperand(0);
    
    auto stringOp = input.getDefiningOp<LLVM::AddressOfOp>();
    if (!stringOp) return failure();

    StringRef strname = stringOp.getGlobalName();

    auto module = op->getParentOfType<ModuleOp>();
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    Attribute strattr = dyn_cast_or_null<LLVM::GlobalOp>(
            symbolTable.lookupSymbolIn(module, rewriter.getStringAttr(strname))).getValueAttr();
    auto str = strattr.cast<StringAttr>().getValue();

    auto resOp = parseSQL(op.getLoc(), rewriter, std::string(str.data()));
    assert(resOp);
    rewriter.replaceOp(op,ValueRange(resOp));
    return success();
  }
};

void SQLRaising::runOnOperation() {
  auto module = getOperation();
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(module);
  auto &context = getContext(); 
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());

  if (!dyn_cast_or_null<func::FuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("itoa")))) {
    mlir::Type argtypes[] = {builder.getI64Type()};
    mlir::Type rettypes[] = {MemRefType::get({-1}, builder.getI8Type())};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "itoa", builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  RewritePatternSet patterns(&context);
  patterns.insert<PQntuplesRaising, UnparsedOpParseFix, PQgetvalueRaising, PQexecRaising>(&getContext());

  for (auto *dialect : context.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : context.getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &context);

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
