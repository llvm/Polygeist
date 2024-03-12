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
#include "polygeist/Ops.h"
#include "sql/Passes/Passes.h"
#include "sql/SQLOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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
    auto rowsfn = dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
        module, rewriter.getStringAttr("PQntuples")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value arg = op.getHandle();
    arg = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI8Type(),
                                              arg);

    arg = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), arg);

    arg = rewriter.create<polygeist::Pointer2MemrefOp>(
        op.getLoc(), rowsfn.getFunctionType().getInput(0), arg);

    // 3) call and replace
    Value args[] = {arg};

    Value res = rewriter.create<mlir::func::CallOp>(op.getLoc(), rowsfn, args)
                    ->getResult(0);

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, rewriter.getIndexType(),
                                                    res);

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
    auto valuefn = dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
        module, rewriter.getStringAttr("PQgetvalue")));

    auto atoifn = dyn_cast_or_null<func::FuncOp>(
        symbolTable.lookupSymbolIn(module, rewriter.getStringAttr("atoi")));

    // 2) convert the args to valid args to postgres_getresult abi
    Value handle = op.getHandle();
    handle = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                 rewriter.getI64Type(), handle);
    handle = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), handle);

    handle = rewriter.create<polygeist::Pointer2MemrefOp>(
        op.getLoc(), valuefn.getFunctionType().getInput(0), handle);

    Value row = op.getRow();
    row = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), valuefn.getFunctionType().getInput(1), row);
    Value column = op.getColumn();
    column = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), valuefn.getFunctionType().getInput(2), column);

    Value args[] = {handle, row, column};

    Value res = rewriter.create<mlir::func::CallOp>(op.getLoc(), valuefn, args)
                    ->getResult(0);

    Value args2[] = {res};

    Value res2 =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), atoifn, args2)
            ->getResult(0);

    if (op.getType() != res2.getType()) {
      if (op.getType().isa<IndexType>())
        res2 = rewriter.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                                   res2);
      else if (auto IT = op.getType().dyn_cast<IntegerType>()) {
        auto IT2 = res2.getType().dyn_cast<IntegerType>();
        if (IT.getWidth() < IT2.getWidth()) {
          res2 =
              rewriter.create<arith::TruncIOp>(op.getLoc(), op.getType(), res2);
        } else if (IT.getWidth() > IT2.getWidth()) {
          res2 =
              rewriter.create<arith::ExtUIOp>(op.getLoc(), op.getType(), res2);
        } else
          assert(0 && "illegal integer type conversion");
      } else {
        assert(0 && "illegal type conversion");
      }
    }
    rewriter.replaceOp(op, res2);

    return success();
  }
};

struct ConstantStringOpLowering
    : public OpRewritePattern<sql::SQLConstantStringOp> {
  using OpRewritePattern<sql::SQLConstantStringOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::SQLConstantStringOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);
    for (auto u : op.getResult().getUsers()) {
      if (isa<sql::SQLStringConcatOp>(u))
        return failure();
    }
    auto expr = op.getInput().str();
    auto name = "str" + std::to_string((long long int)(Operation *)op);
    auto MT = MemRefType::get({expr.size() + 1}, rewriter.getI8Type());
    // auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {});
    auto getglob = rewriter.create<memref::GetGlobalOp>(op.getLoc(), MT, name);

    SmallVector<char, 1> data(expr.begin(), expr.end());
    data.push_back('\0');
    auto attr = DenseElementsAttr::get<char>(
        RankedTensorType::get(MT.getShape(), MT.getElementType()), data);

    auto loc = op.getLoc();
    rewriter.replaceOpWithNewOp<memref::CastOp>(
        op, MemRefType::get({-1}, rewriter.getI8Type()), getglob.getResult());
    rewriter.setInsertionPointToStart(module.getBody());
    auto res = rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name), mlir::StringAttr(),
        mlir::TypeAttr::get(MT), attr, rewriter.getUnitAttr(),
        /*alignment*/ nullptr);

    return success();
  }
};


struct BoolToStringOpLowering : public OpRewritePattern<sql::SQLBoolToStringOp> {
  using OpRewritePattern<sql::SQLBoolToStringOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::SQLBoolToStringOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 2) convert the args to valid args to postgres_getresult abi
    Value expr = op.getExpr();
    auto definingOp = expr.getDefiningOp();
    if (auto andOp = dyn_cast<sql::AndOp>(definingOp)) {
      Value left = rewriter.create<sql::SQLBoolToStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          andOp.getLeft());
      Value right = rewriter.create<sql::SQLBoolToStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          andOp.getRight());
      Value args[] = {left, rewriter.create<sql::SQLConstantStringOp>(
                               op.getLoc(),
                               MemRefType::get({-1}, rewriter.getI8Type()),
                               "AND "), right};
      Value res = rewriter.create<sql::SQLStringConcatOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
      rewriter.replaceOp(op, res);
    } else if (auto orOp = dyn_cast<sql::OrOp>(definingOp)) {
      Value left = rewriter.create<sql::SQLBoolToStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          orOp.getLeft());
      Value right = rewriter.create<sql::SQLBoolToStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          orOp.getRight());
      Value args[] = {left, rewriter.create<sql::SQLConstantStringOp>(
                               op.getLoc(),
                               MemRefType::get({-1}, rewriter.getI8Type()),
                               "OR "), right};
      Value res = rewriter.create<sql::SQLStringConcatOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
      rewriter.replaceOp(op, res);
    } else if (auto calcBoolOp = dyn_cast<sql::CalcBoolOp>(definingOp)){
      auto expr = calcBoolOp.getExpr(); 
      if (expr == "") { 
        Value res = rewriter.create<sql::SQLConstantStringOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
            "");
        rewriter.replaceOp(op, res);
        return success();
      } 
      
      Value res = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), expr);
      Value args[] = {res, rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), " ")};
      res = rewriter.create<sql::SQLStringConcatOp>( 
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
      rewriter.replaceOp(op, res);
    } else {
      assert(0 && "unknown type to convert to string");
    }

    return success();
  }
};


struct ToStringOpLowering : public OpRewritePattern<sql::SQLToStringOp> {
  using OpRewritePattern<sql::SQLToStringOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sql::SQLToStringOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(module);

    // 2) convert the args to valid args to postgres_getresult abi
    Value expr = op.getExpr();
    auto definingOp = expr.getDefiningOp();
    if (auto selectOp = dyn_cast<sql::SelectOp>(definingOp)) {
      Value current = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), "SELECT ");

      bool prevColumn = false;
      auto columns = selectOp.getColumns();
      for (mlir::Value v : columns) {
        if (prevColumn) {
          Value args[] = {
              current, rewriter.create<sql::SQLConstantStringOp>(
                           op.getLoc(),
                           MemRefType::get({-1}, rewriter.getI8Type()), ", ")};
          current = rewriter.create<sql::SQLStringConcatOp>(
              op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
        }
        Value col = rewriter.create<sql::SQLToStringOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), v);
        Value args[] = {
            col,
            rewriter.create<sql::SQLConstantStringOp>(
                op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), " ")};
        col = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
        Value args2[] = {current, col};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args2);
        prevColumn = true;
      }

      auto tableOp = selectOp.getTable().getDefiningOp<sql::TableOp>();
      if (tableOp) {
        Value args[] = {
            current, rewriter.create<sql::SQLConstantStringOp>(
                         op.getLoc(),
                         MemRefType::get({-1}, rewriter.getI8Type()), "FROM ")};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
        Value args2[] = {current,
                         rewriter.create<sql::SQLToStringOp>(
                             op.getLoc(),
                             MemRefType::get({-1}, rewriter.getI8Type()),
                             selectOp.getTable())};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args2);
      }

      Value whereVal = rewriter.create<sql::SQLToStringOp>(
                             op.getLoc(),
                             MemRefType::get({-1}, rewriter.getI8Type()),
                             selectOp.getWhere()); 
      if (whereVal) { 
        auto whereOp = selectOp.getWhere().getDefiningOp<sql::WhereOp>();
        Value args[] = {
            current, rewriter.create<sql::SQLConstantStringOp>(
                         op.getLoc(),
                         MemRefType::get({-1}, rewriter.getI8Type()), "WHERE ")};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
        Value args2[] = {current, whereVal};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args2);
      }
      

      auto limit = selectOp.getLimit();
      if (limit >= 0) {
        Value args[] = {current,
                        rewriter.create<sql::SQLConstantStringOp>(
                            op.getLoc(),
                            MemRefType::get({-1}, rewriter.getI8Type()),
                            "LIMIT ")};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
        Value args2[] = {current,
                         rewriter.create<sql::SQLConstantStringOp>(
                             op.getLoc(),
                             MemRefType::get({-1}, rewriter.getI8Type()),
                             std::to_string(limit))};
        current = rewriter.create<sql::SQLStringConcatOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args2);
      }
      rewriter.replaceOp(op, current);
    } else if (auto tabOp = dyn_cast<sql::TableOp>(definingOp)) {
      Value res = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          tabOp.getExpr());
      Value args[] = { res, rewriter.create<sql::SQLConstantStringOp>(
              op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), " ")};
      res = rewriter.create<sql::SQLStringConcatOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), args);
      rewriter.replaceOp(op, res);
    } else if (auto allColOp = dyn_cast<sql::AllColumnsOp>(definingOp)) {
      Value res = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), "*");
      rewriter.replaceOp(op, res);
    } else if (auto colOp = dyn_cast<sql::ColumnOp>(definingOp)) {
      Value res = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          colOp.getExpr());
      rewriter.replaceOp(op, res);
    } else if (auto intOp = dyn_cast<sql::IntOp>(definingOp)) {
      Value res = rewriter.create<sql::SQLConstantStringOp>(
          op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          intOp.getExpr());
      llvm::errs() << "intOp: " << intOp.getExpr() << "\n";
      rewriter.replaceOp(op, res);
    } else if (auto whereOp = dyn_cast<sql::WhereOp>(definingOp)) {
      Value res = rewriter.create<sql::SQLBoolToStringOp>(
                             op.getLoc(),
                             MemRefType::get({-1}, rewriter.getI8Type()),
                             whereOp.getExpr());
      rewriter.replaceOp(op, res);
    } else {
      assert(0 && "unknown type to convert to string");
    }

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
    conn = rewriter.create<polygeist::Pointer2MemrefOp>(
        op.getLoc(), executefn.getFunctionType().getInput(0), conn);

    Value command = op.getCommand();
    // auto name = "str" + std::to_string((long long int)(Operation
    // *)command.getDefiningOp());
    command = rewriter.create<sql::SQLToStringOp>(
        op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), command);
    llvm::errs() << "command: " << command << "\n";
    llvm::errs() << "command type: " << command.getType() << "\n";
    // auto type = MemRefType::get({-1}, rewriter.getI8Type());

    // auto globalOp = rewriter.create<LLVM::GlobalOp>(
    //     op.getLoc(), type, /* isConstant */ true, LLVM::Linkage::Internal,
    //     name, mlir::Attribute(),
    //     /* alignment */ 0, /* addrSpace */ 0);

    // 3) call and replace
    Value args[] = {conn, command};

    Value res =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), executefn, args)
            ->getResult(0);
    res = rewriter.create<polygeist::Memref2PointerOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), res);
    res = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), rewriter.getI64Type(),
                                            res);
    res = rewriter.create<arith::IndexCastOp>(op.getLoc(), op.getType(), res);

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

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "PQntuples",
        builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  if (!dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("PQgetvalue")))) {
    mlir::Type argtypes[] = {MemRefType::get({-1}, builder.getI8Type()),
                             builder.getI64Type(), builder.getI64Type()};
    mlir::Type rettypes[] = {MemRefType::get({-1}, builder.getI8Type())};

    auto fn = builder.create<func::FuncOp>(
        module.getLoc(), "PQgetvalue",
        builder.getFunctionType(argtypes, rettypes));
    SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
  }

  if (!dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("PQexec")))) {
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
    // mlir::Type argtypes[] =
    // {LLVM::LLVMPointerType::get(builder.getI64Type())};

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
  patterns.insert<ToStringOpLowering>(&getContext());
  patterns.insert<BoolToStringOpLowering>(&getContext());
  patterns.insert<ConstantStringOpLowering>(&getContext());

  for (auto *dialect : getContext().getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : getContext().getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace sql {
std::unique_ptr<Pass> createSQLLowerPass() {
  return std::make_unique<SQLLower>();
}
} // namespace sql
} // namespace mlir
