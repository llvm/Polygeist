//===- SQLOps.cpp - SQL dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "polygeist/Ops.h"
#include "sql/Parser.h"
#include "sql/SQLDialect.h"
#include "sql/SQLOps.h"
#include "sql/SQLTypes.h"

#define GET_OP_CLASSES
#include "sql/SQLOps.cpp.inc"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/SideEffectUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "sql"

using namespace mlir;
using namespace sql;
using namespace mlir::arith;

class GetValueOpTypeFix final : public OpRewritePattern<GetValueOp> {
public:
  using OpRewritePattern<GetValueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetValueOp op,
                                PatternRewriter &rewriter) const override {

    bool changed = false;

    Value handle = op.getOperand(0);
    if (!handle.getType().isa<IndexType>()) {
      handle = rewriter.create<IndexCastOp>(op.getLoc(),
                                            rewriter.getIndexType(), handle);
      changed = true;
    }
    Value row = op.getOperand(1);
    if (!row.getType().isa<IndexType>()) {
      row = rewriter.create<IndexCastOp>(op.getLoc(), rewriter.getIndexType(),
                                         row);
      changed = true;
    }
    Value column = op.getOperand(2);
    if (!column.getType().isa<IndexType>()) {
      column = rewriter.create<IndexCastOp>(op.getLoc(),
                                            rewriter.getIndexType(), column);
      changed = true;
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<GetValueOp>(op, op.getType(), handle, row,
                                            column);

    return success(changed);
  }
};

void GetValueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<GetValueOpTypeFix>(context);
}

class NumResultsOpTypeFix final : public OpRewritePattern<NumResultsOp> {
public:
  using OpRewritePattern<NumResultsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NumResultsOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    Value handle = op->getOperand(0);

    if (handle.getType().isa<IndexType>() &&
        op->getResultTypes()[0].isa<IndexType>())
      return failure();

    if (!handle.getType().isa<IndexType>()) {
      handle = rewriter.create<IndexCastOp>(op.getLoc(),
                                            rewriter.getIndexType(), handle);
      changed = true;
    }

    mlir::Value res = rewriter.create<NumResultsOp>(
        op.getLoc(), rewriter.getIndexType(), handle);

    if (op->getResultTypes()[0].isa<IndexType>()) {
      rewriter.replaceOp(op, res);
    } else {
      rewriter.replaceOpWithNewOp<IndexCastOp>(op, op->getResultTypes()[0],
                                               res);
    }

    return success(changed);
  }
};

void NumResultsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<NumResultsOpTypeFix>(context);
}

// class ExecuteOpTypeFix final : public OpRewritePattern<ExecuteOp> {
// public:
//   using OpRewritePattern<ExecuteOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(ExecuteOp op,
//                                 PatternRewriter &rewriter) const override {
//     bool changed = false;

//     Value conn = op->getOperand(0);
//     Value command = op->getOperand(1);

//     if (conn.getType().isa<IndexType>() && command.getType().isa<IndexType>()
//     && op->getResultTypes()[0].isa<IndexType>())
//         return failure();

//     if (!conn.getType().isa<IndexType>()) {
//         conn = rewriter.create<IndexCastOp>(op.getLoc(),
//                                                    rewriter.getIndexType(),
//                                                    conn);
//         changed = true;
//     }
//     if (command.getType().isa<MemRefType>()) {
//         command = rewriter.create<polygeist::Memref2PointerOp>(op.getLoc(),
//                                                    LLVM::LLVMPointerType::get(rewriter.getI8Type()),
//                                                    command);
//         changed = true;
//     }

//     if (command.getType().isa<LLVM::LLVMPointerType>()) {
//         command = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(),
//                                                    rewriter.getI64Type(),
//                                                    command);
//         changed = true;
//     }
//     if (!command.getType().isa<IndexType>()) {
//         command = rewriter.create<IndexCastOp>(op.getLoc(),
//                                                rewriter.getIndexType(),
//                                                command);
//         changed = true;
//     }

//     if (!changed) return failure();
//     mlir::Value res = rewriter.create<ExecuteOp>(op.getLoc(),
//     rewriter.getIndexType(), conn, command); rewriter.replaceOp(op, res);
//     // if (op->getResultTypes()[0].isa<IndexType>()) {
//     //     rewriter.replaceOp(op, res);
//     // } else {
//     //     rewriter.replaceOpWithNewOp<IndexCastOp>(op,
//     op->getResultTypes()[0], res);
//     // }
//     return success(changed);
//   }
// };

// void ExecuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                             MLIRContext *context) {
//   results.insert<ExecuteOpTypeFix>(context);
// }

template <typename T>
class UnparsedOpInnerCast final : public OpRewritePattern<UnparsedOp> {
public:
  using OpRewritePattern<UnparsedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnparsedOp op,
                                PatternRewriter &rewriter) const override {

    Value input = op->getOperand(0);

    auto cst = input.getDefiningOp<T>();
    if (!cst)
      return failure();

    rewriter.replaceOpWithNewOp<UnparsedOp>(op, op.getType(), cst.getOperand());
    return success();
  }
};

void UnparsedOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<UnparsedOpInnerCast<polygeist::Pointer2MemrefOp>>(context);
}

class SQLStringConcatOpCanonicalization final
    : public OpRewritePattern<SQLStringConcatOp> {
public:
  using OpRewritePattern<SQLStringConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SQLStringConcatOp op,
                                PatternRewriter &rewriter) const override {
    // Whether we changed the state. If we make no simplifications we need to
    // return failure otherwise we will infinite loop
    bool changed = false;
    // Operands to the simplified concat
    SmallVector<Value> operands;
    // Constants that we will merge, "current running constant"
    SmallVector<SQLConstantStringOp> constants;
    for (auto op : op->getOperands()) {
      if (auto constOp = op.getDefiningOp<SQLConstantStringOp>()) {
        constants.push_back(constOp);
        continue;
      }
      if (constants.size() != 0) {
        if (constants.size() == 1) {
          operands.push_back(constants[0]);
        } else {
          std::string nextStr;
          changed = true;
          for (auto str : constants)
            nextStr += str.getInput().str();

          operands.push_back(rewriter.create<SQLConstantStringOp>(
              op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), nextStr));
        }
      }
      constants.clear();
      if (auto concat = op.getDefiningOp<SQLStringConcatOp>()) {
        changed = true;
        for (auto op2 : concat->getOperands())
          operands.push_back(op2);
        continue;
      }
      operands.push_back(op);
    }
    if (constants.size() != 0) {
      if (constants.size() == 1) {
        operands.push_back(constants[0]);
      } else {
        std::string nextStr;
        changed = true;
        for (auto str : constants)
          nextStr = nextStr + str.getInput().str();
        operands.push_back(rewriter.create<SQLConstantStringOp>(
            op.getLoc(), MemRefType::get({-1}, rewriter.getI8Type()), nextStr));
      }
    }
    if (operands.size() == 0) {
      rewriter.replaceOpWithNewOp<SQLConstantStringOp>(op, MemRefType::get({-1}, rewriter.getI8Type()), "");
      return success();
    }
    if (operands.size() == 1) {
      rewriter.replaceOp(op, operands[0]);
      return success();
    }
    if (changed) {
      rewriter.replaceOpWithNewOp<SQLStringConcatOp>(op, MemRefType::get({-1}, rewriter.getI8Type()), operands);
      return success();
    }
    return failure();
  }
};

void SQLStringConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<SQLStringConcatOpCanonicalization>(context);
}
