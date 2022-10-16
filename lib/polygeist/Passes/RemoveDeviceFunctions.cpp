//===- RemoveDeviceFunctions.cpp - Remove unused private functions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "remove-device-functions"

using namespace mlir;
using namespace polygeist;

namespace {

void insertReturn(PatternRewriter &rewriter, func::FuncOp f) {
  rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc());
}
void insertReturn(PatternRewriter &rewriter, LLVM::LLVMFuncOp f) {
  rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), std::vector<Value>{});
}

template <typename FuncType>
struct RemoveFunction : public OpRewritePattern<FuncType> {
  using OpRewritePattern<FuncType>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncType f,
                                PatternRewriter &rewriter) const override {
    if (!isa<ModuleOp>(f->getParentOp())) {
      return failure();
    }
    auto V = f->getAttr("polygeist.device_only_func");
    if (!V) {
      return failure();
    }
    Region *region = &f.getBody();
    if (region->empty())
      return failure();
    rewriter.eraseOp(f);
    // TODO leave an empty function to pass to cudaSetCacheConfig
    //Region *region = &f.getBody();
    //if (region->empty())
    //  return failure();
    //rewriter.eraseBlock(&region->front());
    //region->push_back(new Block());
    //rewriter.setInsertionPointToEnd(&region->front());
    //insertReturn(rewriter, f);
    return success();
  }
};

struct RemoveDeviceFunctionsPass
    : public RemoveDeviceFunctionsBase<RemoveDeviceFunctionsPass> {
  RemoveDeviceFunctionsPass() {}
  void runOnOperation() override {
    ModuleOp m = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveFunction<func::FuncOp>,RemoveFunction<LLVM::LLVMFuncOp>>(&getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};

}

std::unique_ptr<Pass> mlir::polygeist::createRemoveDeviceFunctionsPass() {
  return std::make_unique<RemoveDeviceFunctionsPass>();
}
