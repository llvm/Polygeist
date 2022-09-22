//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "convert-parallel-to-gpu"

using namespace mlir;
using namespace polygeist;

namespace {

struct SharedLLVMAllocaToGlobal : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp ao, PatternRewriter &rewriter) const override {
    auto PT = ao.getType().cast<LLVM::LLVMPointerType>();
    if (PT.getAddressSpace() != 5) {
      return failure();
    }

    auto type = PT.getElementType();
    auto loc = ao->getLoc();
    auto name = "shared_mem." + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!module) {
      return failure();
    }

    rewriter.setInsertionPointToStart(module.getBody());

    auto globalOp = rewriter.create<LLVM::GlobalOp>(loc, type, /* isConstant */ false, LLVM::Linkage::Internal, name, mlir::Attribute(),
                                                    /* alignment */ 0, /* addrSpace */ 3);
    rewriter.setInsertionPoint(ao);
    auto aoo = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
    //assert(aoo.getType() == ao.getType());

    rewriter.replaceOp(ao, aoo->getResults());

    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao, PatternRewriter &rewriter) const override {
    if (ao.getType().getMemorySpace().dyn_cast_or_null<IntegerAttr>().getValue() != 5) {
      return failure();
    }

    auto type = ao.getType();
    auto loc = ao->getLoc();
    auto name = "shared_mem." + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<ModuleOp>();
    Operation *fun = ao->getParentOfType<LLVM::LLVMFuncOp>();
    if (!fun)
      fun = ao->getParentOfType<func::FuncOp>();

    rewriter.setInsertionPointToStart(module.getBody());
    auto initial_value = rewriter.getUnitAttr();
    auto globalOp = rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr(name),
                                                      mlir::StringAttr(), mlir::TypeAttr::get(type),
                                                      initial_value, mlir::UnitAttr(), /*alignment*/ nullptr);
    rewriter.setInsertionPointToStart(&fun->getRegion(0).front());
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);


    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

struct ConvertParallelToGPUPass
    : public ConvertParallelToGPUBase<ConvertParallelToGPUPass> {
  ConvertParallelToGPUPass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SharedLLVMAllocaToGlobal, SharedMemrefAllocaToGlobal>(&getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};

}

std::unique_ptr<Pass> mlir::polygeist::createConvertParallelToGPUPass() {
  return std::make_unique<ConvertParallelToGPUPass>();
}
