//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"

#define DEBUG_TYPE "convert-parallel-to-gpu"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace polygeist;

namespace {

struct SharedLLVMAllocaToGlobal : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto PT = ao.getType().cast<LLVM::LLVMPointerType>();
    if (PT.getAddressSpace() != 5) {
      return failure();
    }

    auto type = PT.getElementType();
    auto loc = ao->getLoc();
    auto name = "shared_mem_" + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!module) {
      return failure();
    }

    rewriter.setInsertionPointToStart(module.getBody());

    auto globalOp = rewriter.create<LLVM::GlobalOp>(
        loc, type, /* isConstant */ false, LLVM::Linkage::Internal, name,
        mlir::Attribute(),
        /* alignment */ 0, /* addrSpace */ 3);
    rewriter.setInsertionPoint(ao);
    auto aoo = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);

    rewriter.replaceOp(ao, aoo->getResults());

    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (mt.getMemorySpaceAsInt() != 5) {
      return failure();
    }

    auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                /* memspace */ 3);
    auto loc = ao->getLoc();
    auto name = "shared_mem_" + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!module) {
      return failure();
    }

    rewriter.setInsertionPointToStart(module.getBody());

    auto initial_value = rewriter.getUnitAttr();
    auto globalOp = rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name),
        /* sym_visibility */ mlir::StringAttr(), mlir::TypeAttr::get(type),
        initial_value, mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

struct ParallelToGPULaunch : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp gridPop,
                                PatternRewriter &rewriter) const override {
    auto loc = gridPop->getLoc();
    if (gridPop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << "[pop-to-launch] ignoring nested parallel op\n");
      return failure();
    }
    // TODO we currently assume that all parallel ops we encouter
    // are in directly nested pairs and do no checks wheteher they can be
    // gpuified or whether the memory they use is actually on the gpu
    scf::ParallelOp blockPop;
    gridPop.getBody()->walk([&](scf::ParallelOp b) {
      blockPop = b;
    });

    polygeist::ParallelWrapperOp gridWrapper = dyn_cast<polygeist::ParallelWrapperOp>(gridPop->getParentOp());
    polygeist::ParallelWrapperOp blockWrapper = dyn_cast<polygeist::ParallelWrapperOp>(blockPop->getParentOp());
    if (!gridWrapper || !blockWrapper) {
      LLVM_DEBUG(DBGS() << "[pop-to-launch] currently only lower parallel ops that were lowered from gpu launch\n");
      return failure();
    }
    assert(blockWrapper->getParentOp() == gridPop && "Block parallel op wrapper must be directly nested in the grid parallel op\n");

    rewriter.setInsertionPoint(blockWrapper);
    rewriter.mergeBlockBefore(blockWrapper.getBody(), blockWrapper);
    rewriter.eraseOp(blockWrapper);
    rewriter.setInsertionPoint(gridWrapper);
    rewriter.mergeBlockBefore(gridWrapper.getBody(), gridWrapper);
    rewriter.eraseOp(gridWrapper);

    // TODO check properties of parallel loops that we need to be able to
    // convert them to gpu: start from 0, no variable blockPop upper bounds,
    // etc?

    // Move operations outisde the blockPop inside it
    rewriter.setInsertionPointToStart(blockPop.getBody());
    BlockAndValueMapping mapping;
    for (Operation &op : *gridPop.getBody()) {
      Operation *newOp;
      if (isa<scf::ParallelOp>(&op)) {
        continue;
      } else if (isa<scf::YieldOp>(&op)) {
        continue;
      } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
        auto mt = alloca.getType();
        auto type = MemRefType::get(mt.getShape(), mt.getElementType(),
                                    {}, /* memspace */ 5);
        auto newAlloca = rewriter.create<memref::AllocaOp>(alloca.getLoc(), type);
        mapping.map(op.getResults(), newAlloca->getResults());
        auto cast = rewriter.create<memref::CastOp>(alloca.getLoc(), alloca.getType(), newAlloca);
        newOp = cast;
      } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
        LLVM_DEBUG(DBGS() << "[pop-to-launch] llvm alloca op shared mem\n");
        return failure();
      } else {
        newOp = rewriter.clone(op, mapping);
      }
      rewriter.replaceOpWithinBlock(&op, newOp->getResults(), blockPop.getBody());
    }

    auto getUpperBounds = [&](scf::ParallelOp pop) -> SmallVector<Value, 3> {
      SmallVector<Value, 3> bounds;
      for (auto bound : pop.getUpperBound()) {
        bounds.push_back(bound);
      }
      return bounds;
    };

    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // TODO make sure we start at zero or else convert the parallel ops to start at 0
    Value gridBounds[3];
    auto popGridBounds = getUpperBounds(gridPop);
    for (unsigned int i = 0; i < 3; i++) {
      if (i < popGridBounds.size())
        gridBounds[i] = popGridBounds[i];
      else
        gridBounds[i] = oneindex;
    }
    Value blockBounds[3];
    auto popBlockBounds = getUpperBounds(blockPop);
    for (unsigned int i = 0; i < 3; i++) {
      if (i < popBlockBounds.size())
        blockBounds[i] = popBlockBounds[i];
      else
        blockBounds[i] = oneindex;
    }

    // TODO handle stream and dependencies - we would have to convert an
    // async{parallel {parallel {}}} to a gpu.launch
    // TODO handle dyn shmem
    rewriter.setInsertionPoint(gridPop);
    auto launchOp = rewriter.create<gpu::LaunchOp>(loc,
                                                   gridBounds[0], gridBounds[1], gridBounds[2],
                                                   blockBounds[0], blockBounds[1], blockBounds[2],
                                                   /*dynamic shmem size*/ nullptr,
                                                   /*token type*/ nullptr,
                                                   /*dependencies*/ SmallVector<Value, 1>());

    auto getDim = [](unsigned index) {
      // TODO what should the order be
      if (index == 0) return gpu::Dimension::x;
      if (index == 1) return gpu::Dimension::y;
      if (index == 2) return gpu::Dimension::z;
      assert(0 && "Invalid index");
    };

    auto launchBlock = &launchOp.getRegion().front();
    rewriter.setInsertionPointToStart(launchBlock);
    SmallVector<Value, 3> argReplacements;
    for (auto en : llvm::enumerate(blockPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto blockIdx = rewriter.create<gpu::ThreadIdOp>(loc, mlir::IndexType::get(rewriter.getContext()), dim);
      argReplacements.push_back(blockIdx);
    }
    rewriter.mergeBlocks(blockPop.getBody(), launchBlock, argReplacements);
    rewriter.setInsertionPointToStart(launchBlock);

    for (auto en : llvm::enumerate(gridPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto gridIdx = rewriter.create<gpu::BlockIdOp>(loc, mlir::IndexType::get(rewriter.getContext()), dim);
      en.value().replaceAllUsesWith(gridIdx);
      argReplacements.push_back(gridIdx);
    }

    // TODO need a way to figure out which value is actually used as a dim, e.g.
    // have a wrapper op that just passes its block and grid dim operands to the
    // block args in it but we still want to retain replacing of constant dims
    // with a constant
    for (auto en : llvm::enumerate(popBlockBounds)) {
      Operation *op = en.value().getDefiningOp();
      if (detail::isConstantLike(op))
        continue;
      assert(op->getNumResults() == 1 && "We do not currently handle ops with multiple results");

      gpu::Dimension dim = getDim(en.index());
      auto blockDim = rewriter.create<gpu::BlockDimOp>(loc, mlir::IndexType::get(rewriter.getContext()), dim);
      rewriter.replaceOpWithinBlock(op, ValueRange({blockDim}), launchBlock);
    }
    for (auto en : llvm::enumerate(popGridBounds)) {
      Operation *op = en.value().getDefiningOp();
      if (detail::isConstantLike(op))
        continue;
      assert(op->getNumResults() == 1 && "We do not currently handle ops with multiple results");

      gpu::Dimension dim = getDim(en.index());
      auto gridDim = rewriter.create<gpu::GridDimOp>(loc, mlir::IndexType::get(rewriter.getContext()), dim);
      rewriter.replaceOpWithinBlock(op, ValueRange({gridDim}), launchBlock);
    }

    rewriter.setInsertionPointToEnd(launchBlock);
    rewriter.create<gpu::TerminatorOp>(loc);

    rewriter.eraseOp(gridPop);

    Operation *yieldOp = nullptr;
    for (auto &op : *launchBlock) {
      if (auto y = dyn_cast<scf::YieldOp>(&op)) {
        assert(!yieldOp && "Multiple yields in the final block? why?");
        yieldOp = y;
      }
    }
    rewriter.eraseOp(yieldOp);

    launchBlock->walk([&](mlir::polygeist::BarrierOp op) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<mlir::NVVM::Barrier0Op>(op);
    });

    return success();

    polygeist::BarrierOp barrier = nullptr;
    std::vector<BlockArgument> barrierArgs;
    gridPop->walk([&](polygeist::BarrierOp b) {
      // TODO maybe do some barrier checks here, but for now we just assume
      // verything is fine and is generated from gpu code
      auto args = b->getOpOperands();
      if (barrier) {
        //assert(args == barrierArgs);
      }
      barrier = b;
      //barrierArgs = args;
    });
    return success();
  }
};

struct ConvertParallelToGPU1Pass
    : public ConvertParallelToGPU1Base<ConvertParallelToGPU1Pass> {
  ConvertParallelToGPU1Pass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SharedLLVMAllocaToGlobal, SharedMemrefAllocaToGlobal>(
        &getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
      return;
    }
  }
};

struct ConvertParallelToGPU2Pass
    : public ConvertParallelToGPU2Base<ConvertParallelToGPU2Pass> {
  ConvertParallelToGPU2Pass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ParallelToGPULaunch>(&getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::polygeist::createConvertParallelToGPUPass1() {
  return std::make_unique<ConvertParallelToGPU1Pass>();
}
std::unique_ptr<Pass> mlir::polygeist::createConvertParallelToGPUPass2() {
  return std::make_unique<ConvertParallelToGPU2Pass>();
}
