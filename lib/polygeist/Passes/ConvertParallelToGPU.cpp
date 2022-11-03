//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

void insertReturn(PatternRewriter &rewriter, func::FuncOp f) {
  rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc());
}
void insertReturn(PatternRewriter &rewriter, LLVM::LLVMFuncOp f) {
  rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(),
                                  std::vector<Value>{});
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
    // Region *region = &f.getBody();
    // if (region->empty())
    //  return failure();
    // rewriter.eraseBlock(&region->front());
    // region->push_back(new Block());
    // rewriter.setInsertionPointToEnd(&region->front());
    // insertReturn(rewriter, f);
    return success();
  }
};

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

struct ParallelToGPULaunch
    : public OpRewritePattern<polygeist::ParallelWrapperOp> {
  using OpRewritePattern<polygeist::ParallelWrapperOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(polygeist::ParallelWrapperOp gridWrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = gridWrapper->getLoc();
    if (gridWrapper->getParentOfType<polygeist::ParallelWrapperOp>()) {
      LLVM_DEBUG(DBGS() << "[pop-to-launch] ignoring nested parallel op\n");
      return failure();
    }
    rewriter.setInsertionPoint(gridWrapper);
    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Add back optimized away single iter parallel ops
    auto insertSingleIterPop = [&](polygeist::ParallelWrapperOp wrapper,
                                   scf::ParallelOp &pop) {
      Block *block = wrapper.getBody();
      rewriter.eraseOp(block->getTerminator());
      rewriter.setInsertionPointToEnd(wrapper.getBody());
      wrapper.getBodyRegion().push_front(new Block());
      rewriter.setInsertionPointToStart(wrapper.getBody());
      pop = rewriter.create<scf::ParallelOp>(
          loc, std::vector<Value>({zeroindex}), std::vector<Value>({oneindex}),
          std::vector<Value>({oneindex}));
      rewriter.create<polygeist::PolygeistYieldOp>(loc);
      rewriter.setInsertionPointToStart(pop.getBody());
      rewriter.mergeBlockBefore(block, pop.getBody()->getTerminator());
    };

    // TODO we currently assume that all parallel ops we encouter are already
    // prepared for conversion to gpu.launch, i.e. two nested parallel loops
    // with lower bounds zero and constant upper bounds for the inner parallel,
    // the memory they use is on the gpu, is there more?
    scf::ParallelOp gridPop = nullptr;
    for (auto &op : *gridWrapper.getBody())
      if (auto cast = dyn_cast<scf::ParallelOp>(&op))
        gridPop = cast;
      else if (auto cast = dyn_cast<AffineParallelOp>(&op)) {
        LLVM_DEBUG(DBGS() << "[pop-to-launch] need to lower affine parallel ops before this pass\n");
        return failure();
      }
    if (!gridPop)
      rewriter.updateRootInPlace(
          gridWrapper, [&] { insertSingleIterPop(gridWrapper, gridPop); });
    polygeist::ParallelWrapperOp blockWrapper;
    for (auto &op : *gridPop.getBody())
      if (auto cast = dyn_cast<polygeist::ParallelWrapperOp>(&op))
        blockWrapper = cast;
    scf::ParallelOp blockPop;
    for (auto &op : *blockWrapper.getBody())
      if (auto cast = dyn_cast<scf::ParallelOp>(&op))
        blockPop = cast;
      else if (auto cast = dyn_cast<AffineParallelOp>(&op)) {
        LLVM_DEBUG(DBGS() << "[pop-to-launch] need to lower affine parallel ops before this pass\n");
        return failure();
      }
    if (!blockPop)
      rewriter.updateRootInPlace(
          blockWrapper, [&] { insertSingleIterPop(blockWrapper, blockPop); });

    rewriter.setInsertionPoint(blockWrapper);
    rewriter.eraseOp(blockWrapper.getBody()->getTerminator());
    rewriter.mergeBlockBefore(blockWrapper.getBody(), blockWrapper);
    rewriter.eraseOp(blockWrapper);
    rewriter.setInsertionPoint(gridWrapper);
    rewriter.eraseOp(gridWrapper.getBody()->getTerminator());
    rewriter.mergeBlockBefore(gridWrapper.getBody(), gridWrapper);
    rewriter.eraseOp(gridWrapper);

    // Move operations outside the blockPop inside it
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
        auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                    /* memspace */ 5);
        auto newAlloca =
            rewriter.create<memref::AllocaOp>(alloca.getLoc(), type);
        mapping.map(op.getResults(), newAlloca->getResults());
        auto cast = rewriter.create<memref::CastOp>(
            alloca.getLoc(), alloca.getType(), newAlloca);
        newOp = cast;
      } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
        LLVM_DEBUG(DBGS() << "[pop-to-launch] llvm alloca op shared mem\n");
        return failure();
      } else {
        newOp = rewriter.clone(op, mapping);
      }
      rewriter.replaceOpWithinBlock(&op, newOp->getResults(),
                                    blockPop.getBody());
    }

    auto getUpperBounds = [&](scf::ParallelOp pop) -> SmallVector<Value, 3> {
      SmallVector<Value, 3> bounds;
      for (auto bound : pop.getUpperBound()) {
        bounds.push_back(bound);
      }
      return bounds;
    };

    // TODO make sure we start at zero or else convert the parallel ops to start
    // at 0
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
    auto launchOp = rewriter.create<gpu::LaunchOp>(
        loc, gridBounds[0], gridBounds[1], gridBounds[2], blockBounds[0],
        blockBounds[1], blockBounds[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ nullptr,
        /*dependencies*/ SmallVector<Value, 1>());

    auto getDim = [](unsigned index) {
      // TODO what should the order be
      if (index == 0)
        return gpu::Dimension::x;
      if (index == 1)
        return gpu::Dimension::y;
      if (index == 2)
        return gpu::Dimension::z;
      assert(0 && "Invalid index");
    };

    auto launchBlock = &launchOp.getRegion().front();
    rewriter.setInsertionPointToStart(launchBlock);
    SmallVector<Value, 3> argReplacements;
    for (auto en : llvm::enumerate(blockPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto blockIdx = rewriter.create<gpu::ThreadIdOp>(
          loc, mlir::IndexType::get(rewriter.getContext()), dim);
      argReplacements.push_back(blockIdx);
    }
    rewriter.mergeBlocks(blockPop.getBody(), launchBlock, argReplacements);
    rewriter.setInsertionPointToStart(launchBlock);

    for (auto en : llvm::enumerate(gridPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto gridIdx = rewriter.create<gpu::BlockIdOp>(
          loc, mlir::IndexType::get(rewriter.getContext()), dim);
      en.value().replaceAllUsesWith(gridIdx);
      argReplacements.push_back(gridIdx);
    }

    // I _think_ replacing these with gpu.block_dim is good because the block
    // dims might already be in registers as opposed to constants of other
    // values TODO investigate
    for (auto en : llvm::enumerate(popBlockBounds)) {
      gpu::Dimension dim = getDim(en.index());
      auto blockDim = rewriter.create<gpu::BlockDimOp>(
          loc, mlir::IndexType::get(rewriter.getContext()), dim);
      rewriter.updateRootInPlace(launchOp, [&] {
        en.value().replaceUsesWithIf(blockDim, [&](OpOperand &operand) {
          return launchOp->isProperAncestor(operand.getOwner());
        });
      });
    }
    for (auto en : llvm::enumerate(popGridBounds)) {
      gpu::Dimension dim = getDim(en.index());
      auto gridDim = rewriter.create<gpu::GridDimOp>(
          loc, mlir::IndexType::get(rewriter.getContext()), dim);
      rewriter.updateRootInPlace(launchOp, [&] {
        en.value().replaceUsesWithIf(gridDim, [&](OpOperand &operand) {
          return launchOp->isProperAncestor(operand.getOwner());
        });
      });
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
        // assert(args == barrierArgs);
      }
      barrier = b;
      // barrierArgs = args;
    });
    return success();
  }
};

struct ConvertParallelToGPU1Pass
    : public ConvertParallelToGPU1Base<ConvertParallelToGPU1Pass> {
  ConvertParallelToGPU1Pass() {}
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

struct ConvertParallelToGPU2Pass
    : public ConvertParallelToGPU2Base<ConvertParallelToGPU2Pass> {
  ConvertParallelToGPU2Pass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SharedLLVMAllocaToGlobal, SharedMemrefAllocaToGlobal,
                    RemoveFunction<func::FuncOp>, RemoveFunction<LLVM::LLVMFuncOp>>(
        &getContext());
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
