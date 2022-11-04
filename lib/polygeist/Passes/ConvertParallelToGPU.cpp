//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

SmallVector<Value, 3> getUpperBounds(scf::ParallelOp pop) {
  SmallVector<Value, 3> bounds;
  for (auto bound : pop.getUpperBound()) {
    bounds.push_back(bound);
  }
  return bounds;
}

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
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name),
        /* sym_visibility */ mlir::StringAttr(), mlir::TypeAttr::get(type),
        initial_value, mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

struct SplitParallelOp : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
#define PATTERN "[parallelize-block-ops] "
  LogicalResult matchAndRewrite(scf::ParallelOp pop,
                                PatternRewriter &rewriter) const override {
    auto wrapper = dyn_cast<polygeist::ParallelWrapperOp>(pop->getParentOp());
    if (!wrapper) {
      LLVM_DEBUG(DBGS() << PATTERN << "parallel not wrapped\n");
      return failure();
    }
    if (pop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << PATTERN << "only single parallel ops\n");
      return failure();
    }
    bool child = false;
    pop->walk([&](scf::ParallelOp p) { if (pop != p) child = true; });
    if (child) {
      LLVM_DEBUG(DBGS() << PATTERN << "only single parallel ops\n");
      return failure();
    }
    auto loc = pop->getLoc();

    // TODO handle lower bounds != 0 and steps != 1

    auto upperBounds = getUpperBounds(pop);
    int totalDims = upperBounds.size();
    SmallVector<Value, 3> blockDims;
    SmallVector<Value, 3> gridDims;

    const unsigned maxThreads = 1024;
    int threadNum = 1;
    for (unsigned i = totalDims - 1; i >= 0; i--) {
      // TODO should be any constant
      auto &bound = upperBounds[i];
      auto cst = dyn_cast<arith::ConstantIndexOp>(bound.getDefiningOp());
      unsigned val = cst ? cst.value() : 1;
      if (cst && blockDims.size() < 3 && threadNum * val <= maxThreads) {
        blockDims.push_back(bound);
        threadNum *= val;
      } else {
        gridDims.push_back(bound);
      }
    }
    //LLVM_DEBUG(DBGS() <<
    llvm::errs() << PATTERN << "converting to block with threadNum " << threadNum << ", dims: " << blockDims.size() << "\n";

    // TODO if the threadNums are not enough, increase them by converting a grid
    // dim to a grid dim + block dim

    // TODO if we have too many dims, we have to merge some of them
    assert(gridDims.size() <= 3);

    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 3> lowerBoundsGrid(gridDims.size(), zeroindex);
    SmallVector<Value, 3> stepsGrid(gridDims.size(), oneindex);
    SmallVector<Value, 3> lowerBoundsBlock(blockDims.size(), zeroindex);
    SmallVector<Value, 3> stepsBlock(blockDims.size(), oneindex);

    rewriter.setInsertionPoint(pop);
    auto gridPop =
        rewriter.create<scf::ParallelOp>(loc, lowerBoundsGrid, gridDims, stepsGrid);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop =
        rewriter.create<scf::ParallelOp>(loc, lowerBoundsBlock, blockDims, stepsBlock);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    BlockAndValueMapping mapping;
    for (int i = 0; i < gridDims.size(); i++)
      mapping.map(gridDims[i], gridPop.getBody()->getArgument(i));
    for (int i = 0; i < blockDims.size(); i++)
      mapping.map(blockDims[i], blockPop.getBody()->getArgument(i));
    rewriter.eraseOp(pop.getBody()->getTerminator());
    for (auto &op : *pop.getBody())
      rewriter.clone(op, mapping);
    rewriter.eraseOp(pop);
    return success();


    // Below an example of how we could do one grid dim -> grid dim + block dim
    // conversion
    {
    int dims = 3;





    // TODO try our best to match some of the dims or make them divisors of the
    // dims
    rewriter.setInsertionPoint(wrapper);
    if (dims == 1)
      blockDims = {
          rewriter.create<arith::ConstantIndexOp>(loc, 1024),
      };
    else if (dims == 2)
      blockDims = {
          rewriter.create<arith::ConstantIndexOp>(loc, 32),
          rewriter.create<arith::ConstantIndexOp>(loc, 32),
      };
    else
      blockDims = {
          rewriter.create<arith::ConstantIndexOp>(loc, 16),
          rewriter.create<arith::ConstantIndexOp>(loc, 16),
          rewriter.create<arith::ConstantIndexOp>(loc, 4),
      };

    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 3> gridDims(dims);
    for (int i = 0; i < dims; i++) {
      // gridDims[i] = ((upperBounds[i] - 1) / blockDims[i]) + 1;
      auto sub = rewriter.create<arith::SubIOp>(loc, upperBounds[i], oneindex);
      auto div = rewriter.create<arith::DivUIOp>(loc, sub, blockDims[i]);
      gridDims[i] = rewriter.create<arith::AddIOp>(loc, div, oneindex);
    }
    auto lowerBounds = pop.getLowerBound();
    auto steps = pop.getStep();

    rewriter.setInsertionPoint(pop);
    auto gridPop =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, gridDims, steps);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, blockDims, steps);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    Value cond;
    SmallVector<Value, 3> threadId(dims);
    for (int i = 0; i < dims; i++) {
      // threadIndex = blockIdx * blockDim + threadIdx
      // threadIndex < original upperBound
      auto mul = rewriter.create<arith::MulIOp>(
          loc, gridPop.getBody()->getArgument(i), blockDims[i]);
      auto add = rewriter.create<arith::AddIOp>(
          loc, mul, blockPop.getBody()->getArgument(i));
      threadId[i] = add;
      auto threadCond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, add, upperBounds[i]);
      if (i == 0)
        cond = threadCond.getResult();
      else
        cond =
            rewriter.create<arith::AndIOp>(loc, threadCond, cond).getResult();
    }

    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    BlockAndValueMapping mapping;
    for (int i = 0; i < dims; i++)
      mapping.map(pop.getBody()->getArgument(i), threadId[i]);
    rewriter.eraseOp(pop.getBody()->getTerminator());
    for (auto &op : *pop.getBody())
      rewriter.clone(op, mapping);
    rewriter.eraseOp(pop);
    return success();
    }
  }
};

// TODO handle something like this if it happens
//
// scf.parallel {
//   scf.parallel {
//     A()
//   }
//   scf.parallel {
//     B()
//   }
// }
//

/// scf.parallel {
///   A()
///   scf.parallel {
///     B()
///   }
///   C()
/// }
///
/// ->
///
/// scf.parallel {
///   scf.parallel {
///     A'()
///     barrier
///     B()
///     barrier
///     C'()
///   }
/// }

struct ParallelizeBlockOps : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

#define PATTERN "[parallelize-block-ops] "
  LogicalResult matchAndRewrite(scf::ParallelOp pop,
                                PatternRewriter &rewriter) const override {
    if (!pop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << PATTERN << "ignoring non nested parallel op\n");
      return failure();
    }
    auto loc = pop->getLoc();
    Block *outerBlock = pop->getBlock();
    Block *innerBlock = pop.getBody();

    if (++ ++outerBlock->begin() == outerBlock->end()) {
      LLVM_DEBUG(DBGS() << PATTERN << "no ops to parallelize\n");
      return failure();
    }

    // Handle ops before the parallel
    //
    // TODO We currently assume that there are no ops with memory effects before
    // the pop
    rewriter.setInsertionPointToStart(innerBlock);
    auto it = outerBlock->begin();
    SmallVector<Operation *> toErase;
    BlockAndValueMapping mapping;
    for (; &*it != pop.getOperation(); ++it) {
      Operation &op = *it;
      Operation *newOp;
      if (isa<scf::ParallelOp>(&op)) {
        assert(0 && "Unhandled case"); break;
      } else if (isa<scf::YieldOp>(&op)) {
        continue;
      } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
        auto mt = alloca.getType();
        auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                    /* memspace */ 5);
        auto newAlloca =
            rewriter.create<memref::AllocaOp>(alloca.getLoc(), type);
        auto cast = rewriter.create<memref::CastOp>(
            alloca.getLoc(), alloca.getType(), newAlloca);
        mapping.map(op.getResults(), cast->getResults());
        newOp = cast;
      } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
        assert(0 && "Unhandled case"); break;
      } else {
        // TODO Consider write memory effects here - we must put them in an if
        // to execute only once
        //
        // TODO How can we even handle an op that does write and read and its
        // result is used in the parallel op? Introduce shared mem I guess?
        newOp = rewriter.clone(op, mapping);
      }
      rewriter.replaceOpWithinBlock(&op, newOp->getResults(), innerBlock);
      toErase.push_back(&op);
    }
    it++;

    // Handle ops after the parallel
    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.setInsertionPoint(innerBlock->getTerminator());
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, zeroindex, innerBlock->getArgument(0));
    Value condition = cmpOp.getResult();
    for (unsigned i = 1; i < innerBlock->getNumArguments(); i++) {
      auto cmpOp2 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, zeroindex, innerBlock->getArgument(i));
      auto andOp = rewriter.create<arith::AndIOp>(loc, condition, cmpOp2);
      condition = andOp.getResult();
    }
    auto ifOp = rewriter.create<scf::IfOp>(loc, condition);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    for (; it != outerBlock->end(); ++it) {
      Operation &op = *it;
      if (isa<scf::ParallelOp>(&op)) {
        assert(0 && "Unhandled case"); break;
      } else if (isa<scf::YieldOp>(&op)) {
        continue;
      } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
        assert(0 && "Unhandled case"); break;
      } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
        assert(0 && "Unhandled case"); break;
      } else {
        rewriter.clone(op, mapping);
      }
      toErase.push_back(&op);
    }

    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

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
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // TODO we currently assume that all parallel ops we encouter are already
    // prepared for conversion to gpu.launch, i.e. two nested parallel loops
    // with lower bounds zero and constant upper bounds for the inner parallel,
    // the memory they use is on the gpu, is there more?
    auto getDirectlyNestedSingleParallel =
        [&](Block *block) -> scf::ParallelOp {
      auto it = block->begin();
      auto pop = dyn_cast<scf::ParallelOp>(&*it);
      it++;
      if (!pop) {
        LLVM_DEBUG(
            DBGS() << "[pop-to-launch] need directly nested parallelop\n");
        return nullptr;
      }
      if (block->getTerminator() != &*it) {
        LLVM_DEBUG(DBGS() << "[pop-to-launch] stray ops in block\n");
        return nullptr;
      }
      it++;
      assert(it == block->end());
      return pop;
    };

    scf::ParallelOp gridPop =
        getDirectlyNestedSingleParallel(gridWrapper.getBody());
    if (!gridPop)
      return failure();
    scf::ParallelOp blockPop =
        getDirectlyNestedSingleParallel(gridPop.getBody());
    if (!blockPop)
      return failure();

    rewriter.setInsertionPoint(gridWrapper);
    rewriter.eraseOp(gridWrapper.getBody()->getTerminator());
    rewriter.mergeBlockBefore(gridWrapper.getBody(), gridWrapper);
    rewriter.eraseOp(gridWrapper);

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
      return gpu::Dimension::z;
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

    //polygeist::BarrierOp barrier = nullptr;
    //std::vector<BlockArgument> barrierArgs;
    //gridPop->walk([&](polygeist::BarrierOp b) {
    //  // TODO maybe do some barrier checks here, but for now we just assume
    //  // verything is fine and is generated from gpu code
    //  auto args = b->getOpOperands();
    //  if (barrier) {
    //    // assert(args == barrierArgs);
    //  }
    //  barrier = b;
    //  // barrierArgs = args;
    //});
    //return success();
  }
};

struct ConvertParallelToGPU1Pass
    : public ConvertParallelToGPU1Base<ConvertParallelToGPU1Pass> {
  ConvertParallelToGPU1Pass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<BarrierElim</*TopLevelOnly*/ false>, ParallelizeBlockOps,
                    SplitParallelOp, ParallelToGPULaunch>(&getContext());
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
    patterns
        .insert<SharedLLVMAllocaToGlobal, SharedMemrefAllocaToGlobal,
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
