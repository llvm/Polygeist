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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"

#define DEBUG_TYPE "convert-parallel-to-gpu"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

using namespace mlir;
using namespace polygeist;

namespace {

template <typename T>
bool hasEffect(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (auto it : effects)
    if (isa<T>(it.getEffect()))
      return true;
  return false;
}

template <int S = 3> SmallVector<Value, S> getUpperBounds(scf::ParallelOp pop) {
  SmallVector<Value, S> bounds;
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

/// TODO implement code motion across the gpu_wrapper, then we would have
/// two options for gpu_wrappers without any parallel ops in them - we
/// could either hoist the computation to the cpu with added cpu-gpu copies or
/// we could run a single iteration gpu kernel - whichever we think might be
/// better for each case
///
/// gpu_wrapper {
///   A()
/// }
/// ->
/// gpu_wrapper {
///   parallel _ = 0 to 1 {
///     parallel _ = 0 to 1 {
///       A()
///     }
///   }
/// }
///
struct CreateParallelOps : public OpRewritePattern<polygeist::GPUWrapperOp> {
  using OpRewritePattern<polygeist::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "create-parallel-ops";
  LogicalResult matchAndRewrite(polygeist::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    scf::ParallelOp pop = nullptr;
    for (Operation &op : *wrapper.getBody()) {
      if (auto p = dyn_cast<scf::ParallelOp>(&op)) {
        pop = p;
      }
    }
    if (pop) {
      LLVM_DEBUG(DBGS() << "parallel already exists\n");
      return failure();
    }
    auto loc = wrapper->getLoc();
    auto terminator = wrapper.getBody()->getTerminator();
    rewriter.setInsertionPoint(wrapper);
    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 1> one(1, oneindex);
    SmallVector<Value, 1> zero(1, zeroindex);
    rewriter.setInsertionPointToEnd(wrapper.getBody());
    auto gridPop = rewriter.create<scf::ParallelOp>(loc, zero, one, one);
    rewriter.clone(*terminator);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop = rewriter.create<scf::ParallelOp>(loc, zero, one, one);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    SmallVector<Operation *> toErase;
    BlockAndValueMapping mapping;
    for (Operation &op : *wrapper.getBody()) {
      toErase.push_back(&op);
      if (terminator == &op)
        break;
      rewriter.clone(op, mapping);
    }
    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

    return success();
  }
};

/// TODO Look or any barriers and if they are we must preserve the threads it
/// syncs at to be block threads
///
/// parallel {
///   A()
/// }
///
/// ->
///
/// parallel {
///   parallel {
///     A()
///   }
/// }
///
///
/// Splitting an iteration variable to grid/block one:
/// parallel i = 0 to i_bound {
///   A(i)
/// }
/// ->
/// parallel i = 0 to i_bound / BLOCK_SIZE {
///   parallel j = 0 to BLOCK_SIZE {
///     A(i * BLOCK_SIZE + j)
///   }
/// }
///
/// Making iteration variables with constant bounds block iteration variables:
/// parallel i = 0 to var_i_bound, j = 0 to const_j_bound {
///   A(i, j)
/// }
/// ->
/// parallel i = 0 to var_i_bound {
///   parallel j = 0 to const_j_bound {
///     A(i, j)
///   }
/// }
///
struct SplitParallelOp : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
  const char *PATTERN = "parallelize-block-ops";
  LogicalResult matchAndRewrite(scf::ParallelOp pop,
                                PatternRewriter &rewriter) const override {
    auto wrapper = dyn_cast<polygeist::GPUWrapperOp>(pop->getParentOp());
    if (!wrapper) {
      LLVM_DEBUG(DBGS() << "parallel not wrapped\n");
      return failure();
    }
    if (pop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << "only single parallel ops\n");
      return failure();
    }
    bool child = false;
    pop->walk([&](scf::ParallelOp p) {
      if (pop != p)
        child = true;
    });
    if (child) {
      LLVM_DEBUG(DBGS() << "only single parallel ops\n");
      return failure();
    }
    auto loc = pop->getLoc();

    // TODO handle lower bounds != 0 and steps != 1

    auto upperBounds = getUpperBounds<6>(pop);
    int totalDims = upperBounds.size();
    SmallVector<Value, 3> blockDims;
    SmallVector<Value, 3> gridDims;
    // Arg ids in the original parallel block
    SmallVector<int, 3> blockArgId;
    SmallVector<int, 3> gridArgId;

    const unsigned maxThreads = 1024;
    unsigned threadNum = 1;
    for (int i = totalDims - 1; i >= 0; i--) {
      // TODO have we covered all possibilities for a constant? maybe use
      // ->hasTrait<OpTrait::ConstantLike>
      auto &bound = upperBounds[i];
      auto cst =
          dyn_cast_or_null<arith::ConstantIndexOp>(bound.getDefiningOp());
      unsigned val = cst ? cst.value() : 1;
      if (cst && blockDims.size() < 3 && threadNum * val <= maxThreads) {
        blockDims.insert(blockDims.begin(), bound);
        blockArgId.insert(blockArgId.begin(), i);
        threadNum *= val;
      } else {
        gridDims.insert(gridDims.begin(), bound);
        gridArgId.insert(gridArgId.begin(), i);
      }
    }
    // TODO if we have too many dims, we have to merge some of them
    assert(gridDims.size() <= 3);

    rewriter.setInsertionPoint(pop);
    auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    unsigned splitDims = 0;
    SmallVector<int, 3> gi;
    SmallVector<int, 3> bi;
    if (gridDims.size() == 0) {
      gridDims.push_back(oneindex);
      // Put a random index, we will override it
      gridArgId.push_back(0);
    } else if (threadNum < 256) {
      // If we are not getting enough parallelism in the block, use part of the
      // grid dims
      //
      // TODO we have to be careful to not exceed max z dimension in block, it
      // is lower than the 1024 max for the x and y

      unsigned threadsLeft = 1;
      while ((float)threadsLeft <= (float)maxThreads / (float)threadNum)
        threadsLeft *= 2;
      threadsLeft /= 2;
      threadNum *= threadsLeft;
      assert(threadNum <= maxThreads);

      // TODO what should the number here be
      // How many dims to take from the grid
      splitDims = 1;
      assert(splitDims <= gridDims.size());
      assert(splitDims + blockDims.size() <= 3);

      // Which grid dims to take
      for (unsigned i = 0; i < splitDims; i++)
        gi.push_back(gridDims.size() - 1 - i);
      // Which block dims they correspond to
      for (unsigned i = 0; i < splitDims; i++) {
        bi.push_back(i);
        blockArgId.insert(blockArgId.begin(), gridArgId[gi[i]]);
      }

      SmallVector<Value, 3> newBlockDims;
      // TODO try our best to make them divisors of the gridDims
      rewriter.setInsertionPoint(pop);
      if (splitDims == 1)
        newBlockDims = {
            rewriter.create<arith::ConstantIndexOp>(loc, threadsLeft),
        };
      else if (splitDims == 2)
        // TODO
        assert(0);
      else if (splitDims == 3)
        // TODO
        assert(0);
      else
        assert(0);
      newBlockDims.insert(newBlockDims.end(), blockDims.begin(),
                          blockDims.end());

      for (unsigned i = 0; i < splitDims; i++) {
        // newGridDims[j] = ((gridDims[j] - 1) / newBlockDims[i]) + 1;
        auto sub =
            rewriter.create<arith::SubIOp>(loc, gridDims[gi[i]], oneindex);
        auto div =
            rewriter.create<arith::DivUIOp>(loc, sub, newBlockDims[bi[i]]);
        gridDims[gi[i]] = rewriter.create<arith::AddIOp>(loc, div, oneindex);
      }
      blockDims = newBlockDims;
    }

    // LLVM_DEBUG(DBGS() <<
    llvm::errs() << "converting to block with threadNum: " << threadNum
                 << ", dims: " << blockDims.size() << "\n";

    SmallVector<Value, 3> lowerBoundsGrid(gridDims.size(), zeroindex);
    SmallVector<Value, 3> stepsGrid(gridDims.size(), oneindex);
    SmallVector<Value, 3> lowerBoundsBlock(blockDims.size(), zeroindex);
    SmallVector<Value, 3> stepsBlock(blockDims.size(), oneindex);

    rewriter.setInsertionPoint(pop);
    auto gridPop = rewriter.create<scf::ParallelOp>(loc, lowerBoundsGrid,
                                                    gridDims, stepsGrid);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop = rewriter.create<scf::ParallelOp>(loc, lowerBoundsBlock,
                                                     blockDims, stepsBlock);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    BlockAndValueMapping mapping;
    for (unsigned i = 0; i < gridDims.size(); i++)
      mapping.map(pop.getBody()->getArgument(gridArgId[i]),
                  gridPop.getBody()->getArgument(i));
    for (unsigned i = 0; i < blockDims.size(); i++)
      mapping.map(pop.getBody()->getArgument(blockArgId[i]),
                  blockPop.getBody()->getArgument(i));

    // For the split dims, calculate the equivalent threadId and map that
    // instead
    if (splitDims > 0) {
      Value cond;
      // SmallVector<Value, 3> threadId(splitDims);
      for (unsigned i = 0; i < splitDims; i++) {
        // threadIndex = blockIdx * blockDim + threadIdx
        // threadIndex < original upperBound
        //
        // Currently we do not care if the split dim correspond to the same
        // block/thread index, so we might do something like blockIdx.x *
        // blockDim.x + threadIdx.y, should we try to rearrange dims to match
        // them?
        auto mul = rewriter.create<arith::MulIOp>(
            loc, gridPop.getBody()->getArgument(gi[i]), blockDims[bi[i]]);
        auto threadId = rewriter.create<arith::AddIOp>(
            loc, mul, blockPop.getBody()->getArgument(bi[i]));
        assert(blockArgId[bi[i]] == gridArgId[gi[i]]);
        mapping.map(pop.getBody()->getArgument(gridArgId[gi[i]]), threadId);
        auto threadCond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, threadId,
            upperBounds[gridArgId[gi[i]]]);
        if (i == 0)
          cond = threadCond.getResult();
        else
          cond =
              rewriter.create<arith::AndIOp>(loc, threadCond, cond).getResult();
      }
      auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
    }

    rewriter.eraseOp(pop.getBody()->getTerminator());
    for (auto &op : *pop.getBody())
      rewriter.clone(op, mapping);
    rewriter.eraseOp(pop);

    return success();
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

  const char *PATTERN = "parallelize-block-ops";
  LogicalResult matchAndRewrite(scf::ParallelOp pop,
                                PatternRewriter &rewriter) const override {
    if (!pop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << "ignoring non nested parallel op\n");
      return failure();
    }
    auto loc = pop->getLoc();
    Block *outerBlock = pop->getBlock();
    Block *innerBlock = pop.getBody();

    if (++ ++outerBlock->begin() == outerBlock->end()) {
      LLVM_DEBUG(DBGS() << "no ops to parallelize\n");
      return failure();
    }

    // Handle ops before the parallel
    scf::IfOp ifOp = nullptr;
    auto getIf = [&]() {
      if (!ifOp) {
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value cond;
        for (unsigned i = 0; i < innerBlock->getNumArguments(); i++) {
          auto threadId = innerBlock->getArgument(i);
          auto threadCond = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, threadId, zero);
          if (i == 0)
            cond = threadCond.getResult();
          else
            cond = rewriter.create<arith::AndIOp>(loc, threadCond, cond)
                       .getResult();
        }
        ifOp = rewriter.create<scf::IfOp>(loc, cond);
        rewriter.create<mlir::polygeist::BarrierOp>(loc,
                                                    innerBlock->getArguments());
        rewriter.setInsertionPoint(ifOp);
      }
    };

    rewriter.setInsertionPointToStart(innerBlock);
    auto it = outerBlock->begin();
    SmallVector<Operation *> toErase;
    BlockAndValueMapping mapping;
    for (; &*it != pop.getOperation(); ++it) {
      Operation &op = *it;
      Operation *newOp;
      if (isa<scf::ParallelOp>(&op)) {
        assert(0 && "Unhandled case");
        break;
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
        assert(0 && "Unhandled case");
        break;
      } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        SmallVector<MemoryEffects::EffectInstance> effects;
        collectEffects(&op, effects, /*ignoreBarriers*/ false);
        if (effects.empty()) {
        } else if (hasEffect<MemoryEffects::Allocate>(effects)) {
          assert(0 && "??");
        } else if (hasEffect<MemoryEffects::Free>(effects)) {
          assert(0 && "??");
        } else if (hasEffect<MemoryEffects::Write>(effects)) {
          getIf();
          assert(ifOp);
          rewriter.setInsertionPoint(ifOp.thenBlock()->getTerminator());
          // TODO currently we assume that ops with write effects will have no
          // uses - we have to introduce shared mem otherwise
          assert(op.use_empty() && "Unhandled case");
        } else if (hasEffect<MemoryEffects::Read>(effects)) {
          // Reads-only ops are legal to parallelize
        }
        newOp = rewriter.clone(op, mapping);
      }
      rewriter.replaceOpWithinBlock(&op, newOp->getResults(), innerBlock);
      toErase.push_back(&op);
    }
    it++;

    // Handle ops after the parallel
    {
      auto zeroindex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.setInsertionPoint(innerBlock->getTerminator());
      auto cmpOp = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, zeroindex, innerBlock->getArgument(0));
      Value condition = cmpOp.getResult();
      for (unsigned i = 1; i < innerBlock->getNumArguments(); i++) {
        auto cmpOp2 = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, zeroindex,
            innerBlock->getArgument(i));
        auto andOp = rewriter.create<arith::AndIOp>(loc, condition, cmpOp2);
        condition = andOp.getResult();
      }
      auto ifOp = rewriter.create<scf::IfOp>(loc, condition);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      for (; it != outerBlock->end(); ++it) {
        Operation &op = *it;
        if (isa<scf::ParallelOp>(&op)) {
          assert(0 && "Unhandled case");
          break;
        } else if (isa<scf::YieldOp>(&op)) {
          continue;
        } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
          assert(0 && "Unhandled case");
          break;
        } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
          assert(0 && "Unhandled case");
          break;
        } else {
          rewriter.clone(op, mapping);
        }
        toErase.push_back(&op);
      }
    }

    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

    return success();
  }
};

/// gpu_wrapper {
///   A()
///   parallel {
///     ...
///   }
///   ...
/// }
/// ->
/// A1()
/// gpu_wrapper {
///   A2()
/// }
/// gpu_wrapper {
///   parallel {
///     A3()
///     ...
///   }
///   ...
/// }
struct HandleWrapperRootOps : public OpRewritePattern<polygeist::GPUWrapperOp> {
  using OpRewritePattern<polygeist::GPUWrapperOp>::OpRewritePattern;

  const char *PATTERN = "split-off-parallel";
  LogicalResult matchAndRewrite(polygeist::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto wrapperBody = wrapper.getBody();
    auto it = wrapperBody->begin();
    if (isa<scf::ParallelOp>(&*it)) {
      LLVM_DEBUG(DBGS() << "first op is a parellel\n");
      return failure();
    }
    SmallVector<Operation *> toHandle;
    scf::ParallelOp pop;
    Operation *firstGridOp;
    for (;; ++it) {
      if (&*it == wrapperBody->getTerminator())
        return failure();
      if (auto p = dyn_cast<scf::ParallelOp>(&*it)) {
        pop = p;
        firstGridOp = &*pop.getBody()->begin();
        break;
      }
      toHandle.push_back(&*it);
    }
    if (toHandle.size() == 0) {
      LLVM_DEBUG(DBGS() << "empty wrapper\n");
      return failure();
    }
    rewriter.setInsertionPoint(wrapper);
    auto newWrapper = rewriter.create<polygeist::GPUWrapperOp>(loc);
    BlockAndValueMapping hoistMapping;
    BlockAndValueMapping splitMapping;
    BlockAndValueMapping parallelizedMapping;
    for (Operation *op : toHandle) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      collectEffects(op, effects, /*ignoreBarriers*/ false);
      bool read = hasEffect<MemoryEffects::Read>(effects);
      bool write = hasEffect<MemoryEffects::Write>(effects);
      Operation *cloned;
      if (effects.empty()) {
        rewriter.setInsertionPoint(firstGridOp);
        rewriter.clone(*op, parallelizedMapping);
        rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
        rewriter.clone(*op, splitMapping);
        rewriter.setInsertionPoint(newWrapper);
        cloned = rewriter.clone(*op, hoistMapping);
      } else if (hasEffect<MemoryEffects::Allocate>(effects)) {
        // I think this can actually happen if we lower a kernel with a barrier
        // and shared memory with gridDim = 1 TODO handle
        cloned = nullptr;
        assert(0 && "what?");
      } else if (hasEffect<MemoryEffects::Free>(effects)) {
        cloned = nullptr;
        assert(0 && "what?");
      } else if (write) {
        rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
        cloned = rewriter.clone(*op, splitMapping);
        // TODO if we find that this has results that get used later in the
        // final parallel we need to introduce temporary gpu cache memory to
        // pass it on
      } else if (read) {
        // Check if we can safely put the read in the grid parallel op, i.e. the
        // ops up to and including the next parallel op may not write to where
        // we read from

        // TODO for nested ops, try to collect all memrefs we load from and do
        // the checks on them
        bool canParallelize = true;
        if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          auto loadMemRef = loadOp.getMemref();
          Operation *op = loadOp;
          while (op != pop.getOperation()) {
            op = op->getNextNode();
            if (mayWriteTo(op, loadMemRef, /*ignoreBarrier*/ false)) {
              canParallelize = false;
            }
          }
        } else {
          canParallelize = false;
        }
        if (canParallelize) {
          rewriter.setInsertionPoint(firstGridOp);
          cloned = rewriter.clone(*op, parallelizedMapping);
        } else {
          // TODO we need to introduce temporary gpu cache memory to pass it on
          assert(0 && "unhandled case");
          cloned = nullptr;
        }
      } else {
        cloned = nullptr;
        assert(0 && "are there other effects?");
      }
      rewriter.replaceOpWithIf(op, cloned->getResults(), [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        while (owner->getBlock() != pop->getBlock())
          owner = owner->getParentOp();
        return pop->getPrevNode()->isBeforeInBlock(owner);
      });
    }
    for (Operation *op : llvm::reverse(toHandle)) {
      assert(op->use_empty());
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// gpu_wrapper {
///   parallel {
///     ...
///   }
///   A()
/// }
/// ->
/// gpu_wrapper {
///   parallel {
///     ...
///   }
/// }
/// gpu_wrapper {
///   A()
/// }
struct SplitOffParallel : public OpRewritePattern<polygeist::GPUWrapperOp> {
  using OpRewritePattern<polygeist::GPUWrapperOp>::OpRewritePattern;

  const char *PATTERN = "split-off-parallel";
  LogicalResult matchAndRewrite(polygeist::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto pop = dyn_cast<scf::ParallelOp>(&(*wrapper.getBody()->begin()));
    if (!pop) {
      LLVM_DEBUG(DBGS() << "first op is not a parellel\n");
      return failure();
    }
    if (pop->getNextNode() == wrapper.getBody()->getTerminator()) {
      LLVM_DEBUG(DBGS() << "pop is the only op in the block\n");
      return failure();
    }
    assert(pop->getNumResults() == 0);

    rewriter.setInsertionPoint(wrapper);
    auto newWrapper = rewriter.create<polygeist::GPUWrapperOp>(loc);
    rewriter.setInsertionPointToStart(newWrapper.getBody());
    rewriter.clone(*pop.getOperation());
    rewriter.eraseOp(pop);
    return success();
  }
};

/// gpu_wrapper {
///   parallel grid_bounds {
///     parallel block_bounds {
///       A()
///     }
///   }
/// }
/// ->
/// gpu.launch grid_bounds, block_bounds {
///   A()
/// }
struct ParallelToGPULaunch : public OpRewritePattern<polygeist::GPUWrapperOp> {
  using OpRewritePattern<polygeist::GPUWrapperOp>::OpRewritePattern;

  const char *PATTERN = "parallel-to-gpu-launch";
  LogicalResult matchAndRewrite(polygeist::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    if (wrapper->getParentOfType<polygeist::GPUWrapperOp>()) {
      LLVM_DEBUG(DBGS() << "[pop-to-launch] ignoring nested parallel op\n");
      return failure();
    }
    rewriter.setInsertionPoint(wrapper);
    auto oneindex = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // TODO we currently assume that all parallel ops we encouter are already
    // prepared for conversion to gpu.launch, i.e. two nested parallel loops
    // with lower bounds zero and constant upper bounds for the inner parallel,
    // the memory they use is on the gpu, are there more conditions?
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
        getDirectlyNestedSingleParallel(wrapper.getBody());
    if (!gridPop)
      return failure();
    scf::ParallelOp blockPop =
        getDirectlyNestedSingleParallel(gridPop.getBody());
    if (!blockPop)
      return failure();

    rewriter.setInsertionPoint(wrapper);
    rewriter.eraseOp(wrapper.getBody()->getTerminator());
    rewriter.mergeBlockBefore(wrapper.getBody(), wrapper);
    rewriter.eraseOp(wrapper);

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

    // polygeist::BarrierOp barrier = nullptr;
    // std::vector<BlockArgument> barrierArgs;
    // gridPop->walk([&](polygeist::BarrierOp b) {
    //   // TODO maybe do some barrier checks here, but for now we just assume
    //   // verything is fine and is generated from gpu code
    //   auto args = b->getOpOperands();
    //   if (barrier) {
    //     // assert(args == barrierArgs);
    //   }
    //   barrier = b;
    //   // barrierArgs = args;
    // });
    // return success();
  }
};

// TODO put single constants in the loop
// TODO parallel wrapper LICM
struct ConvertParallelToGPU1Pass
    : public ConvertParallelToGPU1Base<ConvertParallelToGPU1Pass> {
  ConvertParallelToGPU1Pass() {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.insert<
      BarrierElim</*TopLevelOnly*/ false>,
      CreateParallelOps,
      SplitOffParallel,
      HandleWrapperRootOps,
      ParallelizeBlockOps,
      SplitParallelOp,
      ParallelToGPULaunch
      // TODO insert constants into body
      >(&getContext());
    // clang-format on
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
