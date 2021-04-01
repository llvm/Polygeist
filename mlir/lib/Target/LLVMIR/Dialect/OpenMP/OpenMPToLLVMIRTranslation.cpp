//===- OpenMPToLLVMIRTranslation.cpp - Translate OpenMP dialect to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenMP dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`.
static void convertOmpOpRegions(
    Region &region, StringRef blockName, llvm::BasicBlock &sourceBlock,
    llvm::BasicBlock &continuationBlock, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation, LogicalResult &bodyGenStatus,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr,
    function_ref<LogicalResult(Operation &, llvm::IRBuilderBase &)> convertOp =
        nullptr) {
  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock.getTerminator();

  // Terminators (namely YieldOp) may be forwarding values to the region that
  // need to be available in the continuation block. Collect the types of these
  // operands in preparation of creating PHI nodes.
  SmallVector<llvm::Type *> continuationBlockPHITypes;
  bool operandsProcessed = false;
  unsigned numYields = 0;
  for (Block &bb : region.getBlocks()) {
    if (omp::YieldOp yield = dyn_cast<omp::YieldOp>(bb.getTerminator())) {
      if (!operandsProcessed) {
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          continuationBlockPHITypes.push_back(
              moduleTranslation.convertType(yield->getOperand(i).getType()));
        }
        operandsProcessed = true;
      } else {
        assert(continuationBlockPHITypes.size() == yield->getNumOperands() &&
               "mismatching number of values yielded from the region");
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          llvm::Type *operandType =
              moduleTranslation.convertType(yield->getOperand(i).getType());
          (void)operandType;
          assert(continuationBlockPHITypes[i] == operandType &&
                 "values of mismatching types yielded from the region");
        }
      }
      numYields++;
    }
  }

  // Insert PHI nodes in the continuation block for any values forwarded by the
  // terminators in this region.
  if (!continuationBlockPHITypes.empty())
    assert(
        continuationBlockPHIs &&
        "expected continuation block PHIs if converted regions yield values");
  if (continuationBlockPHIs) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    continuationBlockPHIs->reserve(continuationBlockPHITypes.size());
    builder.SetInsertPoint(&continuationBlock, continuationBlock.begin());
    for (llvm::Type *ty : continuationBlockPHITypes)
      continuationBlockPHIs->push_back(builder.CreatePHI(ty, numYields));
  }

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  llvm::SetVector<Block *> blocks =
      LLVM::detail::getTopologicallySortedBlocks(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == &continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    if (failed(moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder,
                                              convertOp))) {
      bodyGenStatus = failure();
      return;
    }

    // Special handling for `omp.yield` and `omp.terminator` (we may have more
    // than one): they return the control to the parent OpenMP dialect operation
    // so replace them with the branch to the continuation block. We handle this
    // here to avoid relying inter-function communication through the
    // ModuleTranslation class to set up the correct insertion point. This is
    // also consistent with MLIR's idiom of handling special region terminators
    // in the same code that handles the region-owning operation.
    Operation *terminator = bb->getTerminator();
    if (isa<omp::TerminatorOp, omp::YieldOp>(terminator)) {
      builder.CreateBr(&continuationBlock);

      for (unsigned i = 0, e = terminator->getNumOperands(); i < e; ++i)
        (*continuationBlockPHIs)[i]->addIncoming(
            moduleTranslation.lookupValue(terminator->getOperand(i)), llvmBB);
    }
  }
  // Finally, after all blocks have been traversed and values mapped,
  // connect the PHI nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);
}

/// Converts the OpenMP parallel operation to LLVM IR.
static LogicalResult
convertOmpParallel(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // ParallelOp has only one region associated with it.
    auto &region = cast<omp::ParallelOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.par.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                    llvm::Value &, llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;

    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::Value *ifCond = nullptr;
  if (auto ifExprVar = cast<omp::ParallelOp>(opInst).if_expr_var())
    ifCond = moduleTranslation.lookupValue(ifExprVar);
  llvm::Value *numThreads = nullptr;
  if (auto numThreadsVar = cast<omp::ParallelOp>(opInst).num_threads_var())
    numThreads = moduleTranslation.lookupValue(numThreadsVar);
  llvm::omp::ProcBindKind pbKind = llvm::omp::OMP_PROC_BIND_default;
  if (auto bind = cast<omp::ParallelOp>(opInst).proc_bind_val())
    pbKind = llvm::omp::getProcBindKind(bind.getValue());
  // TODO: Is the Parallel construct cancellable?
  bool isCancellable = false;

  // Insert allocas at the entry block of the current function.
  llvm::BasicBlock &funcEntryBlock =
      builder.GetInsertBlock()->getParent()->getEntryBlock();
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      &funcEntryBlock, funcEntryBlock.getFirstInsertionPt());

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createParallel(
      ompLoc, allocaIP, bodyGenCB, privCB, finiCB, ifCond, numThreads, pbKind,
      isCancellable));

  return bodyGenStatus;
}

/// Converts an OpenMP 'master' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMaster(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // MasterOp has only one region associated with it.
    auto &region = cast<omp::MasterOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.master.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createMaster(
      ompLoc, bodyGenCB, finiCB));
  return success();
}

namespace {
class BinOpReductionGenerator {
public:
  BinOpReductionGenerator(llvm::IRBuilderBase &b,
                          llvm::Instruction::BinaryOps opc)
      : builder(b), opcode(opc) {}
  llvm::Value *operator()(llvm::OpenMPIRBuilder::InsertPointTy insertionPoint,
                          llvm::Value *lhs, llvm::Value *rhs) {
    builder.restoreIP(insertionPoint);
    return builder.CreateBinOp(opcode, lhs, rhs);
  }

private:
  llvm::IRBuilderBase &builder;
  llvm::Instruction::BinaryOps opcode;
};

class AtomicRMWReductionGenerator {
public:
  AtomicRMWReductionGenerator(llvm::IRBuilderBase &b,
                              llvm::AtomicRMWInst::BinOp opc)
      : builder(b), opcode(opc) {}
  llvm::Value *operator()(llvm::OpenMPIRBuilder::InsertPointTy insertionPoint,
                          llvm::Value *lhsPtr, llvm::Value *rhsPtr) const {
    builder.restoreIP(insertionPoint);
    llvm::Value *rhs = builder.CreateLoad(rhsPtr);
    builder.CreateAtomicRMW(opcode, lhsPtr, rhs, llvm::None,
                            llvm::AtomicOrdering::Monotonic);
    return rhs;
  }

private:
  llvm::IRBuilderBase &builder;
  llvm::AtomicRMWInst::BinOp opcode;
};
} // namespace

static void
collectReductionInfos(Region &region,
                      SmallVectorImpl<omp::ReductionDeclareOp> &reductions) {
  for (auto reduction : region.getOps<omp::ReductionOp>()) {
    omp::ReductionDeclareOp declaration =
        SymbolTable::lookupNearestSymbolFrom<omp::ReductionDeclareOp>(
            region.getParentOp(), reduction.sym());
    assert(declaration != nullptr && "undeclared OpenMP reduction");
    reductions.push_back(declaration);
  }
}

static LogicalResult inlineConvertOmpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr,
    function_ref<LogicalResult(Operation &, llvm::IRBuilderBase &)> convertOp =
        nullptr) {
  if (region.empty())
    return success();

  // TODO(zinenko): add a special case for single-block regions that doesn't
  // create additional blocks.
  //   if (llvm::hasSingleElement(region)) {
  //     moduleTranslation.mapBlock(&region.front(), builder.GetInsertBlock());
  //     if (failed(moduleTranslation.convertBlock(
  //             region.front(), /*ignoreArguments=*/true, builder, convertOp)))
  //       return failure();

  //     continuationBlockPHIs =
  //         moduleTranslation.lookupValues(region.front().back().getOperands());

  //     return success();
  //   }

  llvm::BasicBlock *continuationBlock =
      llvm::BasicBlock::Create(builder.getContext(), blockName + ".cont",
                               builder.GetInsertBlock()->getParent(),
                               builder.GetInsertBlock()->getNextNode());
  builder.CreateBr(continuationBlock);

  LogicalResult bodyGenStatus = success();
  convertOmpOpRegions(region, blockName, *builder.GetInsertBlock(),
                      *continuationBlock, builder, moduleTranslation,
                      bodyGenStatus, continuationBlockPHIs, convertOp);
  if (failed(bodyGenStatus))
    return failure();
  builder.SetInsertPoint(continuationBlock,
                         continuationBlock->getFirstInsertionPt());
  return success();
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpWsLoop(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto loop = cast<omp::WsLoopOp>(opInst);
  // TODO: this should be in the op verifier instead.
  if (loop.lowerBound().empty())
    return failure();

  if (loop.getNumLoops() != 1)
    return opInst.emitOpError("collapsed loops not yet supported");

  if (loop.schedule_val().hasValue() &&
      omp::symbolizeClauseScheduleKind(loop.schedule_val().getValue()) !=
          omp::ClauseScheduleKind::Static)
    return opInst.emitOpError(
        "only static (default) loop schedule is currently supported");

  // Find the loop configuration.
  llvm::Value *lowerBound = moduleTranslation.lookupValue(loop.lowerBound()[0]);
  llvm::Value *upperBound = moduleTranslation.lookupValue(loop.upperBound()[0]);
  llvm::Value *step = moduleTranslation.lookupValue(loop.step()[0]);
  llvm::Type *ivType = step->getType();
  llvm::Value *chunk =
      loop.schedule_chunk_var()
          ? moduleTranslation.lookupValue(loop.schedule_chunk_var())
          : llvm::ConstantInt::get(ivType, 1);

  SmallVector<omp::ReductionDeclareOp> reductionDecls;
  collectReductionInfos(loop.region(), reductionDecls);

  // TODO: get the alloca insertion point from the parallel operation builder.
  // If we insert the at the top of the current function, they will be passed as
  // extra arguments into the function the parallel operation builder outlines.
  // Put them at the start of the current block for now.
  llvm::BasicBlock *insertBlock = builder.GetInsertBlock();
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      insertBlock, insertBlock->getFirstInsertionPt());

  // Allocate space for privatized reduction variables.
  SmallVector<llvm::Value *> privateReductionVariables;
  unsigned numReductions = loop.reduction_vars().size();
  privateReductionVariables.reserve(numReductions);
  if (numReductions != 0) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.restoreIP(allocaIP);
    for (unsigned i = 0; i < numReductions; ++i) {
      auto reductionType =
          loop.reduction_vars()[i].getType().cast<LLVM::LLVMPointerType>();
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionType.getElementType()));
      privateReductionVariables.push_back(var);
    }
  }

  unsigned numProcessedReductions = 0;
  std::function<LogicalResult(Operation &, llvm::IRBuilderBase &)> convertOp =
      [&](Operation &op, llvm::IRBuilderBase &builder) {
        if (auto reductionOp = dyn_cast<omp::ReductionOp>(op)) {
          Region &reductionRegion =
              reductionDecls[numProcessedReductions].reductionRegion();

          llvm::Value *reductionVal = builder.CreateLoad(
              privateReductionVariables[numProcessedReductions]);
          moduleTranslation.mapValue(reductionRegion.front().getArgument(0),
                                     reductionVal);
          moduleTranslation.mapValue(
              reductionRegion.front().getArgument(1),
              moduleTranslation.lookupValue(reductionOp.operand()));

          SmallVector<llvm::PHINode *> phis;
          if (failed(inlineConvertOmpRegions(
                  reductionRegion, "omp.reduction.body", builder,
                  moduleTranslation, &phis, convertOp)))
            return failure();
          assert(phis.size() == 1 && "expected one value to be yielded from "
                                     "the reduction body declaration region");
          builder.CreateStore(
              phis[0], privateReductionVariables[numProcessedReductions++]);

          return success();
        }
        return moduleTranslation.convertOperation(op, builder);
      };

  // Before the loop, store the initial values of reductions into reduction
  // variables. Although this could be done after allocas, we don't want to mess
  // up with the alloca insertion point.
  for (unsigned i = 0; i < numReductions; ++i) {
    SmallVector<llvm::PHINode *> phis;
    if (failed(inlineConvertOmpRegions(reductionDecls[i].initializerRegion(),
                                       "omp.reduction.neutral", builder,
                                       moduleTranslation, &phis, convertOp)))
      return failure();
    assert(phis.size() == 1 && "expected one value to be yielded from the "
                               "reduction neutral element declaration region");
    builder.CreateStore(phis[0], privateReductionVariables[i]);
  }

  // Set up the source location value for OpenMP runtime.
  llvm::DISubprogram *subprogram =
      builder.GetInsertBlock()->getParent()->getSubprogram();
  const llvm::DILocation *diLoc =
      moduleTranslation.translateLoc(opInst.getLoc(), subprogram);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder.saveIP(),
                                                    llvm::DebugLoc(diLoc));

  // Generator of the canonical loop body. Produces an SESE region of basic
  // blocks.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    llvm::IRBuilder<>::InsertPointGuard guard(builder);

    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(loop.getRegion().front().getArgument(0), iv);

    llvm::BasicBlock *entryBlock = ip.getBlock();
    llvm::BasicBlock *exitBlock =
        entryBlock->splitBasicBlock(ip.getPoint(), "omp.wsloop.exit");

    // Convert the body of the loop.
    convertOmpOpRegions(loop.region(), "omp.wsloop.region", *entryBlock,
                        *exitBlock, builder, moduleTranslation, bodyGenStatus,
                        nullptr, convertOp);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes WsLoop is semantically similar to SCF loop,
  // i.e. it has a positive step, uses signed integer semantics. Reconsider
  // this code when WsLoop clearly supports more cases.
  llvm::CanonicalLoopInfo *loopInfo =
      moduleTranslation.getOpenMPBuilder()->createCanonicalLoop(
          ompLoc, bodyGen, lowerBound, upperBound, step, /*IsSigned=*/true,
          /*InclusiveStop=*/loop.inclusive());
  assert(numProcessedReductions == loop.reduction_vars().size() &&
         "could not process all reductions");
  if (failed(bodyGenStatus))
    return failure();

  // TODO: get the alloca insertion point from the parallel operation builder,
  // and make sure it remains usable.
  // Re-compute the insertion point in case the iterator was invalidated above.
  allocaIP = llvm::OpenMPIRBuilder::InsertPointTy(
      insertBlock, insertBlock->getFirstInsertionPt());
  loopInfo = moduleTranslation.getOpenMPBuilder()->createStaticWorkshareLoop(
      ompLoc, loopInfo, allocaIP, !loop.nowait(), chunk);

  // Continue building IR after the loop.
  builder.restoreIP(loopInfo->getAfterIP());

  using OwningReductionGen = std::function<llvm::OpenMPIRBuilder::InsertPointTy(
      llvm::OpenMPIRBuilder::InsertPointTy, llvm::Value *, llvm::Value *,
      llvm::Value *&)>;
  SmallVector<OwningReductionGen> owningReductionGens,
      owningAtomicReductionGens;
  SmallVector<std::unique_ptr<Region>> owningBodyRegions;
  bool hasAtomic = true;
  for (unsigned i = 0; i < numReductions; ++i) {
    OwningReductionGen gen =
        [&, i](llvm::OpenMPIRBuilder::InsertPointTy insertPoint,
               llvm::Value *lhs, llvm::Value *rhs, llvm::Value *&result) {
          owningBodyRegions.emplace_back(new Region);
          Region &reductionRegion = *owningBodyRegions.back();
          BlockAndValueMapping emptyMapping;
          reductionDecls[i].reductionRegion().cloneInto(&reductionRegion,
                                                        emptyMapping);

          moduleTranslation.mapValue(reductionRegion.front().getArgument(0),
                                     lhs);
          moduleTranslation.mapValue(reductionRegion.front().getArgument(1),
                                     rhs);
          builder.restoreIP(insertPoint);
          SmallVector<llvm::PHINode *> phis;
          if (failed(inlineConvertOmpRegions(
                  reductionRegion, "omp.reduction.nonatomic.body", builder,
                  moduleTranslation, &phis, convertOp)))
            return llvm::OpenMPIRBuilder::InsertPointTy();
          assert(phis.size() == 1);
          result = phis[0];
          return builder.saveIP();
        };
    owningReductionGens.push_back(std::move(gen));

    if (reductionDecls[i].atomicReductionRegion().empty() || !hasAtomic) {
      hasAtomic = false;
      owningAtomicReductionGens.clear();
      continue;
    }

    OwningReductionGen atomicGen =
        [&, i](llvm::OpenMPIRBuilder::InsertPointTy insertPoint,
               llvm::Value *lhs, llvm::Value *rhs, llvm::Value *&) {
          Region &atomicRegion = reductionDecls[i].atomicReductionRegion();
          moduleTranslation.mapValue(atomicRegion.front().getArgument(0), lhs);
          moduleTranslation.mapValue(atomicRegion.front().getArgument(1), rhs);
          builder.restoreIP(insertPoint);
          SmallVector<llvm::PHINode *> phis;
          if (failed(inlineConvertOmpRegions(
                  atomicRegion, "omp.reduction.atomic.body", builder,
                  moduleTranslation, &phis, convertOp)))
            return llvm::OpenMPIRBuilder::InsertPointTy();
          assert(phis.empty());
          return builder.saveIP();
        };
    owningAtomicReductionGens.push_back(std::move(atomicGen));
  }

  auto reductionGens = llvm::to_vector<2>(llvm::map_range(
      owningReductionGens,
      [](const OwningReductionGen &gen)
          -> llvm::OpenMPIRBuilder::ReductionGenTy { return gen; }));
  auto atomicReductionGens = llvm::to_vector<2>(llvm::map_range(
      owningAtomicReductionGens,
      [](const OwningReductionGen &gen)
          -> llvm::OpenMPIRBuilder::ReductionGenTy { return gen; }));

  if (!loop.reduction_vars().empty()) {
    llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
        moduleTranslation.getOpenMPBuilder()->createReductions(
            builder.saveIP(), allocaIP,
            moduleTranslation.lookupValues(loop.reduction_vars()),
            privateReductionVariables, reductionGens, atomicReductionGens,
            /*IsNoWait=*/false);
    if (!contInsertPoint.getBlock())
      return loop->emitOpError() << "failed to convert reductions";
    auto nextInsertionPoint =
        moduleTranslation.getOpenMPBuilder()->createBarrier(
            contInsertPoint, llvm::omp::OMPD_for);
    builder.restoreIP(nextInsertionPoint);
  }

  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenMP dialect to LLVM IR.
class OpenMPDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // end namespace

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
LogicalResult OpenMPDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](omp::BarrierOp) {
        ompBuilder->createBarrier(builder.saveIP(), llvm::omp::OMPD_barrier);
        return success();
      })
      .Case([&](omp::TaskwaitOp) {
        ompBuilder->createTaskwait(builder.saveIP());
        return success();
      })
      .Case([&](omp::TaskyieldOp) {
        ompBuilder->createTaskyield(builder.saveIP());
        return success();
      })
      .Case([&](omp::FlushOp) {
        // No support in Openmp runtime function (__kmpc_flush) to accept
        // the argument list.
        // OpenMP standard states the following:
        //  "An implementation may implement a flush with a list by ignoring
        //   the list, and treating it the same as a flush without a list."
        //
        // The argument list is discarded so that, flush with a list is treated
        // same as a flush without a list.
        ompBuilder->createFlush(builder.saveIP());
        return success();
      })
      .Case([&](omp::ParallelOp) {
        return convertOmpParallel(*op, builder, moduleTranslation);
      })
      .Case([&](omp::MasterOp) {
        return convertOmpMaster(*op, builder, moduleTranslation);
      })
      .Case([&](omp::WsLoopOp) {
        return convertOmpWsLoop(*op, builder, moduleTranslation);
      })
      .Case<omp::YieldOp, omp::TerminatorOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        // assert(op->getNumOperands() == 0 &&
        //        "unexpected OpenMP terminator with operands");
        return success();
      })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OpenMP operation: ")
               << inst->getName();
      });
}

void mlir::registerOpenMPDialectTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addDialectInterface<omp::OpenMPDialect,
                               OpenMPDialectLLVMIRTranslationInterface>();
}

void mlir::registerOpenMPDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenMPDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
