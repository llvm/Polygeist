//===- ModuleTranslation.cpp - MLIR to LLVM conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR LLVM dialect module and
// the corresponding LLVMIR module. It only handles core LLVM IR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "DebugTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

/// Builds a constant of a sequential LLVM type `type`, potentially containing
/// other sequential types recursively, from the individual constant values
/// provided in `constants`. `shape` contains the number of elements in nested
/// sequential types. Reports errors at `loc` and returns nullptr on error.
static llvm::Constant *
buildSequentialConstant(ArrayRef<llvm::Constant *> &constants,
                        ArrayRef<int64_t> shape, llvm::Type *type,
                        Location loc) {
  if (shape.empty()) {
    llvm::Constant *result = constants.front();
    constants = constants.drop_front();
    return result;
  }

  llvm::Type *elementType;
  if (auto *arrayTy = dyn_cast<llvm::ArrayType>(type)) {
    elementType = arrayTy->getElementType();
  } else if (auto *vectorTy = dyn_cast<llvm::VectorType>(type)) {
    elementType = vectorTy->getElementType();
  } else {
    emitError(loc) << "expected sequential LLVM types wrapping a scalar";
    return nullptr;
  }

  SmallVector<llvm::Constant *, 8> nested;
  nested.reserve(shape.front());
  for (int64_t i = 0; i < shape.front(); ++i) {
    nested.push_back(buildSequentialConstant(constants, shape.drop_front(),
                                             elementType, loc));
    if (!nested.back())
      return nullptr;
  }

  if (shape.size() == 1 && type->isVectorTy())
    return llvm::ConstantVector::get(nested);
  return llvm::ConstantArray::get(
      llvm::ArrayType::get(elementType, shape.front()), nested);
}

/// Returns the first non-sequential type nested in sequential types.
static llvm::Type *getInnermostElementType(llvm::Type *type) {
  do {
    if (auto *arrayTy = dyn_cast<llvm::ArrayType>(type)) {
      type = arrayTy->getElementType();
    } else if (auto *vectorTy = dyn_cast<llvm::VectorType>(type)) {
      type = vectorTy->getElementType();
    } else {
      return type;
    }
  } while (true);
}

/// Create an LLVM IR constant of `llvmType` from the MLIR attribute `attr`.
/// This currently supports integer, floating point, splat and dense element
/// attributes and combinations thereof. Also, an array attribute with two
/// elements is supported to represent a complex constant.  In case of error,
/// report it to `loc` and return nullptr.
llvm::Constant *mlir::LLVM::detail::getLLVMConstant(
    llvm::Type *llvmType, Attribute attr, Location loc,
    const ModuleTranslation &moduleTranslation, bool isTopLevel) {
  if (!attr)
    return llvm::UndefValue::get(llvmType);
  if (auto *structType = dyn_cast<::llvm::StructType>(llvmType)) {
    if (!isTopLevel) {
      emitError(loc, "nested struct types are not supported in constants");
      return nullptr;
    }
    auto arrayAttr = attr.cast<ArrayAttr>();
    llvm::Type *elementType = structType->getElementType(0);
    llvm::Constant *real = getLLVMConstant(elementType, arrayAttr[0], loc,
                                           moduleTranslation, false);
    if (!real)
      return nullptr;
    llvm::Constant *imag = getLLVMConstant(elementType, arrayAttr[1], loc,
                                           moduleTranslation, false);
    if (!imag)
      return nullptr;
    return llvm::ConstantStruct::get(structType, {real, imag});
  }
  // For integer types, we allow a mismatch in sizes as the index type in
  // MLIR might have a different size than the index type in the LLVM module.
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return llvm::ConstantInt::get(
        llvmType,
        intAttr.getValue().sextOrTrunc(llvmType->getIntegerBitWidth()));
  if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    if (llvmType !=
        llvm::Type::getFloatingPointTy(llvmType->getContext(),
                                       floatAttr.getValue().getSemantics())) {
      emitError(loc, "FloatAttr does not match expected type of the constant");
      return nullptr;
    }
    return llvm::ConstantFP::get(llvmType, floatAttr.getValue());
  }
  if (auto funcAttr = attr.dyn_cast<FlatSymbolRefAttr>())
    return llvm::ConstantExpr::getBitCast(
        moduleTranslation.lookupFunction(funcAttr.getValue()), llvmType);
  if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
    llvm::Type *elementType;
    uint64_t numElements;
    if (auto *arrayTy = dyn_cast<llvm::ArrayType>(llvmType)) {
      elementType = arrayTy->getElementType();
      numElements = arrayTy->getNumElements();
    } else {
      auto *vectorTy = cast<llvm::FixedVectorType>(llvmType);
      elementType = vectorTy->getElementType();
      numElements = vectorTy->getNumElements();
    }
    // Splat value is a scalar. Extract it only if the element type is not
    // another sequence type. The recursion terminates because each step removes
    // one outer sequential type.
    bool elementTypeSequential =
        isa<llvm::ArrayType, llvm::VectorType>(elementType);
    llvm::Constant *child = getLLVMConstant(
        elementType,
        elementTypeSequential ? splatAttr : splatAttr.getSplatValue(), loc,
        moduleTranslation, false);
    if (!child)
      return nullptr;
    if (llvmType->isVectorTy())
      return llvm::ConstantVector::getSplat(
          llvm::ElementCount::get(numElements, /*Scalable=*/false), child);
    if (llvmType->isArrayTy()) {
      auto *arrayType = llvm::ArrayType::get(elementType, numElements);
      SmallVector<llvm::Constant *, 8> constants(numElements, child);
      return llvm::ConstantArray::get(arrayType, constants);
    }
  }

  if (auto elementsAttr = attr.dyn_cast<ElementsAttr>()) {
    assert(elementsAttr.getType().hasStaticShape());
    assert(elementsAttr.getNumElements() != 0 &&
           "unexpected empty elements attribute");
    assert(!elementsAttr.getType().getShape().empty() &&
           "unexpected empty elements attribute shape");

    SmallVector<llvm::Constant *, 8> constants;
    constants.reserve(elementsAttr.getNumElements());
    llvm::Type *innermostType = getInnermostElementType(llvmType);
    for (auto n : elementsAttr.getValues<Attribute>()) {
      constants.push_back(
          getLLVMConstant(innermostType, n, loc, moduleTranslation, false));
      if (!constants.back())
        return nullptr;
    }
    ArrayRef<llvm::Constant *> constantsRef = constants;
    llvm::Constant *result = buildSequentialConstant(
        constantsRef, elementsAttr.getType().getShape(), llvmType, loc);
    assert(constantsRef.empty() && "did not consume all elemental constants");
    return result;
  }

  if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
    return llvm::ConstantDataArray::get(
        moduleTranslation.getLLVMContext(),
        ArrayRef<char>{stringAttr.getValue().data(),
                       stringAttr.getValue().size()});
  }
  emitError(loc, "unsupported constant value");
  return nullptr;
}

ModuleTranslation::ModuleTranslation(Operation *module,
                                     std::unique_ptr<llvm::Module> llvmModule)
    : mlirModule(module), llvmModule(std::move(llvmModule)),
      debugTranslation(
          std::make_unique<DebugTranslation>(module, *this->llvmModule)),
      typeTranslator(this->llvmModule->getContext()),
      iface(module->getContext()) {
  assert(satisfiesLLVMModule(mlirModule) &&
         "mlirModule should honor LLVM's module semantics.");
}
ModuleTranslation::~ModuleTranslation() {
  if (ompBuilder)
    ompBuilder->finalize();
}

/// Get the SSA value passed to the current block from the terminator operation
/// of its predecessor.
static Value getPHISourceValue(Block *current, Block *pred,
                               unsigned numArguments, unsigned index) {
  Operation &terminator = *pred->getTerminator();
  if (isa<LLVM::BrOp>(terminator))
    return terminator.getOperand(index);

  SuccessorRange successors = terminator.getSuccessors();
  assert(std::adjacent_find(successors.begin(), successors.end()) ==
             successors.end() &&
         "successors with arguments in LLVM branches must be different blocks");
  (void)successors;

  // For instructions that branch based on a condition value, we need to take
  // the operands for the branch that was taken.
  if (auto condBranchOp = dyn_cast<LLVM::CondBrOp>(terminator)) {
    // For conditional branches, we take the operands from either the "true" or
    // the "false" branch.
    return condBranchOp.getSuccessor(0) == current
               ? condBranchOp.trueDestOperands()[index]
               : condBranchOp.falseDestOperands()[index];
  }

  if (auto switchOp = dyn_cast<LLVM::SwitchOp>(terminator)) {
    // For switches, we take the operands from either the default case, or from
    // the case branch that was taken.
    if (switchOp.defaultDestination() == current)
      return switchOp.defaultOperands()[index];
    for (auto i : llvm::enumerate(switchOp.caseDestinations()))
      if (i.value() == current)
        return switchOp.getCaseOperands(i.index())[index];
  }

  llvm_unreachable("only branch or switch operations can be terminators of a "
                   "block that has successors");
}

/// Connect the PHI nodes to the results of preceding blocks.
void mlir::LLVM::detail::connectPHINodes(Region &region,
                                         const ModuleTranslation &state) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (auto it = std::next(region.begin()), eit = region.end(); it != eit;
       ++it) {
    Block *bb = &*it;
    llvm::BasicBlock *llvmBB = state.lookupBlock(bb);
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (auto *pred : bb->getPredecessors()) {
        // Find the LLVM IR block that contains the converted terminator
        // instruction and use it in the PHI node. Note that this block is not
        // necessarily the same as state.lookupBlock(pred), some operations
        // (in particular, OpenMP operations using OpenMPIRBuilder) may have
        // split the blocks.
        llvm::Instruction *terminator =
            state.lookupBranch(pred->getTerminator());
        assert(terminator && "missing the mapping for a terminator");
        phiNode.addIncoming(
            state.lookupValue(getPHISourceValue(bb, pred, numArguments, index)),
            terminator->getParent());
      }
    }
  }
}

/// Sort function blocks topologically.
SetVector<Block *>
mlir::LLVM::detail::getTopologicallySortedBlocks(Region &region) {
  // For each block that has not been visited yet (i.e. that has no
  // predecessors), add it to the list as well as its successors.
  SetVector<Block *> blocks;
  for (Block &b : region) {
    if (blocks.count(&b) == 0) {
      llvm::ReversePostOrderTraversal<Block *> traversal(&b);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == region.getBlocks().size() &&
         "some blocks are not sorted");

  return blocks;
}

llvm::Value *mlir::LLVM::detail::createIntrinsicCall(
    llvm::IRBuilderBase &builder, llvm::Intrinsic::ID intrinsic,
    ArrayRef<llvm::Value *> args, ArrayRef<llvm::Type *> tys) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic, tys);
  return builder.CreateCall(fn, args);
}

/// Convert the OpenMP parallel Operation to LLVM IR.
LogicalResult
ModuleTranslation::convertOmpParallel(Operation &opInst,
                                      llvm::IRBuilder<> &builder) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // ParallelOp has only one region associated with it.
    auto &region = cast<omp::ParallelOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.par.region", valueMapping, blockMapping,
                        *codeGenIP.getBlock(), continuationBlock, builder,
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
    ifCond = valueMapping.lookup(ifExprVar);
  llvm::Value *numThreads = nullptr;
  if (auto numThreadsVar = cast<omp::ParallelOp>(opInst).num_threads_var())
    numThreads = valueMapping.lookup(numThreadsVar);
  llvm::omp::ProcBindKind pbKind = llvm::omp::OMP_PROC_BIND_default;
  if (auto bind = cast<omp::ParallelOp>(opInst).proc_bind_val())
    pbKind = llvm::omp::getProcBindKind(bind.getValue());
  // TODO: Is the Parallel construct cancellable?
  bool isCancellable = false;
  // TODO: Determine the actual alloca insertion point, e.g., the function
  // entry or the alloca insertion point as provided by the body callback
  // above.
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(builder.saveIP());
  if (failed(bodyGenStatus))
    return failure();
  builder.restoreIP(
      ompBuilder->createParallel(builder, allocaIP, bodyGenCB, privCB, finiCB,
                                 ifCond, numThreads, pbKind, isCancellable));
  return success();
}

void ModuleTranslation::convertOmpOpRegions(
    Region &region, StringRef blockName,
    DenseMap<Value, llvm::Value *> &valueMapping,
    DenseMap<Block *, llvm::BasicBlock *> &blockMapping,
    llvm::BasicBlock &sourceBlock, llvm::BasicBlock &continuationBlock,
    llvm::IRBuilder<> &builder, LogicalResult &bodyGenStatus) {
  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent());
    blockMapping[&bb] = llvmBB;
  }

  llvm::Instruction *sourceTerminator = sourceBlock.getTerminator();

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  llvm::SetVector<Block *> blocks = topologicalSort(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = blockMapping[bb];
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == &continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilder<>::InsertPointGuard guard(builder);
    if (failed(convertBlock(*bb, bb->isEntryBlock(), builder))) {
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
    if (isa<omp::TerminatorOp, omp::YieldOp>(bb->getTerminator()))
      builder.CreateBr(&continuationBlock);
  }
  // Finally, after all blocks have been traversed and values mapped,
  // connect the PHI nodes to the results of preceding blocks.
  connectPHINodes(region, valueMapping, blockMapping, branchMapping);
}

LogicalResult ModuleTranslation::convertOmpMaster(Operation &opInst,
                                                  llvm::IRBuilder<> &builder) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // MasterOp has only one region associated with it.
    auto &region = cast<omp::MasterOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.master.region", valueMapping, blockMapping,
                        *codeGenIP.getBlock(), continuationBlock, builder,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  builder.restoreIP(ompBuilder->createMaster(builder, bodyGenCB, finiCB));
  return success();
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
LogicalResult ModuleTranslation::convertOmpWsLoop(Operation &opInst,
                                                  llvm::IRBuilder<> &builder) {
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
  llvm::Value *lowerBound = valueMapping.lookup(loop.lowerBound()[0]);
  llvm::Value *upperBound = valueMapping.lookup(loop.upperBound()[0]);
  llvm::Value *step = valueMapping.lookup(loop.step()[0]);
  llvm::Type *ivType = step->getType();
  llvm::Value *chunk = loop.schedule_chunk_var()
                           ? valueMapping[loop.schedule_chunk_var()]
                           : llvm::ConstantInt::get(ivType, 1);

  // Set up the source location value for OpenMP runtime.
  llvm::DISubprogram *subprogram =
      builder.GetInsertBlock()->getParent()->getSubprogram();
  const llvm::DILocation *diLoc =
      debugTranslation->translateLoc(opInst.getLoc(), subprogram);
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
    valueMapping[loop.getRegion().front().getArgument(0)] = iv;

    llvm::BasicBlock *entryBlock = ip.getBlock();
    llvm::BasicBlock *exitBlock =
        entryBlock->splitBasicBlock(ip.getPoint(), "omp.wsloop.exit");

    // Convert the body of the loop.
    convertOmpOpRegions(loop.region(), "omp.wsloop.region", valueMapping,
                        blockMapping, *entryBlock, *exitBlock, builder,
                        bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes WsLoop is semantically similar to SCF loop,
  // i.e. it has a positive step, uses signed integer semantics. Reconsider
  // this code when WsLoop clearly supports more cases.
  llvm::BasicBlock *insertBlock = builder.GetInsertBlock();
  llvm::CanonicalLoopInfo *loopInfo = ompBuilder->createCanonicalLoop(
      ompLoc, bodyGen, lowerBound, upperBound, step, /*IsSigned=*/true,
      /*InclusiveStop=*/loop.inclusive());
  if (failed(bodyGenStatus))
    return failure();

  // TODO: get the alloca insertion point from the parallel operation builder.
  // If we insert the at the top of the current function, they will be passed as
  // extra arguments into the function the parallel operation builder outlines.
  // Put them at the start of the current block for now.
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      insertBlock, insertBlock->getFirstInsertionPt());
  loopInfo = ompBuilder->createStaticWorkshareLoop(ompLoc, loopInfo, allocaIP,
                                                   !loop.nowait(), chunk);
  // Continue building IR after the loop.
  builder.restoreIP(loopInfo->getAfterIP());
  return success();
}

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
LogicalResult
ModuleTranslation::convertOmpOperation(Operation &opInst,
                                       llvm::IRBuilder<> &builder) {
  if (!ompBuilder) {
    ompBuilder = std::make_unique<llvm::OpenMPIRBuilder>(*llvmModule);
    ompBuilder->initialize();
  }
  return llvm::TypeSwitch<Operation *, LogicalResult>(&opInst)
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
      .Case(
          [&](omp::ParallelOp) { return convertOmpParallel(opInst, builder); })
      .Case([&](omp::MasterOp) { return convertOmpMaster(opInst, builder); })
      .Case([&](omp::WsLoopOp) { return convertOmpWsLoop(opInst, builder); })
      .Case<omp::YieldOp, omp::TerminatorOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OpenMP terminator with operands");
        return success();
      })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OpenMP operation: ")
               << inst->getName();
      });
}

static llvm::FastMathFlags getFastmathFlags(FastmathFlagsInterface &op) {
  using llvmFMF = llvm::FastMathFlags;
  using FuncT = void (llvmFMF::*)(bool);
  const std::pair<FastmathFlags, FuncT> handlers[] = {
      // clang-format off
      {FastmathFlags::nnan,     &llvmFMF::setNoNaNs},
      {FastmathFlags::ninf,     &llvmFMF::setNoInfs},
      {FastmathFlags::nsz,      &llvmFMF::setNoSignedZeros},
      {FastmathFlags::arcp,     &llvmFMF::setAllowReciprocal},
      {FastmathFlags::contract, &llvmFMF::setAllowContract},
      {FastmathFlags::afn,      &llvmFMF::setApproxFunc},
      {FastmathFlags::reassoc,  &llvmFMF::setAllowReassoc},
      {FastmathFlags::fast,     &llvmFMF::setFast},
      // clang-format on
  };
  llvm::FastMathFlags ret;
  auto fmf = op.fastmathFlags();
  for (auto it : handlers)
    if (bitEnumContains(fmf, it.first))
      (ret.*(it.second))(true);
  return ret;
>>>>>>> 163d5a28486e... [mlir] Add translation of omp.wsloop to LLVM IR
>>>>>>> 83082d9b9834... [mlir] Add translation of omp.wsloop to LLVM IR
}

/// Given a single MLIR operation, create the corresponding LLVM IR operation
/// using the `builder`.
LogicalResult
ModuleTranslation::convertOperation(Operation &op,
                                    llvm::IRBuilderBase &builder) {
  const LLVMTranslationDialectInterface *opIface = iface.getInterfaceFor(&op);
  if (!opIface)
    return op.emitError("cannot be converted to LLVM IR: missing "
                        "`LLVMTranslationDialectInterface` registration for "
                        "dialect for op: ")
           << op.getName();

  if (failed(opIface->convertOperation(&op, builder, *this)))
    return op.emitError("LLVM Translation failed for operation: ")
           << op.getName();

  return convertDialectAttributes(&op);
}

/// Convert block to LLVM IR.  Unless `ignoreArguments` is set, emit PHI nodes
/// to define values corresponding to the MLIR block arguments.  These nodes
/// are not connected to the source basic blocks, which may not exist yet.  Uses
/// `builder` to construct the LLVM IR. Expects the LLVM IR basic block to have
/// been created for `bb` and included in the block mapping.  Inserts new
/// instructions at the end of the block and leaves `builder` in a state
/// suitable for further insertion into the end of the block.
LogicalResult ModuleTranslation::convertBlock(Block &bb, bool ignoreArguments,
                                              llvm::IRBuilderBase &builder) {
  builder.SetInsertPoint(lookupBlock(&bb));
  auto *subprogram = builder.GetInsertBlock()->getParent()->getSubprogram();

  // Before traversing operations, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (auto arg : bb.getArguments()) {
      auto wrappedType = arg.getType();
      if (!isCompatibleType(wrappedType))
        return emitError(bb.front().getLoc(),
                         "block argument does not have an LLVM type");
      llvm::Type *type = convertType(wrappedType);
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      mapValue(arg, phi);
    }
  }

  // Traverse operations.
  for (auto &op : bb) {
    // Set the current debug location within the builder.
    builder.SetCurrentDebugLocation(
        debugTranslation->translateLoc(op.getLoc(), subprogram));

    if (failed(convertOperation(op, builder)))
      return failure();
  }

  return success();
}

/// A helper method to get the single Block in an operation honoring LLVM's
/// module requirements.
static Block &getModuleBody(Operation *module) {
  return module->getRegion(0).front();
}

/// A helper method to decide if a constant must not be set as a global variable
/// initializer.
static bool shouldDropGlobalInitializer(llvm::GlobalValue::LinkageTypes linkage,
                                        llvm::Constant *cst) {
  return (linkage == llvm::GlobalVariable::ExternalLinkage &&
          isa<llvm::UndefValue>(cst)) ||
         linkage == llvm::GlobalVariable::ExternalWeakLinkage;
}

/// Create named global variables that correspond to llvm.mlir.global
/// definitions.
LogicalResult ModuleTranslation::convertGlobals() {
  for (auto op : getModuleBody(mlirModule).getOps<LLVM::GlobalOp>()) {
    llvm::Type *type = convertType(op.getType());
    llvm::Constant *cst = llvm::UndefValue::get(type);
    if (op.getValueOrNull()) {
      // String attributes are treated separately because they cannot appear as
      // in-function constants and are thus not supported by getLLVMConstant.
      if (auto strAttr = op.getValueOrNull().dyn_cast_or_null<StringAttr>()) {
        cst = llvm::ConstantDataArray::getString(
            llvmModule->getContext(), strAttr.getValue(), /*AddNull=*/false);
        type = cst->getType();
      } else if (!(cst = getLLVMConstant(type, op.getValueOrNull(), op.getLoc(),
                                         *this))) {
        return failure();
      }
    }

    auto linkage = convertLinkageToLLVM(op.linkage());
    auto addrSpace = op.addr_space();
    auto *var = new llvm::GlobalVariable(
        *llvmModule, type, op.constant(), linkage,
        shouldDropGlobalInitializer(linkage, cst) ? nullptr : cst,
        op.sym_name(),
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal, addrSpace);

    if (op.unnamed_addr().hasValue())
      var->setUnnamedAddr(convertUnnamedAddrToLLVM(*op.unnamed_addr()));

    if (op.section().hasValue())
      var->setSection(*op.section());

    Optional<uint64_t> alignment = op.alignment();
    if (alignment.hasValue())
      var->setAlignment(llvm::MaybeAlign(alignment.getValue()));

    globalsMapping.try_emplace(op, var);
  }

  // Convert global variable bodies. This is done after all global variables
  // have been created in LLVM IR because a global body may refer to another
  // global or itself. So all global variables need to be mapped first.
  for (auto op : getModuleBody(mlirModule).getOps<LLVM::GlobalOp>()) {
    if (Block *initializer = op.getInitializerBlock()) {
      llvm::IRBuilder<> builder(llvmModule->getContext());
      for (auto &op : initializer->without_terminator()) {
        if (failed(convertOperation(op, builder)) ||
            !isa<llvm::Constant>(lookupValue(op.getResult(0))))
          return emitError(op.getLoc(), "unemittable constant value");
      }
      ReturnOp ret = cast<ReturnOp>(initializer->getTerminator());
      llvm::Constant *cst =
          cast<llvm::Constant>(lookupValue(ret.getOperand(0)));
      auto *global = cast<llvm::GlobalVariable>(lookupGlobal(op));
      if (!shouldDropGlobalInitializer(global->getLinkage(), cst))
        global->setInitializer(cst);
    }
  }

  return success();
}

/// Attempts to add an attribute identified by `key`, optionally with the given
/// `value` to LLVM function `llvmFunc`. Reports errors at `loc` if any. If the
/// attribute has a kind known to LLVM IR, create the attribute of this kind,
/// otherwise keep it as a string attribute. Performs additional checks for
/// attributes known to have or not have a value in order to avoid assertions
/// inside LLVM upon construction.
static LogicalResult checkedAddLLVMFnAttribute(Location loc,
                                               llvm::Function *llvmFunc,
                                               StringRef key,
                                               StringRef value = StringRef()) {
  auto kind = llvm::Attribute::getAttrKindFromName(key);
  if (kind == llvm::Attribute::None) {
    llvmFunc->addFnAttr(key, value);
    return success();
  }

  if (llvm::Attribute::doesAttrKindHaveArgument(kind)) {
    if (value.empty())
      return emitError(loc) << "LLVM attribute '" << key << "' expects a value";

    int result;
    if (!value.getAsInteger(/*Radix=*/0, result))
      llvmFunc->addFnAttr(
          llvm::Attribute::get(llvmFunc->getContext(), kind, result));
    else
      llvmFunc->addFnAttr(key, value);
    return success();
  }

  if (!value.empty())
    return emitError(loc) << "LLVM attribute '" << key
                          << "' does not expect a value, found '" << value
                          << "'";

  llvmFunc->addFnAttr(kind);
  return success();
}

/// Attaches the attributes listed in the given array attribute to `llvmFunc`.
/// Reports error to `loc` if any and returns immediately. Expects `attributes`
/// to be an array attribute containing either string attributes, treated as
/// value-less LLVM attributes, or array attributes containing two string
/// attributes, with the first string being the name of the corresponding LLVM
/// attribute and the second string beings its value. Note that even integer
/// attributes are expected to have their values expressed as strings.
static LogicalResult
forwardPassthroughAttributes(Location loc, Optional<ArrayAttr> attributes,
                             llvm::Function *llvmFunc) {
  if (!attributes)
    return success();

  for (Attribute attr : *attributes) {
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      if (failed(
              checkedAddLLVMFnAttribute(loc, llvmFunc, stringAttr.getValue())))
        return failure();
      continue;
    }

    auto arrayAttr = attr.dyn_cast<ArrayAttr>();
    if (!arrayAttr || arrayAttr.size() != 2)
      return emitError(loc)
             << "expected 'passthrough' to contain string or array attributes";

    auto keyAttr = arrayAttr[0].dyn_cast<StringAttr>();
    auto valueAttr = arrayAttr[1].dyn_cast<StringAttr>();
    if (!keyAttr || !valueAttr)
      return emitError(loc)
             << "expected arrays within 'passthrough' to contain two strings";

    if (failed(checkedAddLLVMFnAttribute(loc, llvmFunc, keyAttr.getValue(),
                                         valueAttr.getValue())))
      return failure();
  }
  return success();
}

LogicalResult ModuleTranslation::convertOneFunction(LLVMFuncOp func) {
  // Clear the block, branch value mappings, they are only relevant within one
  // function.
  blockMapping.clear();
  valueMapping.clear();
  branchMapping.clear();
  llvm::Function *llvmFunc = lookupFunction(func.getName());

  // Translate the debug information for this function.
  debugTranslation->translate(func, *llvmFunc);

  // Add function arguments to the value remapping table.
  // If there was noalias info then we decorate each argument accordingly.
  unsigned int argIdx = 0;
  for (auto kvp : llvm::zip(func.getArguments(), llvmFunc->args())) {
    llvm::Argument &llvmArg = std::get<1>(kvp);
    BlockArgument mlirArg = std::get<0>(kvp);

    if (auto attr = func.getArgAttrOfType<UnitAttr>(
            argIdx, LLVMDialect::getNoAliasAttrName())) {
      // NB: Attribute already verified to be boolean, so check if we can indeed
      // attach the attribute to this argument, based on its type.
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.noalias attribute attached to LLVM non-pointer argument");
      llvmArg.addAttr(llvm::Attribute::AttrKind::NoAlias);
    }

    if (auto attr = func.getArgAttrOfType<IntegerAttr>(
            argIdx, LLVMDialect::getAlignAttrName())) {
      // NB: Attribute already verified to be int, so check if we can indeed
      // attach the attribute to this argument, based on its type.
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.align attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(
          llvm::AttrBuilder().addAlignmentAttr(llvm::Align(attr.getInt())));
    }

    if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "llvm.sret")) {
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.sret attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(llvm::AttrBuilder().addStructRetAttr(
          llvmArg.getType()->getPointerElementType()));
    }

    if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "llvm.byval")) {
      auto argTy = mlirArg.getType();
      if (!argTy.isa<LLVM::LLVMPointerType>())
        return func.emitError(
            "llvm.byval attribute attached to LLVM non-pointer argument");
      llvmArg.addAttrs(llvm::AttrBuilder().addByValAttr(
          llvmArg.getType()->getPointerElementType()));
    }

    mapValue(mlirArg, &llvmArg);
    argIdx++;
  }

  // Check the personality and set it.
  if (func.personality().hasValue()) {
    llvm::Type *ty = llvm::Type::getInt8PtrTy(llvmFunc->getContext());
    if (llvm::Constant *pfunc =
            getLLVMConstant(ty, func.personalityAttr(), func.getLoc(), *this))
      llvmFunc->setPersonalityFn(pfunc);
  }

  // First, create all blocks so we can jump to them.
  llvm::LLVMContext &llvmContext = llvmFunc->getContext();
  for (auto &bb : func) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(llvmFunc);
    mapBlock(&bb, llvmBB);
  }

  // Then, convert blocks one by one in topological order to ensure defs are
  // converted before uses.
  auto blocks = detail::getTopologicallySortedBlocks(func.getBody());
  for (Block *bb : blocks) {
    llvm::IRBuilder<> builder(llvmContext);
    if (failed(convertBlock(*bb, bb->isEntryBlock(), builder)))
      return failure();
  }

  // After all blocks have been traversed and values mapped, connect the PHI
  // nodes to the results of preceding blocks.
  detail::connectPHINodes(func.getBody(), *this);

  // Finally, convert dialect attributes attached to the function.
  return convertDialectAttributes(func);
}

LogicalResult ModuleTranslation::convertDialectAttributes(Operation *op) {
  for (NamedAttribute attribute : op->getDialectAttrs())
    if (failed(iface.amendOperation(op, attribute, *this)))
      return failure();
  return success();
}

/// Check whether the module contains only supported ops directly in its body.
static LogicalResult checkSupportedModuleOps(Operation *m) {
  for (Operation &o : getModuleBody(m).getOperations())
    if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::MetadataOp>(&o) &&
        !o.hasTrait<OpTrait::IsTerminator>())
      return o.emitOpError("unsupported module-level operation");
  return success();
}

LogicalResult ModuleTranslation::convertFunctionSignatures() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles, or global initializers that reference functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    llvm::FunctionCallee llvmFuncCst = llvmModule->getOrInsertFunction(
        function.getName(),
        cast<llvm::FunctionType>(convertType(function.getType())));
    llvm::Value* fval = llvmFuncCst.getCallee();
    while (auto CE = dyn_cast<llvm::ConstantExpr>(fval)) {
      fval = CE->getOperand(0);
    }
    llvm::Function *llvmFunc = cast<llvm::Function>(fval);
    llvmFunc->setLinkage(convertLinkageToLLVM(function.linkage()));
    mapFunction(function.getName(), llvmFunc);

    // Forward the pass-through attributes to LLVM.
    if (failed(forwardPassthroughAttributes(function.getLoc(),
                                            function.passthrough(), llvmFunc)))
      return failure();
  }

  return success();
}

LogicalResult ModuleTranslation::convertFunctions() {
  // Convert functions.
  for (auto function : getModuleBody(mlirModule).getOps<LLVMFuncOp>()) {
    // Ignore external functions.
    if (function.isExternal())
      continue;

    if (failed(convertOneFunction(function)))
      return failure();
  }

  return success();
}

llvm::MDNode *
ModuleTranslation::getAccessGroup(Operation &opInst,
                                  SymbolRefAttr accessGroupRef) const {
  auto metadataName = accessGroupRef.getRootReference();
  auto accessGroupName = accessGroupRef.getLeafReference();
  auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
      opInst.getParentOp(), metadataName);
  auto *accessGroupOp =
      SymbolTable::lookupNearestSymbolFrom(metadataOp, accessGroupName);
  return accessGroupMetadataMapping.lookup(accessGroupOp);
}

LogicalResult ModuleTranslation::createAccessGroupMetadata() {
  mlirModule->walk([&](LLVM::MetadataOp metadatas) {
    metadatas.walk([&](LLVM::AccessGroupMetadataOp op) {
      llvm::LLVMContext &ctx = llvmModule->getContext();
      llvm::MDNode *accessGroup = llvm::MDNode::getDistinct(ctx, {});
      accessGroupMetadataMapping.insert({op, accessGroup});
    });
  });
  return success();
}

void ModuleTranslation::setAccessGroupsMetadata(Operation *op,
                                                llvm::Instruction *inst) {
  auto accessGroups =
      op->getAttrOfType<ArrayAttr>(LLVMDialect::getAccessGroupsAttrName());
  if (accessGroups && !accessGroups.empty()) {
    llvm::Module *module = inst->getModule();
    SmallVector<llvm::Metadata *> metadatas;
    for (SymbolRefAttr accessGroupRef :
         accessGroups.getAsRange<SymbolRefAttr>())
      metadatas.push_back(getAccessGroup(*op, accessGroupRef));

    llvm::MDNode *unionMD = nullptr;
    if (metadatas.size() == 1)
      unionMD = llvm::cast<llvm::MDNode>(metadatas.front());
    else if (metadatas.size() >= 2)
      unionMD = llvm::MDNode::get(module->getContext(), metadatas);

    inst->setMetadata(module->getMDKindID("llvm.access.group"), unionMD);
  }
}

llvm::Type *ModuleTranslation::convertType(Type type) {
  return typeTranslator.translateType(type);
}

/// A helper to look up remapped operands in the value remapping table.`
SmallVector<llvm::Value *, 8>
ModuleTranslation::lookupValues(ValueRange values) {
  SmallVector<llvm::Value *, 8> remapped;
  remapped.reserve(values.size());
  for (Value v : values)
    remapped.push_back(lookupValue(v));
  return remapped;
}

const llvm::DILocation *
ModuleTranslation::translateLoc(Location loc, llvm::DILocalScope *scope) {
  return debugTranslation->translateLoc(loc, scope);
}

llvm::NamedMDNode *
ModuleTranslation::getOrInsertNamedModuleMetadata(StringRef name) {
  return llvmModule->getOrInsertNamedMetadata(name);
}

void ModuleTranslation::StackFrame::anchor() {}

static std::unique_ptr<llvm::Module>
prepareLLVMModule(Operation *m, llvm::LLVMContext &llvmContext,
                  StringRef name) {
  m->getContext()->getOrLoadDialect<LLVM::LLVMDialect>();
  auto llvmModule = std::make_unique<llvm::Module>(name, llvmContext);
  if (auto dataLayoutAttr =
          m->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvmModule->setDataLayout(dataLayoutAttr.cast<StringAttr>().getValue());
  }
  if (auto targetTripleAttr =
          m->getAttr(LLVM::LLVMDialect::getTargetTripleAttrName()))
    llvmModule->setTargetTriple(targetTripleAttr.cast<StringAttr>().getValue());

  // Inject declarations for `malloc` and `free` functions that can be used in
  // memref allocation/deallocation coming from standard ops lowering.
  llvm::IRBuilder<> builder(llvmContext);
  llvmModule->getOrInsertFunction("malloc", builder.getInt8PtrTy(),
                                  builder.getInt64Ty());
  llvmModule->getOrInsertFunction("free", builder.getVoidTy(),
                                  builder.getInt8PtrTy());

  return llvmModule;
}

std::unique_ptr<llvm::Module>
mlir::translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext,
                              StringRef name) {
  if (!satisfiesLLVMModule(module))
    return nullptr;
  if (failed(checkSupportedModuleOps(module)))
    return nullptr;
  std::unique_ptr<llvm::Module> llvmModule =
      prepareLLVMModule(module, llvmContext, name);

  LLVM::ensureDistinctSuccessors(module);

  ModuleTranslation translator(module, std::move(llvmModule));
  if (failed(translator.convertFunctionSignatures()))
    return nullptr;
  if (failed(translator.convertGlobals()))
    return nullptr;
  if (failed(translator.createAccessGroupMetadata()))
    return nullptr;
  if (failed(translator.convertFunctions()))
    return nullptr;
  if (llvm::verifyModule(*translator.llvmModule, &llvm::errs()))
    return nullptr;

  return std::move(translator.llvmModule);
}
