//===- SCFToOpenMP.cpp - Structured Control Flow to OpenMP conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into OpenMP
// parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "../PassDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

template <typename... OpTy> static bool matchSimpleReduction(Block &block) {
  if (block.empty() || llvm::hasSingleElement(block) ||
      std::next(block.begin(), 2) != block.end())
    return false;
  return isa<OpTy...>(block.front()) &&
         isa<scf::ReduceReturnOp>(block.back()) &&
         block.back().getOperand(0) == block.front().getResult(0);
}

static bool matchSelectReduction(Block &block, bool min = true) {
  if (block.empty() || llvm::hasSingleElement(block) ||
      std::next(block.begin(), 2) == block.end() ||
      std::next(block.begin(), 3) != block.end())
    return false;

  auto compare = dyn_cast<CmpFOp>(block.front());
  auto select = dyn_cast<SelectOp>(block.front().getNextNode());
  auto terminator = dyn_cast<scf::ReduceReturnOp>(block.back());
  if (!compare || !select || !terminator)
    return false;

  bool matched = false;
  bool switchOperands = !min;
  if (compare.predicate() == CmpFPredicate::OLE ||
      compare.predicate() == CmpFPredicate::OLT) {
    matched = true;
  } else if (compare.predicate() == CmpFPredicate::OGE ||
             compare.predicate() == CmpFPredicate::OGT) {
    matched = true;
    switchOperands = !switchOperands;
  }
  if (!matched)
    return false;

  if (select.condition() != compare.result())
    return false;

  if (select.true_value() != (switchOperands ? compare.lhs() : compare.rhs()) ||
      select.false_value() != (switchOperands ? compare.rhs() : compare.lhs()))
    return false;

  return select.result() == terminator.result();
}

static const llvm::fltSemantics &fltSemanticsForType(FloatType type) {
  if (type.isF16())
    return llvm::APFloat::IEEEhalf();
  if (type.isF32())
    return llvm::APFloat::IEEEsingle();
  if (type.isF64())
    return llvm::APFloat::IEEEdouble();
  if (type.isF128())
    return llvm::APFloat::IEEEquad();
  if (type.isBF16())
    return llvm::APFloat::BFloat();
  if (type.isF80())
    return llvm::APFloat::x87DoubleExtended();
  llvm_unreachable("unknown float type");
}

static Attribute minMaxValueForType(Type type, bool min) {
  if (auto fltType = type.dyn_cast<FloatType>())
    return FloatAttr::get(type,
                          llvm::APFloat(fltSemanticsForType(fltType), min));
  return nullptr;
}

static omp::ReductionDeclareOp declareReduction(PatternRewriter &builder,
                                                scf::ReduceOp reduce) {
  Operation *container = SymbolTable::getNearestSymbolTable(reduce);
  SymbolTable symbolTable(container);

  // Insert reduction declarations the ancestor of the reduction block that
  // lives in a symbol table.
  Operation *insertionPoint = reduce;
  while (insertionPoint->getParentOp() != container)
    insertionPoint = insertionPoint->getParentOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertionPoint);

  auto decl = builder.create<omp::ReductionDeclareOp>(
      reduce.getLoc(), "__scf_reduction", reduce.operand().getType());
  symbolTable.insert(decl);

  Type type = reduce.operand().getType();
  Type ptrType = LLVM::LLVMPointerType::get(type);
  if (matchSimpleReduction<AddFOp, LLVM::FAddOp>(reduce.getRegion().front())) {
    builder.createBlock(&decl.initializerRegion(),
                        decl.initializerRegion().end());
    Value init = builder.create<LLVM::ConstantOp>(
        reduce.getLoc(), type, builder.getFloatAttr(type, 0.0));
    builder.create<omp::YieldOp>(reduce.getLoc(), init);

    Operation *terminator = &reduce.getRegion().front().back();
    assert(isa<scf::ReduceReturnOp>(terminator));
    builder.setInsertionPoint(terminator);
    builder.replaceOpWithNewOp<omp::YieldOp>(terminator,
                                             terminator->getOperands());
    builder.inlineRegionBefore(reduce.getRegion(), decl.reductionRegion(),
                               decl.reductionRegion().end());

    Block *atomicBlock = builder.createBlock(&decl.atomicReductionRegion(),
                                             decl.atomicReductionRegion().end(),
                                             {ptrType, ptrType});

    Value loaded = builder.create<LLVM::LoadOp>(reduce.getLoc(),
                                                atomicBlock->getArgument(1));
    builder.create<LLVM::AtomicRMWOp>(
        reduce.getLoc(), type, LLVM::AtomicBinOp::fadd,
        atomicBlock->getArgument(0), loaded, LLVM::AtomicOrdering::monotonic);
    builder.create<omp::YieldOp>(reduce.getLoc(), ArrayRef<Value>());

    return decl;
  }
  if (matchSimpleReduction<MulFOp, LLVM::FMulOp>(reduce.getRegion().front())) {
    builder.createBlock(&decl.initializerRegion(),
                        decl.initializerRegion().end());
    Value init = builder.create<LLVM::ConstantOp>(
        reduce.getLoc(), type, builder.getFloatAttr(type, 1.0));
    builder.create<omp::YieldOp>(reduce.getLoc(), init);

    Operation *terminator = &reduce.getRegion().front().back();
    assert(isa<scf::ReduceReturnOp>(terminator));
    builder.setInsertionPoint(terminator);
    builder.replaceOpWithNewOp<omp::YieldOp>(terminator,
                                             terminator->getOperands());
    builder.inlineRegionBefore(reduce.getRegion(), decl.reductionRegion(),
                               decl.reductionRegion().end());

    // TODO: add atomic region using cmpxchg (which needs atomic load to be
    // available as an op)
    return decl;
  }
  return nullptr;
}

namespace {

struct ParallelOpLowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    // Replace SCF yield with OpenMP yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(parallelOp.getBody());
      assert(llvm::hasSingleElement(parallelOp.region()) &&
             "expected scf.parallel to have one block");
      rewriter.replaceOpWithNewOp<omp::YieldOp>(
          parallelOp.getBody()->getTerminator(), ValueRange());
    }

    // Create the parallel wrapper.
    Location loc = parallelOp.getLoc();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(1));
    SmallVector<Value> reductionVariables;
    reductionVariables.reserve(parallelOp.getNumReductions());

    // Make sure the we don't overflow the stack with local `alloca`s by saving
    // and restoring the stack pointer.
    Value token = rewriter.create<LLVM::StackSaveOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    for (Value init : parallelOp.initVals()) {
      assert(LLVM::isCompatibleType(init.getType()));
      Value storage = rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(init.getType()), one, 0);
      rewriter.create<LLVM::StoreOp>(loc, init, storage);
      reductionVariables.push_back(storage);
    }

    auto ompParallel = rewriter.create<omp::ParallelOp>(loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ompParallel.region());

      // Replace SCF yield with OpenMP yield.
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToEnd(parallelOp.getBody());
        assert(llvm::hasSingleElement(parallelOp.region()) &&
               "expected scf.parallel to have one block");
        rewriter.replaceOpWithNewOp<omp::YieldOp>(
            parallelOp.getBody()->getTerminator(), ValueRange());
      }

      // Replace the loop.
      auto loop = rewriter.create<omp::WsLoopOp>(
          parallelOp.getLoc(), parallelOp.lowerBound(), parallelOp.upperBound(),
          parallelOp.step());
      rewriter.create<omp::TerminatorOp>(loc);

      rewriter.inlineRegionBefore(parallelOp.region(), loop.region(),
                                  loop.region().begin());
      loop.reduction_varsMutable().append(reductionVariables);
    }

    // Load loop results.
    SmallVector<Value> results;
    results.reserve(reductionVariables.size());
    for (Value variable : reductionVariables) {
      Value res = rewriter.create<LLVM::LoadOp>(loc, variable);
      results.push_back(res);
    }
    rewriter.replaceOp(parallelOp, results);

    rewriter.create<LLVM::StackRestoreOp>(loc, token);
    return success();
  }
};

struct ReduceOpLowering : public OpRewritePattern<scf::ReduceOp> {
  using OpRewritePattern<scf::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    omp::ReductionDeclareOp decl = declareReduction(rewriter, reduceOp);
    if (!decl)
      return failure();

    rewriter.replaceOpWithNewOp<omp::ReductionOp>(
        reduceOp, reduceOp.operand(),
        SymbolRefAttr::get(rewriter.getContext(), decl.sym_name()));
    return success();
  }
};

/// Applies the conversion patterns in the given function.
static LogicalResult applyPatterns(FuncOp func) {
  ConversionTarget target(*func.getContext());
  target.addDynamicallyLegalOp<scf::ReduceOp>(
      [](scf::ReduceOp op) { return isa<scf::ParallelOp>(op->getParentOp()); });
  target.addDynamicallyLegalOp<scf::ReduceReturnOp>([](scf::ReduceReturnOp op) {
    return isa<scf::ReduceOp>(op->getParentOp());
  });
  target.addDynamicallyLegalOp<scf::ParallelOp>([](scf::ParallelOp op) {
    return op->getParentOfType<omp::ParallelOp>() != nullptr;
  });
  target.addDynamicallyLegalOp<scf::YieldOp>(
      [](scf::YieldOp op) { return !isa<scf::ParallelOp>(op->getParentOp()); });
  target.addLegalDialect<omp::OpenMPDialect, LLVM::LLVMDialect>();

  RewritePatternSet patterns(func.getContext());
  patterns.add<ParallelOpLowering, ReduceOpLowering>(func.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(func, target, frozen);
}

/// A pass converting SCF operations to OpenMP operations.
struct SCFToOpenMPPass : public ConvertSCFToOpenMPBase<SCFToOpenMPPass> {
  /// Pass entry point.
  void runOnFunction() override {
    if (failed(applyPatterns(getFunction())))
      signalPassFailure();
  }
};

} // end namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvertSCFToOpenMPPass() {
  return std::make_unique<SCFToOpenMPPass>();
}
