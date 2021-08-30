//===- BarrierUtil.h - Utilities for barrier removal --------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_
#define MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_

#include "mlir/IR/Block.h"
#include "llvm/ADT/SetVector.h"
#include "polygeist/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

void findValuesUsedBelow(mlir::polygeist::BarrierOp barrier,
                         llvm::SetVector<mlir::Value> &crossing);

std::pair<mlir::Block *, mlir::Block::iterator>
findInsertionPointAfterLoopOperands(mlir::scf::ParallelOp op);

/// Emits the IR  computing the total number of iterations in the loop. We don't
/// need to linearize them since we can allocate an nD array instead.
static llvm::SmallVector<mlir::Value> emitIterationCounts(mlir::OpBuilder &rewriter,
                                                    mlir::scf::ParallelOp op) {
  using namespace mlir;
  SmallVector<Value> iterationCounts;
  for (auto bounds : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
    Value lowerBound = std::get<0>(bounds);
    Value upperBound = std::get<1>(bounds);
    Value step = std::get<2>(bounds);
    Value diff = rewriter.create<SubIOp>(op.getLoc(), upperBound, lowerBound);
    Value count = rewriter.create<SignedCeilDivIOp>(op.getLoc(), diff, step);
    iterationCounts.push_back(count);
  }
  return iterationCounts;
}

template<typename T>
static T allocateTemporaryBuffer(mlir::OpBuilder &rewriter, mlir::Value value,
                                     mlir::ValueRange iterationCounts) {
  using namespace mlir;
  SmallVector<int64_t> bufferSize(iterationCounts.size(),
                                  ShapedType::kDynamicSize);
  mlir::Type ty = value.getType();
  if (auto allocaOp = value.getDefiningOp<memref::AllocaOp>()) {
      auto mt = allocaOp.getType();
      bool hasDynamicSize = false;
      for(auto s : mt.getShape()) {
          if (s == ShapedType::kDynamicSize) {
              hasDynamicSize = true;
              break;
          }
      }
      if (!hasDynamicSize) {
          for(auto s : mt.getShape()) {
              bufferSize.push_back(s);
          }
          ty = mt.getElementType();
      }
  }
  auto type = MemRefType::get(bufferSize, ty);
  return rewriter.create<T>(value.getLoc(), type, iterationCounts);
}
#endif // MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_

