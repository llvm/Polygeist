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
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class OpBuilder;
class Value;
class ValueRange;

namespace scf {
class ParallelOp;
} // namespace scf

namespace polygeist {
class BarrierOp;
} // namespace polygeist
} // namespace mlir

void findValuesUsedBelow(mlir::polygeist::BarrierOp barrier,
                         llvm::SetVector<mlir::Value> &crossing);

std::pair<mlir::Block *, mlir::Block::iterator>
findInsertionPointAfterLoopOperands(mlir::scf::ParallelOp op);

llvm::SmallVector<mlir::Value> emitIterationCounts(mlir::OpBuilder &builder,
                                                   mlir::scf::ParallelOp op);

template <typename AllocTy>
AllocTy allocateTemporaryBuffer(mlir::OpBuilder &builder, mlir::Value value,
                                    mlir::ValueRange iterationCounts) {
  llvm::SmallVector<int64_t> bufferSize(iterationCounts.size(),
                                        mlir::ShapedType::kDynamicSize);
  auto type = mlir::MemRefType::get(bufferSize, value.getType());
  return builder.create<AllocTy>(value.getLoc(), type, iterationCounts);
}

#endif // MLIR_LIB_DIALECT_SCF_TRANSFORMS_BARRIERUTILS_H_
