//===- Utils.cc -------------------------------------------------*- C++ -*-===//
//
// This file implements some generic utility functions.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/Utils.h"
#include "polymer/Support/PolymerUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "polymer-utils"

void polymer::inferBlockArgs(Block *block, llvm::SetVector<Value> &args) {
  // Any value being used will be added to the set first.
  block->walk([&](Operation *op) {
    for (mlir::Value operand : op->getOperands())
      args.insert(operand);
  });

  // Then we remove them if they are actually defined by the operations within
  // the current block.
  block->walk([&](Operation *op) {
    for (mlir::Value result : op->getResults())
      args.remove(result);

    Block *curr = op->getBlock();
    for (mlir::Value blkArg : curr->getArguments())
      args.remove(blkArg);
  });
}

void polymer::dedupIndexCast(mlir::func::FuncOp f) {
  if (f.getBlocks().empty())
    return;

  Block &entry = f.getBlocks().front();
  llvm::MapVector<Value, Value> argToCast;
  SmallVector<Operation *> toErase;
  for (auto &op : entry) {
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(&op)) {
      auto arg = dyn_cast<BlockArgument>(indexCast.getOperand());
      if (argToCast.count(arg)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found duplicated index_cast: " << indexCast << '\n');
        indexCast.replaceAllUsesWith(argToCast.lookup(arg));
        toErase.push_back(indexCast);
      } else {
        argToCast[arg] = indexCast;
      }
    }
  }

  for (auto op : toErase)
    op->erase();
}
