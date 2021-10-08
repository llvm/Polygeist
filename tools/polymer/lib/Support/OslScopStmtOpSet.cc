//===- OslScopStmtOpSet.cc --------------------------------------*- C++ -*-===//
//
// This file implements the class OslScopStmtOpSet.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScopStmtOpSet.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace mlir;
using namespace polymer;

void OslScopStmtOpSet::insert(mlir::Operation *op) {
  opSet.insert(op);
  if (isa<mlir::AffineStoreOp>(op)) {
    assert(!storeOp && "There should be only one AffineStoreOp in the set.");
    storeOp = op;
  }
}

LogicalResult OslScopStmtOpSet::getEnclosingOps(
    SmallVectorImpl<mlir::Operation *> &enclosingOps) {
  SmallVector<Operation *, 8> ops;
  SmallPtrSet<Operation *, 8> visited;
  for (auto op : opSet) {
    if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(op)) {
      ops.clear();
      getEnclosingAffineForAndIfOps(*op, &ops);
      for (auto enclosingOp : ops) {
        if (visited.find(enclosingOp) == visited.end()) {
          visited.insert(enclosingOp);
          enclosingOps.push_back(enclosingOp);
        }
      }
    }
  }

  return success();
}

LogicalResult
OslScopStmtOpSet::getDomain(FlatAffineValueConstraints &domain,
                            SmallVectorImpl<mlir::Operation *> &enclosingOps) {
  return getIndexSet(enclosingOps, &domain);
}

LogicalResult OslScopStmtOpSet::getDomain(FlatAffineValueConstraints &domain) {
  SmallVector<Operation *, 8> enclosingOps;
  if (failed(getEnclosingOps(enclosingOps)))
    return failure();

  return getDomain(domain, enclosingOps);
}
