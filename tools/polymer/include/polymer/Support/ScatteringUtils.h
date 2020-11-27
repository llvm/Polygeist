//===- ScatteringUtils.h ----------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop scattering.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_SCATTERINGUTILS_H
#define POLYMER_SUPPORT_SCATTERINGUTILS_H

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Value;
class Operation;
} // namespace mlir

namespace polymer {

class ScatTreeNodeImpl;

/// Tree that holds scattering information. This node can represent an induction
/// variable or a statement. A statement is constructed as a leaf node.
class ScatTreeNode {
public:
  ScatTreeNode();
  ScatTreeNode(mlir::Value iv);

  ~ScatTreeNode();

  ScatTreeNode(ScatTreeNode &&);
  ScatTreeNode(const ScatTreeNode &) = delete;
  ScatTreeNode &operator=(ScatTreeNode &&);
  ScatTreeNode &operator=(const ScatTreeNode &) = delete;

  void insertScopStmt(llvm::ArrayRef<mlir::Operation *> ops,
                      llvm::SmallVectorImpl<unsigned> &scats);
  /// Get the depth of the tree starting from this node.
  unsigned getDepth() const;

private:
  std::unique_ptr<ScatTreeNodeImpl> impl;
};

} // namespace polymer

#endif
