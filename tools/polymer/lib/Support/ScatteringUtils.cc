//===- ScatteringUtils.cc ---------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop scattering.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScatteringUtils.h"

#include <memory>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

namespace polymer {

class ScatTreeNodeImpl {
public:
  ScatTreeNodeImpl() {}
  ScatTreeNodeImpl(mlir::Value iv) : iv(iv) {}

  /// Get the depth of the tree starting from the given root node.
  unsigned getDepth(ScatTreeNodeImpl *root) const;

  /// Children of the current node.
  std::vector<std::unique_ptr<ScatTreeNodeImpl>> children;
  /// Mapping from IV to child ID.
  llvm::DenseMap<mlir::Value, unsigned> valueIdMap;
  /// Induction variable stored.
  mlir::Value iv;
};

unsigned ScatTreeNodeImpl::getDepth(ScatTreeNodeImpl *root) const {
  assert(root && "The root node should not be NULL.");

  llvm::SmallVector<std::pair<ScatTreeNodeImpl *, unsigned>, 8> nodes;
  nodes.push_back(std::make_pair(root, 1));
  unsigned maxDepth = 1;

  while (!nodes.empty()) {
    auto curr = nodes.pop_back_val();
    maxDepth = std::max(maxDepth, curr.second);

    for (auto &child : curr.first->children)
      nodes.push_back(std::make_pair(child.get(), curr.second + 1));
  }

  return maxDepth;
}

/// Insert a statement characterized by its enclosing operations into a
/// "scattering tree". This is done by iterating through every enclosing for-op
/// from the outermost to the innermost, and we try to traverse the tree by the
/// IVs of these ops. If an IV does not exist, we will insert it into the tree.
/// After that, we insert the current load/store statement into the tree as a
/// leaf. In this progress, we keep track of all the IDs of each child we meet
/// and the final leaf node, which will be used as the scattering.
void insertStatement(ScatTreeNodeImpl *root, ArrayRef<Operation *> enclosingOps,
                     SmallVectorImpl<unsigned> &scats) {
  ScatTreeNodeImpl *curr = root;

  for (unsigned i = 0, e = enclosingOps.size(); i < e; i++) {
    Operation *op = enclosingOps[i];
    // We only handle for op here.
    // TODO: is it necessary to deal with if?
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      SmallVector<mlir::Value, 4> indices;
      extractForInductionVars(forOp, &indices);

      for (const auto &iv : indices) {
        auto it = curr->valueIdMap.find(iv);
        if (it != curr->valueIdMap.end()) {
          // Add a new element to the scattering.
          scats.push_back(it->second);
          // move to the next IV along the tree.
          curr = curr->children[it->second].get();
        } else {
          // No existing node for such IV is found, create a new one.
          auto node = std::make_unique<ScatTreeNodeImpl>(iv);

          // Then insert the newly created node into the children set, update
          // the value to child ID map, and move the cursor to this new node.
          curr->children.push_back(std::move(node));
          unsigned valueId = curr->children.size() - 1;
          curr->valueIdMap[iv] = valueId;
          scats.push_back(valueId);
          curr = curr->children.back().get();
        }
      }
    }
  }

  // Append the leaf node for statement
  auto leaf = std::make_unique<ScatTreeNodeImpl>();
  curr->children.push_back(std::move(leaf));
  scats.push_back(curr->children.size() - 1);
}

ScatTreeNode::ScatTreeNode()
    : impl{std::move(std::make_unique<ScatTreeNodeImpl>())} {}
ScatTreeNode::ScatTreeNode(mlir::Value value)
    : impl{std::move(std::make_unique<ScatTreeNodeImpl>(value))} {}
ScatTreeNode::~ScatTreeNode() = default;
ScatTreeNode::ScatTreeNode(ScatTreeNode &&) = default;
ScatTreeNode &ScatTreeNode::operator=(ScatTreeNode &&) = default;

void ScatTreeNode::insertScopStmt(llvm::ArrayRef<mlir::Operation *> ops,
                                  llvm::SmallVectorImpl<unsigned> &scats) {
  insertStatement(impl.get(), ops, scats);
}

unsigned ScatTreeNode::getDepth() const { return impl->getDepth(impl.get()); }

} // namespace polymer
