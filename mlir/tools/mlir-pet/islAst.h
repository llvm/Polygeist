#ifndef PETMLIR_ISL_AST_H
#define PETMLIR_ISL_AST_H

#include "scop.h"

namespace ast {

class IslAst {
public:
  explicit IslAst(pet::Scop &scop);
  IslAst(const IslAst &) = delete;
  IslAst(IslAst &&) = default;

  IslAst &operator=(const IslAst &) = delete;
  IslAst &operator=(IslAst &&) = default;

  // dump the current ast.
  void dump() const;

  // get ast root node.
  isl::ast_node getRoot() const;

  // get reference to scop.
  pet::Scop &getScop() const;

private:
  // current scop.
  pet::Scop &scop_;

  // root node for the isl ast.
  isl::ast_node root_;
};

} // end namespace ast

#endif
