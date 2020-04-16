#ifndef PETMLIR_ISL_NODE_BUILDER_H
#define PETMLIR_ISL_NODE_BUILDER_H

#include "islAst.h"
#include "mlirCodegen.h"
#include "pet.h"

namespace codegen {

class IslNodeBuilder {
public:
  IslNodeBuilder(ast::IslAst &ast, MLIRCodegen &MLIRBuilder)
      : islAst_(ast), MLIRBuilder_(MLIRBuilder){};

  // entry point to generate MLIR IR
  // by walking the isl ast.
  void MLIRFromISLAst();

private:
  // reference to islAst.
  ast::IslAst &islAst_;

  // reference to mlir IR builder.
  MLIRCodegen &MLIRBuilder_;

  // TODO: void or LogicalResult?
  // handles MLIR IR code generation.
  void MLIRFromISLAstImpl(isl::ast_node node);

  // TODO: void or LogicalResult?
  // create a forOp
  void createFor(isl::ast_node forNode);

  // TODO: void or LogicalResult?
  // create a statement.
  void createUser(isl::ast_node userNode);

  // TODO: void or LogicalResult?
  // create a compound statements.
  void createBlock(isl::ast_node blockNode);

  // TODO: void or LogicalResult?
  // create an ifOp.
  void createIf(isl::ast_node ifNode);
};

} // end namespace codegen

#endif
