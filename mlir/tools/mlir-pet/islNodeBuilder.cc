#include "islNodeBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace codegen;
using namespace mlir;
using namespace llvm;
using namespace pet;

// TODO: check loop direction.
static isl::ast_expr getUpperBound(isl::ast_node nodeFor) {
  if (isl_ast_node_get_type(nodeFor.get()) != isl_ast_node_for)
    llvm_unreachable("expect a for node");
  isl::ast_expr condition = nodeFor.for_get_cond();
  if (isl_ast_expr_get_type(condition.get()) != isl_ast_expr_op)
    llvm_unreachable("conditional expression is not an atomic upper bound");
  return condition.get_op_arg(1);
}

// TODO: Is Int enough or should we return a Value? see:
// https://github.com/llvm/llvm-project/blob/2c1a142a78ffe8ed06fd7bfd17750afdceeaecc9/polly/lib/CodeGen/IslExprBuilder.cpp#L746
static int createIntFromIslExpr(isl::ast_expr expression) {
  if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_int)
    llvm_unreachable("expect isl_ast_expr_int expression");
  auto val = expression.get_val();
  return std::stoi(val.to_str());
}

// TODO: See how we can get location information.
// TODO: handle degenerate loop (see isl_ast_node_for_is_degenerate)
// TODO: See how to handle more complex expression in the loop.
void IslNodeBuilder::createFor(isl::ast_node forNode) {
  auto lowerBound = forNode.for_get_init();
  auto increment = forNode.for_get_inc();
  auto iterator = forNode.for_get_iterator();
  auto iteratorId = iterator.get_id().to_str();
  auto upperBound = getUpperBound(forNode);

  auto lowerBoundAsInt = createIntFromIslExpr(lowerBound);
  auto incrementAsInt = createIntFromIslExpr(increment);
  auto upperBoundAsInt = createIntFromIslExpr(upperBound);

  auto loop =
      MLIRBuilder_.createLoop(lowerBoundAsInt, upperBoundAsInt, incrementAsInt);

  auto resInsertion =
      MLIRBuilder_.getLoopTable().insert(iteratorId, loop.getInductionVar());
  if (failed(resInsertion))
    llvm_unreachable("failed to insert in loop table");

  // create loop body.
  MLIRFromISLAstImpl(forNode.for_get_body());

  // induction variable goes out of scop. Remove from
  // loopTable
  MLIRBuilder_.getLoopTable().erase(iteratorId);

  // set the insertion point after the loop operation.
  MLIRBuilder_.setInsertionPointAfter(&loop);
}

void IslNodeBuilder::createUser(isl::ast_node userNode) {
  auto expression = userNode.user_get_expr();
  if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_op)
    llvm_unreachable("expect isl_ast_expr_op");
  if (isl_ast_expr_get_op_type(expression.get()) != isl_ast_op_call)
    llvm_unreachable("expect operation of type call");
  auto stmtExpression = expression.get_op_arg(0);
  auto stmtId = stmtExpression.get_id();
  auto stmt = islAst_.getScop().getStmt(stmtId);
  if (!stmt)
    llvm_unreachable("cannot find statement");
  auto body = stmt->body;
  if (pet_tree_get_type(body) != pet_tree_expr)
    llvm_unreachable("expect pet_tree_expr");
  auto expr = pet_tree_expr_get_expr(body);

  if (failed(MLIRBuilder_.createStmt(expr))) {
    llvm_unreachable("cannot generate statement");
  }
}

void IslNodeBuilder::createBlock(isl::ast_node blockNode) {
  auto list = blockNode.block_get_children();
  for (size_t i = 0; i < list.n_ast_node(); i++)
    MLIRFromISLAstImpl(list.get_ast_node(i));
}

void IslNodeBuilder::createIf(isl::ast_node ifNode) {
  outs() << __func__ << "\n";
}

void IslNodeBuilder::MLIRFromISLAstImpl(isl::ast_node node) {
  switch (isl_ast_node_get_type(node.get())) {
  case isl_ast_node_error:
    llvm_unreachable("code generation error");
  case isl_ast_node_for:
    createFor(node);
    return;
  case isl_ast_node_user:
    createUser(node);
    return;
  case isl_ast_node_block:
    createBlock(node);
    return;
  case isl_ast_node_mark:
    // simply return, don't generate.
    return;
  case isl_ast_node_if:
    createIf(node);
    return;
  }
  llvm_unreachable("unknown isl_ast_node_type");
}

void IslNodeBuilder::MLIRFromISLAst() {
  isl::ast_node root = islAst_.getRoot();
  MLIRFromISLAstImpl(root);
  // insert return statement.
  MLIRBuilder_.createReturn();
  // verify the module after we have finisched constructing it.
  if (failed(MLIRBuilder_.verifyModule()))
    llvm_unreachable("module verification error");
}
