#include "islNodeBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

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

static bool isInt(isl::ast_expr expression) {
  return isl_ast_expr_get_type(expression.get()) == isl_ast_expr_int;
}

static int getIntFromIslExpr(isl::ast_expr expression) {
  if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_int)
    llvm_unreachable("expect isl_ast_expr_int expression");
  auto val = expression.get_val();
  return std::stoi(val.to_str());
}

// Simplistic function that looks for an expression of type coeff * i + inc or i
// + inc.
static AffineExpr getAffineFromIslExpr(isl::ast_expr expr, MLIRContext *ctx) {
  assert(((isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id) ||
          (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_op)) &&
         "expect isl_ast_expr_id or isl_ast_epxr op");
  AffineExpr i;
  bindDims(ctx, i);
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id)
    return i;

  int coeff = 1;
  int inc = 1;
  auto sumOrMinusExpr = isl_ast_expr_get_op_type(expr.get());
  assert(((sumOrMinusExpr == isl_ast_op_add) ||
          (sumOrMinusExpr == isl_ast_op_minus)) &&
         "expect isl_ast_sum or isl_ast_minus");

  // assume the rhs of the sum is an expr_int.
  auto rhsSum = expr.get_op_arg(1);
  assert((isl_ast_expr_get_type(rhsSum.get()) == isl_ast_expr_int) &&
         "expect an isl_ast_expr_int");
  auto incVal = rhsSum.get_val();
  inc = std::stoi(incVal.to_str());

  // check if we have a nested mul.
  auto mulOrId = expr.get_op_arg(0);
  assert(((isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_id) ||
          (isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_op)) &&
         "expect isl_ast_expr_id or isl_ast_expr_op");
  if (isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_id)
    return i + inc;

  // if so get the value of the mul.
  auto mulType = isl_ast_expr_get_op_type(mulOrId.get());
  assert((mulType == isl_ast_op_mul) && "expect isl_ast_mul");
  auto lhsMul = mulOrId.get_op_arg(0);
  assert((isl_ast_expr_get_type(lhsMul.get()) == isl_ast_expr_int) &&
         "expect an isl_ast_expr_int");
  auto coeffVal = lhsMul.get_val();
  coeff = std::stoi(coeffVal.to_str());
  return coeff * i + inc;
}

// walk an isl::ast_expr looking for an isl_ast_expr_id if
// any.
static void getBoundId(isl::ast_expr expr, std::string &id) {
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id)
    id = expr.get_id().to_str();
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_int)
    return;
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_op)
    for (int i = 0; i < expr.get_op_n_arg(); i++)
      getBoundId(expr.get_op_arg(i), id);
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
  auto incrementAsInt = std::abs(getIntFromIslExpr(increment));

  auto ctx = MLIRBuilder_.getContext();
  AffineForOp loop;
  if (isInt(lowerBound) && isInt(upperBound)) {
    auto upperBoundAsInt = getIntFromIslExpr(upperBound) + 1;
    auto lowerBoundAsInt = getIntFromIslExpr(lowerBound);
    loop = MLIRBuilder_.createLoop(lowerBoundAsInt, upperBoundAsInt,
                                   incrementAsInt);
  } else if (isInt(lowerBound) && !isInt(upperBound)) {
    auto upperBoundAsExpr = getAffineFromIslExpr(upperBound, ctx);
    std::string upperBoundId = "";
    getBoundId(upperBound, upperBoundId);
    auto lowerBoundAsInt = getIntFromIslExpr(lowerBound);
    loop = MLIRBuilder_.createLoop(lowerBoundAsInt, upperBoundAsExpr,
                                   upperBoundId, incrementAsInt);
  } else if (!isInt(lowerBound) && isInt(upperBound)) {
    auto upperBoundAsInt = getIntFromIslExpr(upperBound) + 1;
    auto lowerBoundAsExpr = getAffineFromIslExpr(lowerBound, ctx);
    std::string lowerBoundId = "";
    getBoundId(lowerBound, lowerBoundId);
    loop = MLIRBuilder_.createLoop(lowerBoundAsExpr, lowerBoundId,
                                   upperBoundAsInt, incrementAsInt);
  } else {
    auto upperBoundAsExpr = getAffineFromIslExpr(upperBound, ctx);
    auto lowerBoundAsExpr = getAffineFromIslExpr(lowerBound, ctx);
    std::string upperBoundId = "";
    getBoundId(upperBound, upperBoundId);
    std::string lowerBoundId = "";
    getBoundId(lowerBound, lowerBoundId);
    loop =
        MLIRBuilder_.createLoop(lowerBoundAsExpr, lowerBoundId,
                                upperBoundAsExpr, upperBoundId, incrementAsInt);
  }

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
    MLIRBuilder_.dump();
    llvm_unreachable("cannot generate statement");
  }
}

void IslNodeBuilder::createBlock(isl::ast_node blockNode) {
  auto list = blockNode.block_get_children();
  for (int i = 0; i < list.n_ast_node(); i++)
    MLIRFromISLAstImpl(list.get_ast_node(i));
}

void IslNodeBuilder::createIf(isl::ast_node ifNode) {
  outs() << __func__ << "\n";
}

void IslNodeBuilder::MLIRFromISLAstImpl(isl::ast_node node) {
  // std::cout << node.to_str() << "\n";
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
