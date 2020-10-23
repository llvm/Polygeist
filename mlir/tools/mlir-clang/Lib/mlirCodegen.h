#ifndef PETMLIR_MLIR_CODEGENERATION_H
#define PETMLIR_MLIR_CODEGENERATION_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "pet.h"
#include "scop.h"
#include <map>

namespace codegen {

enum class BinaryOpType { ADD, MUL, SUB, DIV };

class SymbolTable {
public:
  SymbolTable() = default;

  // insert new element with id "id" and value "value".
  mlir::LogicalResult insert(std::string id, mlir::Value value);

  // erase element wit id "id"
  mlir::LogicalResult erase(std::string id);

  // find element with id "id" and bind to value.
  mlir::LogicalResult find(std::string id, mlir::Value &value) const;

  // find element with id "id"
  mlir::LogicalResult find(std::string id) const;

  // get size.
  size_t size() const;

  // dump the current state.
  void dump() const;

private:
  // the symbol table maps a loop value to the loop id.
  std::map<std::string, mlir::Value> symbolTable_;

protected:
  // dump helper.
  void dumpImpl() const;

public:
  decltype(symbolTable_.begin()) begin() { return symbolTable_.begin(); }
  decltype(symbolTable_.cbegin()) begin() const {
    return symbolTable_.cbegin();
  }
};

// We access the induction variable by position as
// isl and pet use different id for the idv vars
// (i.e., 'c0' (isl) and 'i' (pet). Thus we prefer
// to use a different table to collect induction vars
// instead of using the symbol table.
class LoopTable : public SymbolTable {
public:
  // get value at position "pos"
  mlir::LogicalResult getValueAtPos(size_t pos, mlir::Value &value) const;
  mlir::LogicalResult getIdAtPos(size_t pos, std::string &valueId) const;
  // check if a mapping is in the table.
  mlir::LogicalResult lookUpPetMapping(std::string idPet) const;
  mlir::LogicalResult lookUpIslMapping(std::string idIsl,
                                       std::string &mapping) const;
  // insert a new mapping for induction variables.
  // i.e., i mapped with c0.
  void insertMapping(std::string idPet, std::string idIsl);

  // dump the current state.
  void dump() const;

private:
  // mapped indVars. (i.e., i -> c0)
  std::map<std::string, std::string> mappedIndVars_;
};

class MLIRCodegen {
public:
  MLIRCodegen(mlir::MLIRContext &context);

  mlir::MLIRContext *getContext() { return theModule_.getContext(); };

  // dump the current state of "theModule_"
  void dump();

  // print the current state of "theModule_";
  void print(llvm::raw_ostream &os);

  // verify if the module after we have finished
  // constructing it, this will check the structural
  // properties of the IR.
  mlir::LogicalResult verifyModule();

  // create an affineForOp or parallelforop.
  mlir::Operation *createLoop(int lb, int ub, int step, std::string iteratorId,
                              bool parallel);
  mlir::Operation *createLoop(int lb, mlir::AffineExpr ubExpr, std::string ubId,
                              int step, bool leqBound, std::string iteratorId,
                              bool parallel);
  mlir::Operation *createLoop(mlir::AffineExpr lbExpr, std::string lb, int ub,
                              int step, std::string iteratorId, bool parallel);
  mlir::Operation *createLoop(mlir::AffineExpr lbExpr, std::string lb,
                              mlir::AffineExpr ubExpr, std::string ub, int step,
                              std::string iteratorId, bool parallel);

  // return a reference to loop table.
  LoopTable &getLoopTable();

  // create a std::return op. TODO: mlir::ReturnOp instead of void?
  void createReturn();

  // create a statement from a pet_expr.
  mlir::LogicalResult createStmt(__isl_keep pet_expr *expr);

  // set the insertion point after the loop op. This method
  // is used by islNodeBuilder to move the insertion point
  // after the body of a loop has been created.
  void setInsertionPointAfter(mlir::Operation *op);

  // This should not be public, but it is used by IslNodeBuilder.
  mlir::LogicalResult getSymbolInductionVar(std::string idIsl,
                                            mlir::Value &indVar) const;

public:
  // The builder is an helper class to create IR inside
  // a function.
  mlir::OpBuilder builder_;

  // A "module" matched a scop source file.
  mlir::ModuleOp theModule_;

  // The symbol table maps a declared variable to
  // a value.
  SymbolTable symbolTable_;

  // The loop table maps a loop id to a value
  LoopTable loopTable_;

  // get the type of input arrays for the current scop.
  llvm::SmallVector<mlir::Type, 8> getFunctionArgumentsTypes(
      mlir::MLIRContext &context,
      const llvm::SmallVector<pet::PetArray, 4> &inputTensors);

  // return the type for the current "inputTensor"
  mlir::Type getTensorType(mlir::MLIRContext &context,
                           const pet::PetArray &inputTensor);

  // insert a new declaration in the symbol table.
  mlir::LogicalResult declare(std::string id, mlir::Value value);

  // create a new expr from the pet_expr "expr"
  mlir::Value createExpr(__isl_take pet_expr *expr, mlir::Type t = nullptr);

  // create a load op from the pet_expr_access "expr".
  mlir::Value createLoad(__isl_take pet_expr *expr);

  // create a store op from the pet_expr_access "expr".
  mlir::Value createStore(__isl_take pet_expr *expr, mlir::Value op);

  // get the symbol "expr". The symbol variable must be
  // declared before the invocation of this function and available
  // in the symbol table.
  mlir::LogicalResult getSymbol(__isl_keep pet_expr *expr,
                                mlir::Value &scalar) const;
  mlir::LogicalResult getSymbolInductionVar(__isl_keep pet_expr *expr,
                                            mlir::Value &indVar) const;

  // check if "expr" is already in the symbol table.
  mlir::LogicalResult isInSymbolTable(__isl_keep pet_expr *expr) const;

  // get the induction variables associated with "expr". The induction variables
  // must be available in the loop table.
  mlir::LogicalResult
  getSymbolInductionVar(__isl_keep pet_expr *expr,
                        llvm::SmallVector<mlir::Value, 4> &loopIvs);

  // create op from pet_expr_op "expr"
  mlir::Value createOp(__isl_take pet_expr *expr, mlir::Type t);

  // create an assignement op (pet_expr_assign)
  mlir::Value createAssignmentOp(__isl_take pet_expr *expr);

  // create an assignement op with operation (i.e., pet_op_add_assign)
  mlir::Value createAssignmentWithOp(__isl_take pet_expr *expr);

  // create a binary operation.
  mlir::Value createBinaryOp(mlir::Location &loc, mlir::Value &lhs,
                             mlir::Value &rhs, BinaryOpType type);

  // create a constant operation.
  mlir::Value createConstantOp(__isl_take pet_expr *expr,
                               pet::ElementType type);

  // create a floating point constant operation (f32).
  mlir::Value createConstantFloatOp(float val, mlir::Location &loc);

  // create a floating point constant operation (f64).
  mlir::Value createConstantDoubleOp(double val, mlir::Location &loc);

  // create a integer constant operation (32-bit).
  mlir::Value createConstantIntOp(int val, mlir::Location &loc);

  // create postInc operation.
  mlir::Value createPostInc(__isl_take pet_expr *expr);

  // create alloc operation.
  mlir::Value createDefinition(__isl_take pet_expr *expr);
  mlir::Value createAllocOp(__isl_keep pet_expr *expr, mlir::Type t,
                            mlir::Value v = nullptr);

  // create call operation.
  mlir::Value createCallOp(__isl_take pet_expr *expr, mlir::Type t);

  // check if 'expr' is a multi-dimensional array.
  bool isMultiDimensionalArray(__isl_keep pet_expr *expr) const;

  // get 'expr' dimensionality.
  size_t getDimensionalityExpr(__isl_keep pet_expr *expr) const;

  // return memref type for 'expr'.
  mlir::MemRefType convertExprToMemRef(__isl_keep pet_expr *expr,
                                       mlir::Type t) const;

  // apply "expr" to loopIvs.
  llvm::SmallVector<mlir::Value, 4>
  applyAccessExpression(__isl_keep pet_expr *expr,
                        llvm::SmallVector<mlir::Value, 4> &loopIvs);
  // create expr `expr` for induction variable `indvar`.
  mlir::Value composeInductionExpression(__isl_keep pet_expr *expr,
                                         mlir::Value indVar);

  // get indexes as mlir::Value from 'muaff'.
  mlir::LogicalResult getIndexes(isl::multi_pw_aff muaff,
                                 llvm::SmallVector<mlir::Value, 4> &loopIvs);
};

} // end namespace codegen

#endif
