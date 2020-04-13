#include "mlirCodegen.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace codegen;
using namespace mlir;
using namespace llvm;
using namespace pet;

LoopTable &MLIRCodegen::getLoopTable() { return loopTable_; }

// helper function to get the indexes of the output dimensions
// by comparing them with the input dimensions. This function
// assume "muaff" *not* to be scheduled.
static SmallVector<int, 4> getIndexesPos(isl::multi_pw_aff muaff) {
  auto spaceDimOut = muaff.get_space().dim(isl::dim::out);
  if (spaceDimOut == 0)
    llvm_unreachable("expect multi-dimensional array");
  SmallVector<int, 4> indexPos{};
  auto umap = isl::union_map::from(muaff);
  if (umap.n_map() != 1)
    return {};
  auto map = isl::map::from_union_map(umap);
  for (size_t i = 0; i < map.dim(isl::dim::out); i++) {
    isl::pw_aff pwaff = muaff.get_pw_aff(i);
    pwaff.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
      for (size_t j = 0; j < map.dim(isl::dim::in); j++) {
        auto val = a.get_coefficient_val(isl::dim::in, j);
        if (!val.is_zero())
          indexPos.push_back(j);
      }
      return isl_stat_ok;
    });
  }
  return indexPos;
}

// helper function. Is pet_expr a multi-dimensional access?
static bool isMultiDimensionalArray(__isl_keep pet_expr *expr) {
  if (pet_expr_get_type(expr) != pet_expr_access)
    llvm_unreachable("expect pet_expr_access type");
  auto indexes = isl::manage(pet_expr_access_get_index(expr));
  auto dimSpaceOut = indexes.get_space().dim(isl::dim::out);
  if (dimSpaceOut != 0)
    return true;
  return false;
}

LogicalResult MLIRCodegen::getSymbol(__isl_keep pet_expr *expr, Value &scalar) {
  auto arrayId = isl::manage(pet_expr_access_get_id(expr));
  if (failed(symbolTable_.find(arrayId.to_str(), scalar)))
    return failure();
  return success();
}

LogicalResult
MLIRCodegen::getSymbolInductionVar(__isl_keep pet_expr *expr,
                                   SmallVector<Value, 4> &loopIvs) {
  auto arrayId = isl::manage(pet_expr_access_get_id(expr));
  auto petArray = scop_.getArrayFromId(arrayId);
  auto indexes = isl::manage(pet_expr_access_get_index(expr));
  auto indexesPos = getIndexesPos(indexes);

  if (petArray.getDimensionality() != indexesPos.size())
    return failure();

  Value symbol;
  if (failed(getSymbol(expr, symbol)))
    return failure();

  if (loopTable_.size() < indexesPos.size())
    return failure();

  for (size_t i = 0; i < indexesPos.size(); i++) {
    Value val;
    if (failed(loopTable_.getElemAtPos(indexesPos[i], val)))
      return failure();
    loopIvs.push_back(val);
  }
  return success();
}

// TODO: check that expr is freed for all the possible
// exit points. Do we want also to free in case we return
// nullptr? Can we do a C++ wrapper around pet_expr?
Value MLIRCodegen::createLoad(__isl_take pet_expr *expr) {
  if (pet_expr_get_type(expr) != pet_expr_access)
    llvm_unreachable("expect pet_expr_access type");

  if (!isMultiDimensionalArray(expr)) {
    Value scalar;
    if (failed(getSymbol(expr, scalar))) {
      pet_expr_free(expr);
      return nullptr;
    }
    pet_expr_free(expr);
    // FIXME: handle store/load for scalar values.
    return scalar;
  }

  Value symbol;
  if (failed(getSymbol(expr, symbol))) {
    pet_expr_free(expr);
    return nullptr;
  }

  SmallVector<Value, 4> loopIvs;
  if (failed(getSymbolInductionVar(expr, loopIvs))) {
    pet_expr_free(expr);
    return nullptr;
  }

  pet_expr_free(expr);
  auto location = builder_.getUnknownLoc();
  return builder_.create<AffineLoadOp>(location, symbol, loopIvs);
}

Value MLIRCodegen::createStore(__isl_take pet_expr *expr, Value op) {
  if (pet_expr_get_type(expr) != pet_expr_access)
    llvm_unreachable("expect pet_expr_access type");
  if (!op)
    llvm_unreachable("expect non null value");

  auto location = builder_.getUnknownLoc();
  SmallVector<Value, 4> loopIvs;

  if (!isMultiDimensionalArray(expr)) {
    Value scalar;
    if (failed(getSymbol(expr, scalar))) {
      pet_expr_free(expr);
      return nullptr;
    }
    pet_expr_free(expr);
    // FIXME: handle store/load for scalar values.
    return scalar;
  }

  Value symbol;
  if (failed(getSymbol(expr, symbol))) {
    pet_expr_free(expr);
    return nullptr;
  }

  if (failed(getSymbolInductionVar(expr, loopIvs))) {
    pet_expr_free(expr);
    return nullptr;
  }

  pet_expr_free(expr);
  builder_.create<AffineStoreOp>(location, op, symbol, loopIvs);
  return op;
}

Value MLIRCodegen::createAssignmentOp(__isl_take pet_expr *expr) {
  Value rhs = createExpr(pet_expr_get_arg(expr, 1));
  if (!rhs)
    return nullptr;
  Value lhs = createStore(pet_expr_get_arg(expr, 0), rhs);
  if (!lhs)
    return nullptr;
  pet_expr_free(expr);
  return lhs;
}

Value MLIRCodegen::createBinaryOp(Location &loc, Value &lhs, Value &rhs,
                                  BinaryOpType type) {
  auto typeLhs = lhs.getType();
  auto typeRhs = rhs.getType();
  if (typeLhs != typeRhs)
    return nullptr;
  if (((!typeLhs.isInt()) && (!typeLhs.isFloat())) ||
      ((!typeRhs.isInt()) && (!typeRhs.isFloat())))
    return nullptr;
  switch (type) {
  case BinaryOpType::ADD: {
    if (typeLhs.isFloat())
      return builder_.create<AddFOp>(loc, lhs, rhs);
    else
      return builder_.create<AddIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::SUB: {
    if (typeLhs.isFloat())
      return builder_.create<SubFOp>(loc, lhs, rhs);
    else
      return builder_.create<SubIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::MUL: {
    if (typeLhs.isFloat())
      return builder_.create<MulFOp>(loc, lhs, rhs);
    else
      return builder_.create<MulIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::DIV: {
    if (typeLhs.isFloat())
      return builder_.create<DivFOp>(loc, lhs, rhs);
    else
      return nullptr;
  }
  default:
    llvm_unreachable("operation not supported yet.");
  }
}

Value MLIRCodegen::createAssignmentWithOp(__isl_take pet_expr *expr) {
  Value rhs = createExpr(pet_expr_get_arg(expr, 1));
  if (!rhs)
    return nullptr;
  Value rhsLoad = createLoad(pet_expr_get_arg(expr, 0));
  if (!rhsLoad)
    return nullptr;

  auto location = builder_.getUnknownLoc();
  Value op;
  switch (pet_expr_op_get_type(expr)) {
  case pet_op_mul_assign: {
    op = createBinaryOp(location, rhs, rhsLoad, BinaryOpType::MUL);
    break;
  }
  case pet_op_add_assign: {
    op = createBinaryOp(location, rhs, rhsLoad, BinaryOpType::ADD);
    break;
  }
  case pet_op_sub_assign: {
    op = createBinaryOp(location, rhs, rhsLoad, BinaryOpType::SUB);
    break;
  }
  case pet_op_div_assign: {
    op = createBinaryOp(location, rhs, rhsLoad, BinaryOpType::DIV);
    break;
  }
  case pet_op_and_assign:
  case pet_op_xor_assign:
  case pet_op_or_assign: {
    llvm_unreachable("assignement with operation not implemented");
    return nullptr;
  }
  default: {
    llvm_unreachable("unexpected operation");
    return nullptr;
  }
  }
  if (!op)
    return nullptr;
  Value lhs = createStore(pet_expr_get_arg(expr, 0), op);
  if (!lhs)
    return nullptr;
  pet_expr_free(expr);
  return lhs;
}

// TODO: here we need to check the type of the variable
// we are incrementing. For now we assume only float.
Value MLIRCodegen::createPostInc(__isl_take pet_expr *expr) {
  auto loc = builder_.getUnknownLoc();
  Value rhs = createExpr(pet_expr_get_arg(expr, 0));
  if (!rhs)
    return nullptr;
  Value constant = createConstantFloatOp(1.0, loc);
  Value operation = nullptr;
  switch (pet_expr_op_get_type(expr)) {
  case pet_op_post_inc: {
    operation = createBinaryOp(loc, rhs, constant, BinaryOpType::ADD);
    break;
  }
  case pet_op_post_dec: {
    operation = createBinaryOp(loc, rhs, constant, BinaryOpType::SUB);
    break;
  }
  default:
    llvm_unreachable("handle only post_inc and post_dec");
  }

  Value lhs = createStore(pet_expr_get_arg(expr, 0), operation);
  if (!lhs)
    return nullptr;
  pet_expr_free(expr);
  return lhs;
}

// TODO: check pet_expr_free, there is a better way of doing it?
Value MLIRCodegen::createOp(__isl_take pet_expr *expr) {
  // std::cout << __func__ << std::endl;
  // handle pet_*_assing
  if (pet_expr_op_get_type(expr) == pet_op_assign)
    return createAssignmentOp(expr);
  if ((pet_expr_op_get_type(expr) == pet_op_add_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_sub_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_mul_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_div_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_and_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_xor_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_or_assign))
    return createAssignmentWithOp(expr);

  if ((pet_expr_op_get_type(expr) == pet_op_post_inc) ||
      (pet_expr_op_get_type(expr) == pet_op_post_dec)) {
    return createPostInc(expr);
  }

  Value lhs = createExpr(pet_expr_get_arg(expr, 0));
  if (!lhs)
    return nullptr;
  Value rhs = createExpr(pet_expr_get_arg(expr, 1));
  if (!rhs)
    return nullptr;
  auto location = builder_.getUnknownLoc();

  switch (pet_expr_op_get_type(expr)) {
  case pet_op_add_assign:
  case pet_op_sub_assign:
  case pet_op_mul_assign:
  case pet_op_div_assign:
  case pet_op_and_assign:
  case pet_op_xor_assign:
  case pet_op_or_assign:
  case pet_op_assign: {
    llvm_unreachable("unexpected pet_expr_op here");
    return nullptr;
  }
  case pet_op_add: {
    pet_expr_free(expr);
    return createBinaryOp(location, lhs, rhs, BinaryOpType::ADD);
  }
  case pet_op_sub: {
    pet_expr_free(expr);
    return createBinaryOp(location, lhs, rhs, BinaryOpType::SUB);
  }
  case pet_op_mul: {
    pet_expr_free(expr);
    return createBinaryOp(location, lhs, rhs, BinaryOpType::MUL);
  }
  case pet_op_div: {
    pet_expr_free(expr);
    return createBinaryOp(location, lhs, rhs, BinaryOpType::DIV);
  }
  case pet_op_mod:
  case pet_op_shl:
  case pet_op_shr:
  case pet_op_eq:
  case pet_op_ne:
  case pet_op_le:
  case pet_op_ge:
  case pet_op_lt:
  case pet_op_gt:
  case pet_op_minus:
  case pet_op_pre_inc:
  case pet_op_pre_dec:
  case pet_op_address_of:
  case pet_op_assume:
  case pet_op_kill:
  case pet_op_and:
  case pet_op_xor:
  case pet_op_or:
  case pet_op_not:
  case pet_op_land:
  case pet_op_lor:
  case pet_op_lnot:
  case pet_op_cond:
  case pet_op_last: {
    llvm_unreachable("operation not handled");
    return nullptr;
  }
  case pet_op_post_inc:
  case pet_op_post_dec: {
    llvm_unreachable("not expected here");
    return nullptr;
  }
  }
  emitError(location, "invalid binary operator");
  return nullptr;
}

Value MLIRCodegen::createConstantOp(__isl_take pet_expr *expr,
                                    ElementType type) {
  auto loc = builder_.getUnknownLoc();
  switch (type) {
  case ElementType::INT: {
    isl::val value = isl::manage(pet_expr_int_get_val(expr));
    int valueAsInt = std::stoi(value.to_str());
    auto valueAttr =
        builder_.getIntegerAttr(builder_.getIntegerType(32), valueAsInt);
    pet_expr_free(expr);
    return builder_.create<ConstantOp>(loc, builder_.getIntegerType(32),
                                       valueAttr);
  }
  case ElementType::FLOAT: {
    float valueAsFloat = std::stof(std::string(pet_expr_double_get_str(expr)));
    pet_expr_free(expr);
    return createConstantFloatOp(valueAsFloat, loc);
  }
  case ElementType::DOUBLE: {
    // XXX: here pet only exposes get_str method, why?
    double valueAsDouble =
        std::stod(std::string(pet_expr_double_get_str(expr)));
    auto valueAttr =
        builder_.getFloatAttr(builder_.getF64Type(), valueAsDouble);
    pet_expr_free(expr);
    return builder_.create<ConstantOp>(loc, builder_.getF64Type(), valueAttr);
  }
  }
  return nullptr;
}

Value MLIRCodegen::createConstantFloatOp(float val, Location &loc) {
  auto valueAttr = builder_.getFloatAttr(builder_.getF32Type(), val);
  return builder_.create<ConstantOp>(loc, builder_.getF32Type(), valueAttr);
}

Value MLIRCodegen::createExpr(__isl_keep pet_expr *expr) {
  switch (pet_expr_get_type(expr)) {
  case pet_expr_error:
    return nullptr;
  case pet_expr_access:
    return createLoad(expr);
  case pet_expr_int:
    return createConstantOp(expr, ElementType::INT);
  case pet_expr_double:
    // XXX: How to distinguish FLOAT and DOUBLE, here?
    return createConstantOp(expr, ElementType::FLOAT);
  case pet_expr_call:
  case pet_expr_cast: {
    llvm_unreachable("type not handled");
    return nullptr;
  }
  case pet_expr_op:
    return createOp(expr);
  }

  return nullptr;
}

LogicalResult MLIRCodegen::createStmt(__isl_keep pet_expr *expr) {
  // pet_expr_dump(expr);
  auto Value = createExpr(expr);
  if (!Value)
    return failure();
  return success();
}

static mlir::Type getType(const PetArray &array, MLIRContext &context) {
  auto type = array.getType();
  switch (type) {
  case ElementType::FLOAT:
    return FloatType::get(StandardTypes::F32, &context);
  case ElementType::DOUBLE:
    return FloatType::get(StandardTypes::F64, &context);
  case ElementType::INT:
    return IntegerType::get(32, &context);
  }
  llvm_unreachable("unknown type");
}

// TODO: handle properly affineMapComposition while
// building class MemRefType.
Type MLIRCodegen::getTensorType(MLIRContext &context,
                                const PetArray &inputTensor) {
  auto tensorType = getType(inputTensor, context);
  SmallVector<int64_t, 4> shape;
  size_t dimensionality = inputTensor.getDimensionality();
  for (size_t i = 0; i < dimensionality; i++)
    shape.push_back(inputTensor.getExtentOnDimension(i));
  if (dimensionality)
    return MemRefType::get(shape, tensorType);
  return tensorType;
}

SmallVector<Type, 8> MLIRCodegen::getFunctionArgumentsTypes(
    MLIRContext &context, const SmallVector<PetArray, 4> &inputTensors) {
  SmallVector<Type, 8> argTypes;
  if (!inputTensors.size())
    return {};
  for (const auto inputTensor : inputTensors)
    argTypes.push_back(getTensorType(context, inputTensor));
  return argTypes;
}

LogicalResult MLIRCodegen::declare(std::string id, mlir::Value value) {
  if (succeeded(symbolTable_.find(id)))
    return failure();
  symbolTable_.insert(id, value);
  return success();
}

MLIRCodegen::MLIRCodegen(MLIRContext &context, Scop &scop)
    : scop_(scop), builder_(&context) {
  theModule_ = ModuleOp::create(builder_.getUnknownLoc());
  auto inputTensors = scop_.getInputTensors();
  auto argTypes = getFunctionArgumentsTypes(context, inputTensors);
  auto funcType = builder_.getFunctionType(argTypes, llvm::None);
  FuncOp function(
      FuncOp::create(builder_.getUnknownLoc(), "scop_entry", funcType));
  if (!function)
    llvm_unreachable("failed to create scop_entry function");
  auto &entryBlock = *function.addEntryBlock();

  // declare all the function arguments in the symbol table.
  for (const auto &nameValue : zip(inputTensors, entryBlock.getArguments())) {
    if (failed(
            declare(std::get<0>(nameValue).getName(), std::get<1>(nameValue))))
      llvm_unreachable("failed to declare function arguments");
  }

  builder_.setInsertionPointToStart(&entryBlock);
  theModule_.push_back(function);
}

void MLIRCodegen::dump() { theModule_.dump(); }
void MLIRCodegen::print(raw_ostream &os) { theModule_.print(os); }

LogicalResult MLIRCodegen::verifyModule() {
  if (failed(verify(theModule_)))
    return failure();
  return success();
}

AffineForOp MLIRCodegen::createLoop(int upperBound, int lowerBound, int step) {
  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), upperBound,
                                           lowerBound, step);
  loop.getBody()->clear();
  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineTerminatorOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());
  return loop;
}

void MLIRCodegen::createReturn() {
  builder_.create<ReturnOp>(builder_.getUnknownLoc());
}

void MLIRCodegen::setInsertionPointAfter(AffineForOp *affineForOp) {
  builder_.setInsertionPointAfter(affineForOp->getOperation());
}

LogicalResult codegen::SymbolTable::insert(std::string id, Value value) {
  auto it = symbolTable_.find(id);
  if (it != symbolTable_.end())
    return failure();
  symbolTable_.insert(std::pair<std::string, Value>(id, value));
  return success();
}

LogicalResult codegen::SymbolTable::erase(std::string id) {
  auto it = symbolTable_.find(id);
  if (it == symbolTable_.end())
    return failure();
  symbolTable_.erase(id);
  return success();
}

LogicalResult codegen::SymbolTable::find(std::string id, Value &val) {
  auto it = symbolTable_.find(id);
  if (it == symbolTable_.end())
    return failure();
  val = it->second;
  return success();
}

LogicalResult codegen::SymbolTable::find(std::string id) {
  Value dummy;
  return find(id, dummy);
}

LogicalResult LoopTable::getElemAtPos(size_t pos, Value &value) {
  if (pos > size())
    return failure();
  auto it = begin();
  std::advance(it, pos);
  value = it->second;
  return success();
}

size_t codegen::SymbolTable::size() const { return symbolTable_.size(); }

void codegen::SymbolTable::dump() const {
  outs() << "Loop table: \n";
  outs() << "   size: " << symbolTable_.size() << "\n";
  for (auto it = symbolTable_.begin(); it != symbolTable_.end(); it++)
    outs() << "   id: " << it->first << "\n";
}
