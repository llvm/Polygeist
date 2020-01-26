#include "mlirCodegen.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"

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
    if (failed(getSymbol(expr, scalar)))
      return nullptr;
    pet_expr_free(expr);
    return scalar;
  }

  Value symbol;
  if (failed(getSymbol(expr, symbol)))
    return nullptr;

  SmallVector<Value, 4> loopIvs;
  if (failed(getSymbolInductionVar(expr, loopIvs)))
    return nullptr;

  pet_expr_free(expr);
  auto location = builder_.getUnknownLoc();
  return builder_.create<AffineLoadOp>(location, symbol, loopIvs);
}

Value MLIRCodegen::createStore(__isl_take pet_expr *expr, Value op) {
  if (pet_expr_get_type(expr) != pet_expr_access)
    llvm_unreachable("expect pet_expr_access type");
  if (!op)
    llvm_unreachable("expect non null value");

  if (!isMultiDimensionalArray(expr)) {
    Value scalar;
    if (failed(getSymbol(expr, scalar)))
      return nullptr;
    pet_expr_free(expr);
    return scalar;
  }

  Value symbol;
  if (failed(getSymbol(expr, symbol)))
    return nullptr;

  SmallVector<Value, 4> loopIvs;
  if (failed(getSymbolInductionVar(expr, loopIvs)))
    return nullptr;

  pet_expr_free(expr);
  auto location = builder_.getUnknownLoc();
  builder_.create<AffineStoreOp>(location, op, symbol, loopIvs);
  return op;
}

Value MLIRCodegen::createAssignementOp(__isl_take pet_expr *expr) {
  Value rhs = createExpr(pet_expr_get_arg(expr, 1));
  if (!rhs)
    return nullptr;
  Value lhs = createStore(pet_expr_get_arg(expr, 0), rhs);
  if (!lhs)
    return nullptr;
  return lhs;
}

Value MLIRCodegen::createAssignementWithOp(__isl_take pet_expr *expr) {
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
    op = builder_.create<MulFOp>(location, rhs, rhsLoad);
    break;
  }
  case pet_op_add_assign: {
    op = builder_.create<AddFOp>(location, rhs, rhsLoad);
    break;
  }
  case pet_op_sub_assign: {
    op = builder_.create<SubFOp>(location, rhs, rhsLoad);
    break;
  }
  case pet_op_div_assign: {
    op = builder_.create<DivFOp>(location, rhs, rhsLoad);
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
  return lhs;
}

// TODO: check pet_expr_free, there is a better way of doing it?
// TODO: add remaining corner cases and op types.
Value MLIRCodegen::createOp(__isl_take pet_expr *expr) {
  // handle corner cases (i.e., pet_*_assing)
  if (pet_expr_op_get_type(expr) == pet_op_assign)
    return createAssignementOp(expr);
  if ((pet_expr_op_get_type(expr) == pet_op_add_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_sub_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_mul_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_div_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_and_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_xor_assign) ||
      (pet_expr_op_get_type(expr) == pet_op_or_assign))
    return createAssignementWithOp(expr);

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
    return builder_.create<AddFOp>(location, lhs, rhs);
  }
  case pet_op_sub: {
    pet_expr_free(expr);
    return builder_.create<SubFOp>(location, lhs, rhs);
  }
  case pet_op_mul: {
    pet_expr_free(expr);
    return builder_.create<MulFOp>(location, lhs, rhs);
  }
  case pet_op_div:
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
  case pet_op_post_inc:
  case pet_op_post_dec:
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
  }
  emitError(location, "invalid binary operator");
  return nullptr;
}

Value MLIRCodegen::createExpr(__isl_take pet_expr *expr) {
  switch (pet_expr_get_type(expr)) {
  case pet_expr_error:
    return nullptr;
  case pet_expr_access:
    return createLoad(expr);
  case pet_expr_call:
  case pet_expr_cast:
  case pet_expr_int:
  case pet_expr_double: {
    outs() << "not handled!";
    return nullptr;
  }
  case pet_expr_op:
    return createOp(expr);
  }
  return nullptr;
}

LogicalResult MLIRCodegen::createStmt(__isl_take pet_expr *expr) {
  auto Value = createExpr(expr);
  pet_expr_free(expr);
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
// TODO: for variable we create memref<f64> is this correct?
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

LogicalResult MLIRCodegen::verifyModule() {
  if (failed(verify(theModule_)))
    return failure();
  return success();
}

AffineForOp MLIRCodegen::createLoop(int upperBound, int lowerBound, int step) {
  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lowerBound,
                                           upperBound, step);
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
