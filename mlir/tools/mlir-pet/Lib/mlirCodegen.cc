#include "mlirCodegen.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace codegen;
using namespace mlir;
using namespace llvm;
using namespace pet;

#define DEBUG_TYPE "pet-to-mlir-codegen"

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

size_t MLIRCodegen::getDimensionalityExpr(__isl_keep pet_expr *expr) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access");
  auto idArray = isl::manage(pet_expr_access_get_id(expr));
  return scop_.getArrayFromId(idArray).getDimensionality();
}

bool MLIRCodegen::isMultiDimensionalArray(__isl_keep pet_expr *expr) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access type");
  auto dims = getDimensionalityExpr(expr);
  if (dims != 0)
    return true;
  return false;
}

// Convert the given pet_expr to the
// corresponding MemRefType. Since pet_expr does not
// carry any type information the type should be provided
// to this function.
MemRefType MLIRCodegen::convertExprToMemRef(__isl_keep pet_expr *expr,
                                            Type t) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access");
  auto dims = getDimensionalityExpr(expr);
  // for scalar create a memref<1xtype>
  if (dims == 0)
    return MemRefType::get(1, t);

  auto idArray = isl::manage(pet_expr_access_get_id(expr));
  auto petArray = scop_.getArrayFromId(idArray);
  assert((dims == petArray.getDimensionality()) && "must be equal");
  std::vector<int64_t> extent;
  for (size_t i = 0; i < dims; i++)
    extent.push_back(petArray.getExtentOnDimension(i));

  return MemRefType::get(extent, t);
}

LogicalResult MLIRCodegen::getSymbol(__isl_keep pet_expr *expr,
                                     Value &scalar) const {
  auto arrayId = isl::manage(pet_expr_access_get_id(expr));
  if (failed(symbolTable_.find(arrayId.to_str(), scalar)))
    return failure();
  return success();
}

LogicalResult MLIRCodegen::isInSymbolTable(__isl_keep pet_expr *expr) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto id = isl::manage(pet_expr_access_get_id(expr));
  if (failed(symbolTable_.find(id.to_str())))
    return failure();
  return success();
}

LogicalResult
MLIRCodegen::getSymbolInductionVar(__isl_keep pet_expr *expr,
                                   SmallVector<Value, 4> &loopIvs) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
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
// FIXME: pet allows also expression that are like:
// ref_id: __pet_ref_3
// index: { S_2[] -> [(123)] }
// depth: 1
// read: 1
// write: 0
// to be of type pet_expr_access. This will
// trigger 'tuple has not id' in isMultiDimensionalArray.
// To reproduce use:
// int main() {
//  int i = 10;
//  i = i + 23;
// }

Value MLIRCodegen::createLoad(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access type");

  auto location = builder_.getUnknownLoc();
  if (!isMultiDimensionalArray(expr)) {
    Value scalar;
    if (failed(getSymbol(expr, scalar))) {
      pet_expr_free(expr);
      return nullptr;
    }
    pet_expr_free(expr);
    // emit the load only if we are dealing with a memref.
    if (scalar.getType().dyn_cast<MemRefType>()) {
      Value zeroIndex = builder_.create<ConstantIndexOp>(location, 0);
      scalar = builder_.create<AffineLoadOp>(location, scalar, zeroIndex);
    }
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
  return builder_.create<AffineLoadOp>(location, symbol, loopIvs);
}

Value MLIRCodegen::createStore(__isl_take pet_expr *expr, Value op) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access type");
  assert(op && "expect non null value");

  auto location = builder_.getUnknownLoc();
  SmallVector<Value, 4> loopIvs;

  if (!isMultiDimensionalArray(expr)) {
    Value scalar;
    Value zeroIndex = builder_.create<ConstantIndexOp>(location, 0);
    if (failed(getSymbol(expr, scalar))) {
      // if not in symbol table allocate
      // the scalar.
      scalar = createAllocOp(expr, op.getType(), op);
    }
    builder_.create<AffineStoreOp>(location, op, scalar, zeroIndex);
    pet_expr_free(expr);
    return op;
  }

  // if the array is not in the symbol
  // table allocate it.
  Value symbol;
  if (failed(getSymbol(expr, symbol))) {
    symbol = createAllocOp(expr, op.getType(), op);
  }

  if (failed(getSymbolInductionVar(expr, loopIvs))) {
    pet_expr_free(expr);
    return nullptr;
  }

  pet_expr_free(expr);
  builder_.create<AffineStoreOp>(location, op, symbol, loopIvs);
  return op;
}

// Value is null if we come from a pet_kill_op, while
// it is not null if we come from the createStoreOp.
// This is because kill_stmt can be removed when
// rescheduling.
Value MLIRCodegen::createAllocOp(__isl_keep pet_expr *expr, Type t, Value v) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access");
  auto location = builder_.getUnknownLoc();
  auto memRefType = convertExprToMemRef(expr, t);
  if (v) {
    auto operation = v.getDefiningOp();
    auto func = operation->getParentOfType<FuncOp>();
    // insert the allocations as first operations
    // in the scop FuncOp.
    builder_.setInsertionPointToStart(&func.front());
  }
  auto alloc = builder_.create<AllocOp>(location, memRefType);
  if (v) {
    auto operation = v.getDefiningOp();
    builder_.setInsertionPointAfter(operation);
  }
  auto id = isl::manage(pet_expr_access_get_id(expr));
  symbolTable_.insert(id.to_str(), alloc);
  return alloc;
}

Value MLIRCodegen::createAssignmentOp(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  pet_expr_dump(expr);

  // get type for lhs.
  auto lhsPetExpr = pet_expr_get_arg(expr, 0);
  Value symbolLhs = nullptr;
  if (failed(getSymbol(lhsPetExpr, symbolLhs)))
    llvm_unreachable("symbol must be available in symbol table");
  pet_expr_free(lhsPetExpr);

  Value rhs = createExpr(pet_expr_get_arg(expr, 1), symbolLhs.getType());
  if (!rhs)
    return nullptr;

  Value lhs = createStore(pet_expr_get_arg(expr, 0), rhs);
  if (!lhs)
    return nullptr;

  pet_expr_free(expr);
  return lhs;
}

static bool isInt(Type type) { return type.isa<IntegerType>(); }

static bool isFloat(Type type) { return type.isa<FloatType>(); }

Value MLIRCodegen::createBinaryOp(Location &loc, Value &lhs, Value &rhs,
                                  BinaryOpType type) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto typeLhs = lhs.getType();
  auto typeRhs = rhs.getType();
  if (typeLhs != typeRhs)
    return nullptr;
  if (((!isInt(typeLhs)) && (!isFloat(typeLhs))) ||
      ((!isInt(typeRhs)) && (!isFloat(typeRhs))))
    return nullptr;

  auto isLhsFloat = isFloat(typeLhs);
  switch (type) {
  case BinaryOpType::ADD: {
    if (isLhsFloat)
      return builder_.create<AddFOp>(loc, lhs, rhs);
    else
      return builder_.create<AddIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::SUB: {
    if (isLhsFloat)
      return builder_.create<SubFOp>(loc, lhs, rhs);
    else
      return builder_.create<SubIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::MUL: {
    if (isLhsFloat)
      return builder_.create<MulFOp>(loc, lhs, rhs);
    else
      return builder_.create<MulIOp>(loc, lhs, rhs);
  }
  case BinaryOpType::DIV: {
    if (isLhsFloat)
      return builder_.create<DivFOp>(loc, lhs, rhs);
    else
      return nullptr;
  }
  default:
    llvm_unreachable("operation not supported yet.");
  }
}

Value MLIRCodegen::createAssignmentWithOp(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
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
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto loc = builder_.getUnknownLoc();
  Value rhs = createExpr(pet_expr_get_arg(expr, 0));
  if (!rhs)
    return nullptr;

  Value constant = nullptr;
  auto rhsType = rhs.getType();
  if (rhsType.isF32())
    constant = createConstantFloatOp(1, loc);
  if (rhsType.isF64())
    constant = createConstantDoubleOp(1, loc);
  if (rhsType.isInteger(32))
    constant = createConstantIntOp(1, loc);

  assert(constant && "unsupported type: expect F32, F64 or int (32 bit)");

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

Value MLIRCodegen::createDefinition(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_n_arg(expr) == 1) && "expect single arg for kill");
  auto arg = pet_expr_get_arg(expr, 0);
  auto argType = pet_expr_get_type(arg);
  assert((argType == pet_expr_access) &&
         "expect pet_expr_access as arg for kill");
  auto idArray = isl::manage(pet_expr_access_get_id(arg));
  auto petArray = scop_.getArrayFromId(idArray);
  auto elementType = petArray.getType();

  pet_expr_free(expr);

  Value allocation = nullptr;
  switch (elementType) {
  case ElementType::FLOAT: {
    allocation = createAllocOp(arg, builder_.getF32Type());
    break;
  }
  case ElementType::DOUBLE: {
    allocation = createAllocOp(arg, builder_.getF64Type());
    break;
  }
  case ElementType::INT: {
    allocation = createAllocOp(arg, builder_.getIntegerType(32));
    break;
  }
  }
  pet_expr_free(arg);
  return allocation;
}

// TODO: check pet_expr_free, there is a better way of doing it?
Value MLIRCodegen::createOp(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
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

  if ((pet_expr_op_get_type(expr) == pet_op_kill)) {
    return createDefinition(expr);
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
  case pet_op_kill:
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
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto loc = builder_.getUnknownLoc();
  switch (type) {
  case ElementType::INT: {
    isl::val value = isl::manage(pet_expr_int_get_val(expr));
    int valueAsInt = std::stoi(value.to_str());
    pet_expr_free(expr);
    return createConstantIntOp(valueAsInt, loc);
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
    pet_expr_free(expr);
    return createConstantDoubleOp(valueAsDouble, loc);
  }
  }
  return nullptr;
}

Value MLIRCodegen::createConstantFloatOp(float val, Location &loc) {
  auto valueAttr = builder_.getFloatAttr(builder_.getF32Type(), val);
  return builder_.create<ConstantOp>(loc, builder_.getF32Type(), valueAttr);
}

Value MLIRCodegen::createConstantDoubleOp(double val, Location &loc) {
  auto valueAttr = builder_.getFloatAttr(builder_.getF64Type(), val);
  return builder_.create<ConstantOp>(loc, builder_.getF64Type(), valueAttr);
}

Value MLIRCodegen::createConstantIntOp(int val, Location &loc) {
  auto valueAttr = builder_.getIntegerAttr(builder_.getIntegerType(32), val);
  return builder_.create<ConstantOp>(loc, builder_.getIntegerType(32),
                                     valueAttr);
}

// insert a symbol reference to "fName", inserting it into the module
// if necessary.
static FlatSymbolRefAttr
getOrInsertFunction(OpBuilder &rewriter, ModuleOp module, std::string fName,
                    const llvm::ArrayRef<mlir::Type> typeOperands,
                    const llvm::ArrayRef<mlir::Type> typeResults = {}) {
  auto *context = module.getContext();
  if (module.lookupSymbol(fName))
    return SymbolRefAttr::get(fName, context);
  auto libFnInfoType =
      FunctionType::get(typeOperands, typeResults, rewriter.getContext());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  rewriter.create<FuncOp>(module.getLoc(), fName, libFnInfoType,
                          llvm::ArrayRef<mlir::NamedAttribute>{});
  return mlir::SymbolRefAttr::get(fName, context);
}

Value MLIRCodegen::createCallOp(__isl_take pet_expr *expr) {
  auto nameFunc = std::string(pet_expr_call_get_name(expr));
  assert((nameFunc == "print_memref_f32") && "only print name");
  assert((pet_expr_get_n_arg(expr) == 1) && "must have 1 arg only");

  auto subExpr = pet_expr_get_arg(expr, 0);
  assert((pet_expr_get_type(subExpr) == pet_expr_access) &&
         "expect pet_expr_access");

  Value symbol = nullptr;
  if (failed(getSymbol(subExpr, symbol)))
    llvm_unreachable("must be in symbol table");

  // for now we allow only memref.
  auto memRef = symbol.getType().dyn_cast_or_null<MemRefType>();
  if (!memRef) {
    LLVM_DEBUG(dbgs() << "createCallOp supports only memref");
    return nullptr;
  }

  // enforce memref to be a f32.
  auto elemType = memRef.getElementType();
  if (!elemType.isF32()) {
    LLVM_DEBUG(
        dbgs()
        << "createCallOp supports only memref with elements of type F32");
    return nullptr;
  }

  // cast the memref to unranked type.
  auto loc = builder_.getUnknownLoc();
  auto newMemRefType =
      UnrankedMemRefType::get(memRef.getElementType(), memRef.getMemorySpace());
  auto castedMemRef = builder_.create<MemRefCastOp>(loc, symbol, newMemRefType);

  // insert the function.
  auto module = castedMemRef.getParentOfType<ModuleOp>();
  auto symbolFn =
      getOrInsertFunction(builder_, module, "print_memref_f32",
                          llvm::ArrayRef<Type>{castedMemRef.getType()});
  builder_.create<CallOp>(loc, symbolFn, /*return type*/ llvm::ArrayRef<Type>{},
                          llvm::ArrayRef<Value>{castedMemRef});

  pet_expr_free(subExpr);
  pet_expr_free(expr);
  return symbol;
}

static ElementType getElementTypeFromMLIRType(Type t) {
  // default.
  if (!t)
    return ElementType::FLOAT;
  // type is valid and must be a memref.
  auto memRef = t.dyn_cast<MemRefType>();
  assert(memRef && "expect memref type for conversion");
  auto memRefElemType = memRef.getElementType();
  if (memRefElemType.isF32())
    return ElementType::FLOAT;
  if (memRefElemType.isF64())
    return ElementType::DOUBLE;
  llvm_unreachable("expect type F32 or F64");
  return ElementType::FLOAT;
}

Value MLIRCodegen::createExpr(__isl_keep pet_expr *expr, Type t) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  switch (pet_expr_get_type(expr)) {
  case pet_expr_error:
    return nullptr;
  case pet_expr_access:
    return createLoad(expr);
  case pet_expr_int:
    return createConstantOp(expr, ElementType::INT);
  case pet_expr_double: {
    auto floatType = getElementTypeFromMLIRType(t);
    return createConstantOp(expr, floatType);
  }
  case pet_expr_call:
    return createCallOp(expr);
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
  LLVM_DEBUG(dbgs() << __func__ << "\n");
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
  auto inputTensors = scop_.getInputArrays();
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
  if (failed(verify(theModule_))) {
    theModule_.dump();
    return failure();
  }
  return success();
}

AffineForOp MLIRCodegen::createLoop(int lb, int ub, int step) {

  auto loop =
      builder_.create<AffineForOp>(builder_.getUnknownLoc(), lb, ub, step);
  loop.getBody()->clear();
  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineTerminatorOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  return loop;
}

AffineForOp MLIRCodegen::createLoop(int lb, std::string ub_id, int step) {

  Value ub;
  if (failed(this->getLoopTable().find(ub_id, ub)))
    llvm_unreachable("Couldn't find the bound in the loop table.");

  auto lbMap = AffineMap::getConstantMap(lb, builder_.getContext());
  auto ubMap = AffineMap::getMultiDimIdentityMap(1, builder_.getContext());

  ValueRange ubOperands = ValueRange(ub);
  ValueRange lbOperands = {};
  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
  loop.getBody()->clear();

  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineTerminatorOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  return loop;
}

AffineForOp MLIRCodegen::createLoop(std::string lb_id, int ub, int step) {
  Value lb;
  if (failed(this->getLoopTable().find(lb_id, lb)))
    llvm_unreachable("Couldn't find the bound in the loop table.");

  auto ubMap = AffineMap::getConstantMap(ub, builder_.getContext());
  auto lbMap = AffineMap::getMultiDimIdentityMap(1, builder_.getContext());

  ValueRange ubOperands = {};
  ValueRange lbOperands = ValueRange(lb);

  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
  loop.getBody()->clear();

  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineTerminatorOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  return loop;
}

AffineForOp MLIRCodegen::createLoop(std::string lb_id, std::string ub_id,
                                    int step) {
  Value ub, lb;
  if (failed(this->getLoopTable().find(ub_id, ub)))
    llvm_unreachable("Couldn't find the bound in the loop table.");
  if (failed(this->getLoopTable().find(lb_id, lb)))
    llvm_unreachable("Couldn't find the bound in the loop table.");

  auto lbMap = AffineMap::getMultiDimIdentityMap(1, builder_.getContext());
  auto ubMap = AffineMap::getMultiDimIdentityMap(1, builder_.getContext());

  ValueRange ubOperands = ValueRange(ub);
  ValueRange lbOperands = ValueRange(lb);
  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
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

LogicalResult codegen::SymbolTable::find(std::string id, Value &val) const {
  auto it = symbolTable_.find(id);
  if (it == symbolTable_.end())
    return failure();
  val = it->second;
  return success();
}

LogicalResult codegen::SymbolTable::find(std::string id) const {
  Value dummy;
  return find(id, dummy);
}

LogicalResult LoopTable::getElemAtPos(size_t pos, Value &value) const {
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
