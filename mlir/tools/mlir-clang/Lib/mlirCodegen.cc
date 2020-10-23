#include "mlirCodegen.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace codegen;
using namespace mlir;
using namespace llvm;
using namespace pet;

#define DEBUG_TYPE "pet-to-mlir-codegen"

LoopTable &MLIRCodegen::getLoopTable() { return loopTable_; }

static int getConstantValFromPwaff(isl::pw_aff pwaff) {
  int res = 0;
  pwaff.foreach_piece([&res](isl::set s, isl::aff a) -> isl_stat {
    auto val = a.get_constant_val();
    res = std::stoi(val.to_str());
    return isl_stat_ok;
  });
  return res;
}

// helper function to get the indexes of the output dimensions
// by comparing them with the input dimensions. This function
// assume "muaff" *not* to be scheduled.
LogicalResult MLIRCodegen::getIndexes(isl::multi_pw_aff muaff,
                                      SmallVector<Value, 4> &loopIvs) {
  auto space = muaff.get_space();
  auto spaceDimIn = space.dim(isl::dim::in);
  assert(loopTable_.size() == spaceDimIn);
  // map islId with petId for induction variables.
  // i.e., S[i, j, k] with a loop table containing
  // c0, c1, c2, will map to i = c0 - j = c1 - k = c2
  for (size_t i = 0; i < spaceDimIn; i++) {
    assert(space.has_dim_id(isl::dim::in, i));
    auto dimId = space.get_dim_id(isl::dim::in, i).to_str();
    auto petId = dimId.substr(0, dimId.find("@"));
    std::string islId;
    if (failed(loopTable_.getIdAtPos(i, islId)))
      llvm_unreachable("index not found in symbol table.");
    loopTable_.insertMapping(petId, islId);
  }
  auto spaceDimOut = muaff.get_space().dim(isl::dim::out);
  if (spaceDimOut == 0)
    llvm_unreachable("expect multi-dimensional array");
  auto loc = builder_.getUnknownLoc();
  auto umap = isl::union_map::from(muaff);
  if (umap.n_map() != 1)
    return failure();
  auto map = isl::map::from_union_map(umap);
  for (size_t i = 0; i < map.dim(isl::dim::out); i++) {
    isl::pw_aff pwaff = muaff.get_pw_aff(i);
    if (pwaff.is_cst()) {
      auto constantIndex =
          builder_.create<ConstantIndexOp>(loc, getConstantValFromPwaff(pwaff));
      loopIvs.push_back(constantIndex);
      continue;
    }
    pwaff.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
      for (size_t j = 0; j < map.dim(isl::dim::in); j++) {
        auto val = a.get_coefficient_val(isl::dim::in, j);
        if (!val.is_zero()) {
          Value v = nullptr;
          std::string valueId = "null";
          if (failed(loopTable_.getValueAtPos(j, v)))
            llvm_unreachable("index not found in symbol table.");
          loopIvs.push_back(v);
        }
      }
      return isl_stat_ok;
    });
  }
  return success();
}

size_t MLIRCodegen::getDimensionalityExpr(__isl_keep pet_expr *expr) const {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_type(expr) == pet_expr_access) &&
         "expect pet_expr_access");
  auto idArray = isl::manage(pet_expr_access_get_id(expr));
  // if the expression has no id, it is not an array.
  if (!idArray)
    return 0;
  // return scop_.getArrayFromId(idArray).getDimensionality();
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
  /*
  auto petArray = scop_.getArrayFromId(idArray);
  assert((dims == petArray.getDimensionality()) && "must be equal");
  std::vector<int64_t> extent;
  for (size_t i = 0; i < dims; i++)
    extent.push_back(petArray.getExtentOnDimension(i));

  return MemRefType::get(extent, t);
  */
}

LogicalResult MLIRCodegen::getSymbol(__isl_keep pet_expr *expr,
                                     Value &scalar) const {
  auto arrayId = isl::manage(pet_expr_access_get_id(expr));
  if (!arrayId) {
    // in case we don't have any id, we are dealing with
    // in induction variable expression. Currently, we
    // assume a simple form of such expression, with a
    // single induction variable.
    auto index = isl::manage(pet_expr_access_get_index(expr));
    assert(index.dim(isl::dim::out) == 1);
    auto pwaff = index.get_pw_aff(0);
    assert(pwaff.n_piece() == 1);
    std::string indVarSymbol;
    auto getIndVarId = [&indVarSymbol](isl::set set, isl::aff aff) {
      for (int i = 0; i < aff.dim(isl::dim::in); i++) {
        isl::val coeff = aff.get_coefficient_val(isl::dim::in, i);
        if (!coeff.is_zero())
          indVarSymbol = std::string(aff.get_dim_name(isl::dim::in, i));
      }
      return isl_stat_ok;
    };
    pwaff.foreach_piece(getIndVarId);
    if (failed(symbolTable_.find(indVarSymbol, scalar)))
      return failure();
    return success();
  }
  if (failed(symbolTable_.find(arrayId.to_str(), scalar)))
    return failure();
  return success();
}

LogicalResult MLIRCodegen::getSymbolInductionVar(__isl_keep pet_expr *expr,
                                                 Value &indVar) const {
  auto indVarId = isl::manage(pet_expr_access_get_id(expr));
  if (failed(loopTable_.lookUpPetMapping(indVarId.to_str())))
    return failure();
  if (failed(symbolTable_.find(indVarId.to_str(), indVar)))
    return failure();
  return success();
}

LogicalResult MLIRCodegen::getSymbolInductionVar(std::string id,
                                                 Value &indVar) const {
  std::string petId = "null";
  if (failed(loopTable_.lookUpIslMapping(id, petId)))
    return failure();
  if (failed(symbolTable_.find(petId, indVar)))
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
                                   SmallVector<Value, 4> &loopIvs) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto indexes = isl::manage(pet_expr_access_get_index(expr));
  if (failed(getIndexes(indexes, loopIvs)))
    return failure();
  return success();
}

static int getStride(isl::aff aff) {
  auto val = aff.get_constant_val();
  assert((val.is_zero() || val.is_one() || val.is_negone()) &&
         "expect 1, -1 or 0");
  if (val.is_one())
    return 1;
  if (val.is_negone())
    return -1;
  return 0;
}

SmallVector<Value, 4>
MLIRCodegen::applyAccessExpression(__isl_keep pet_expr *expr,
                                   SmallVector<Value, 4> &loopIvs) {
  assert(loopIvs.size() && "must be non empty");
  SmallVector<Value, 4> res;
  auto mpwaff = isl::manage(pet_expr_access_get_index(expr));

  auto islmap = isl_map_from_multi_pw_aff(pet_expr_access_get_index(expr));
  isl_map_dump(islmap);

  SmallVector<isl::pw_aff, 2> pwaffs;
  for (size_t i = 0; i < mpwaff.dim(isl::dim::out); i++)
    pwaffs.push_back(mpwaff.get_pw_aff(i));

  // assert((loopIvs.size() == pwaffs.size()) && "expect same size");

  auto ctx = loopIvs[0].getContext();
  auto loc = builder_.getUnknownLoc();
  for (auto pwaff : pwaffs) {
    // isl_pw_aff_dump(pwaff.get());
    // llvm::errs() << pwaff.str() << "\n";
    // for (const auto &iv : loopIvs) {
    // auto pwaff = pwaffs[&iv - &loopIvs[0]];
    assert(pwaff.n_piece() == 1 && "expect single piece");
    isl::aff aff = nullptr;
    auto extractAff = [&](isl::set s, isl::aff a) {
      aff = a;
      return isl_stat_ok;
    };
    pwaff.foreach_piece(extractAff);
    isl_aff_dump(aff.get());
    auto space = isl_aff_get_space(aff.get());
    isl_space_dump(space);
    AffineExpr i;
    bindDims(ctx, i);
    auto stride = getStride(aff);
    // todo fix
    auto iv = loopIvs[0];
    if (stride == 0) {
      res.push_back(iv);
      continue;
    }
    auto affineMap = AffineMap::get(1, 0, i + stride);
    auto newVal = builder_.create<AffineApplyOp>(loc, affineMap, iv);
    res.push_back(newVal);
  }
  return res;
}

Value MLIRCodegen::composeInductionExpression(__isl_keep pet_expr *expr,
                                              Value indVar) {
  // an induction expr does not have an outer array id.
  auto id = isl::manage(pet_expr_access_get_id(expr));
  if (id)
    return indVar;
  auto muaff = isl::manage(pet_expr_access_get_index(expr));
  assert(muaff.dim(isl::dim::out) == 1);
  auto pwaff = muaff.get_pw_aff(0);

  int increment = 0;
  auto extractExpr = [&](isl::set set, isl::aff aff) {
    for (int i = 0; i < aff.dim(isl::dim::out); i++) {
      isl::val incr = aff.get_constant_val();
      increment = std::stoi(incr.to_str());
    }
    return isl_stat_ok;
  };
  pwaff.foreach_piece(extractExpr);
  if (!increment)
    return indVar;
  // FIXME: We assume  i + something atm
  // an i with type int
  auto loc = builder_.getUnknownLoc();
  Value incrVal = createConstantIntOp(increment, loc);
  indVar = createBinaryOp(loc, indVar, incrVal, BinaryOpType::ADD);
  return indVar;
}

Value MLIRCodegen::createLoad(__isl_take pet_expr *expr) {
  // pet_expr_dump(expr);
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
    // emit the load only if we are dealing with a memref.
    if (scalar.getType().dyn_cast<MemRefType>()) {
      Value zeroIndex = builder_.create<ConstantIndexOp>(location, 0);
      scalar = builder_.create<LoadOp>(location, scalar, zeroIndex);
    }
    // if we are dealing with a complex expr (i.e., i + 1) for
    // the induction variable we may need to create
    // extra operations.
    scalar = composeInductionExpression(expr, scalar);
    pet_expr_free(expr);
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

  loopIvs = applyAccessExpression(expr, loopIvs);
  pet_expr_free(expr);
  return builder_.create<LoadOp>(location, symbol, loopIvs);
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
    builder_.create<StoreOp>(location, op, scalar, zeroIndex);
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
  builder_.create<StoreOp>(location, op, symbol, loopIvs);
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
  // pet_expr_dump(expr);
  // get type for lhs.
  auto lhsPetExpr = pet_expr_get_arg(expr, 0);

  // check if we are dealing with an induction variable.
  // If so, do not update it. We will update at the end
  // of the for.
  Value indVar = nullptr;
  if (succeeded(getSymbolInductionVar(lhsPetExpr, indVar))) {
    pet_expr_free(expr);
    pet_expr_free(lhsPetExpr);
    return indVar;
  }

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

  auto rhsExpr = pet_expr_get_arg(expr, 0);
  Value symbolRhs = nullptr;
  if (failed(getSymbol(rhsExpr, symbolRhs)))
    llvm_unreachable("symbol must be available in symbol table");
  pet_expr_free(rhsExpr);

  Value rhsLoad = createLoad(pet_expr_get_arg(expr, 0));
  if (!rhsLoad)
    return nullptr;
  Value rhs = createExpr(pet_expr_get_arg(expr, 1), symbolRhs.getType());
  if (!rhs)
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
    op = createBinaryOp(location, rhsLoad, rhs, BinaryOpType::SUB);
    break;
  }
  case pet_op_div_assign: {
    op = createBinaryOp(location, rhsLoad, rhs, BinaryOpType::DIV);
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

// Kill statement are introduced automatically for locally
// defined variables (scalars and/or vectors). One at the poin
// of the declaration and one when the declaration goes out of
// scope.
// Use --reschedule options to avoid emission of kill stmts.
Value MLIRCodegen::createDefinition(__isl_take pet_expr *expr) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  assert((pet_expr_get_n_arg(expr) == 1) && "expect single arg for kill");
  auto arg = pet_expr_get_arg(expr, 0);
  auto argType = pet_expr_get_type(arg);
  assert((argType == pet_expr_access) &&
         "expect pet_expr_access as arg for kill");
  // avoid to emit a declaration if we hit a kill
  // statement and the array is already in the symbol
  // table.
  Value symbol = nullptr;
  if (succeeded(getSymbol(arg, symbol))) {
    pet_expr_free(arg);
    pet_expr_free(expr);
    return symbol;
  }
  auto idArray = isl::manage(pet_expr_access_get_id(arg));
  /*
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
  */
}

// TODO: check pet_expr_free, there is a better way of doing it?
Value MLIRCodegen::createOp(__isl_take pet_expr *expr, Type t) {
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

  Value lhs = createExpr(pet_expr_get_arg(expr, 0), t);
  if (!lhs)
    return nullptr;
  Value rhs = createExpr(pet_expr_get_arg(expr, 1), t);
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

// the type of the pet expr may discord with the passed value type.
// Consider y[0] = 0 where y is a memref<f64>.
// In this case, type will be double but `0` will be
// a pet_expr_int.
// Thus we first switch on the type of pet_expr `expr` then we adjust
// the type obtained to match `type<
Value MLIRCodegen::createConstantOp(__isl_take pet_expr *expr,
                                    ElementType type) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
  auto loc = builder_.getUnknownLoc();

  int valueAsInt = std::numeric_limits<int>::max();
  std::string valueAsDoubleOrFloat;
  switch (pet_expr_get_type(expr)) {
  case pet_expr_int: {
    valueAsInt = std::stoi(isl::manage(pet_expr_int_get_val(expr)).to_str());
    break;
  }
  case pet_expr_double: {
    valueAsDoubleOrFloat = std::string(pet_expr_double_get_str(expr));
    break;
  }
  default:
    llvm_unreachable("expect only pet_expr_int or double");
  }

  switch (type) {
  case ElementType::INT: {
    int intForConstant = (valueAsInt == std::numeric_limits<int>::max())
                             ? std::stoi(valueAsDoubleOrFloat)
                             : valueAsInt;
    pet_expr_free(expr);
    return createConstantIntOp(intForConstant, loc);
  }
  case ElementType::FLOAT: {
    float floatForConstant = (valueAsInt == std::numeric_limits<int>::max())
                                 ? std::stof(valueAsDoubleOrFloat)
                                 : (float)valueAsInt;
    pet_expr_free(expr);
    return createConstantFloatOp(floatForConstant, loc);
  }
  case ElementType::DOUBLE: {
    // XXX: here pet only exposes get_str method, why?
    double doubleForConstant = (valueAsInt == std::numeric_limits<int>::max())
                                   ? std::stod(valueAsDoubleOrFloat)
                                   : (double)valueAsInt;
    pet_expr_free(expr);
    return createConstantDoubleOp(doubleForConstant, loc);
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

#include "mlir/Dialect/SCF/SCF.h"
Value MLIRCodegen::createCallOp(__isl_take pet_expr *expr, Type t) {
  auto nameFunc = std::string(pet_expr_call_get_name(expr));
  if (nameFunc == "barrier") {
    auto loc = builder_.getUnknownLoc();
    builder_.create<mlir::scf::BarrierOp>(loc, /*return type*/ ArrayRef<Type>{},
                                          ArrayRef<Value>{});
    // even though barrier doesn't have a meaningful return right now
    // mlir-pet assumes that a null return indicates error
    // Thus we return constant 0.
    return createConstantIntOp(0, loc);
  }
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

  // cast the memref to unranked type.
  auto loc = builder_.getUnknownLoc();
  auto newMemRefType =
      UnrankedMemRefType::get(memRef.getElementType(), memRef.getMemorySpace());
  auto castedMemRef = builder_.create<MemRefCastOp>(loc, symbol, newMemRefType);

  // insert the function.
  auto module = castedMemRef.getParentOfType<ModuleOp>();

  pet_expr_free(subExpr);
  pet_expr_free(expr);

  // void function.
  if (!t) {
    auto symbolFn =
        getOrInsertFunction(builder_, module, nameFunc,
                            llvm::ArrayRef<Type>{castedMemRef.getType()});
    builder_.create<CallOp>(loc, symbolFn, /*return type*/ ArrayRef<Type>{},
                            ArrayRef<Value>{castedMemRef});
    return symbol;
  }
  // memref<* x TYPE> -> TYPE (single result).
  if (auto typeArg = t.dyn_cast_or_null<MemRefType>()) {
    auto symbolFn = getOrInsertFunction(
        builder_, module, nameFunc, ArrayRef<Type>{castedMemRef.getType()},
        ArrayRef<Type>{typeArg.getElementType()});
    symbol = builder_
                 .create<CallOp>(loc, symbolFn,
                                 ArrayRef<Type>{typeArg.getElementType()},
                                 ArrayRef<Value>{castedMemRef})
                 .getResult(0);
    return symbol;
  }
  return symbol;
}

static ElementType getElementTypeFromMLIRType(Type t) {
  LLVM_DEBUG(dbgs() << __func__ << "\n");
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
  if (memRefElemType.isInteger(32))
    return ElementType::INT;
  llvm_unreachable("expect type F32/F64/Int32");
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
  case pet_expr_double: {
    auto type = getElementTypeFromMLIRType(t);
    return createConstantOp(expr, type);
  }
  case pet_expr_call:
    return createCallOp(expr, t);
  case pet_expr_cast: {
    llvm_unreachable("type not handled");
    return nullptr;
  }
  case pet_expr_op:
    return createOp(expr, t);
  }

  return nullptr;
}

LogicalResult MLIRCodegen::createStmt(__isl_keep pet_expr *expr) {
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
    return FloatType::getF32(&context);
  case ElementType::DOUBLE:
    return FloatType::getF64(&context);
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
  if (dimensionality == 0)
    return MemRefType::get({1}, tensorType);
  return MemRefType::get(shape, tensorType);
}

SmallVector<Type, 8> MLIRCodegen::getFunctionArgumentsTypes(
    MLIRContext &context, const SmallVector<PetArray, 4> &inputTensors) {
  SmallVector<Type, 8> argTypes;
  if (!inputTensors.size())
    return {};
  for (const auto &inputTensor : inputTensors)
    argTypes.push_back(getTensorType(context, inputTensor));
  return argTypes;
}

LogicalResult MLIRCodegen::declare(std::string id, mlir::Value value) {
  if (succeeded(symbolTable_.find(id)))
    return failure();
  symbolTable_.insert(id, value);
  return success();
}

MLIRCodegen::MLIRCodegen(MLIRContext &context) : builder_(&context) {
  theModule_ = ModuleOp::create(builder_.getUnknownLoc());
  // auto inputTensors = scop_.getInputArrays();
  // auto argTypes = getFunctionArgumentsTypes(context, inputTensors);
  // auto funcType = builder_.getFunctionType(argTypes, llvm::None);

  /*
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
  */
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

Operation *MLIRCodegen::createLoop(int lb, int ub, int step,
                                   std::string iteratorId, bool parallel) {
  if (parallel) {
    auto loc = builder_.getUnknownLoc();
    auto loop = builder_.create<scf::ParallelOp>(
        loc, SmallVector<Value, 1>({builder_.create<ConstantIndexOp>(loc, lb)}),
        SmallVector<Value, 1>({builder_.create<ConstantIndexOp>(loc, ub)}),
        SmallVector<Value, 1>({builder_.create<ConstantIndexOp>(loc, step)}));
    builder_.setInsertionPointToStart(loop.getBody());

    auto resInsertion =
        getLoopTable().insert(iteratorId, loop.getInductionVars()[0]);
    if (failed(resInsertion))
      llvm_unreachable("failed to insert in loop table");
    return loop;
  } else {
    auto loop =
        builder_.create<AffineForOp>(builder_.getUnknownLoc(), lb, ub, step);
    builder_.setInsertionPointToStart(loop.getBody());

    auto resInsertion =
        getLoopTable().insert(iteratorId, loop.getInductionVar());
    if (failed(resInsertion))
      llvm_unreachable("failed to insert in loop table");
    return loop;
  }
}

Operation *MLIRCodegen::createLoop(int lb, AffineExpr ubExpr, std::string ubId,
                                   int step, bool leqBound,
                                   std::string iteratorId, bool parallel) {
  assert(!parallel);
  Value ub;
  if (failed(this->getLoopTable().find(ubId, ub)))
    llvm_unreachable("Couldn't find the bound in the loop table.");

  auto lbMap = AffineMap::getConstantMap(lb, builder_.getContext());
  mlir::AffineMap ubMap;

  if (leqBound) {
    AffineExpr affExpr = getAffineConstantExpr(1, builder_.getContext());
    ubMap = AffineMap::get(1, 0, affExpr + ubExpr);
  } else {
    ubMap = AffineMap::get(1, 0, ubExpr);
  }

  ValueRange ubOperands = ValueRange(ub);
  ValueRange lbOperands = {};

  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
  loop.getBody()->clear();

  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineYieldOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  auto resInsertion = getLoopTable().insert(iteratorId, loop.getInductionVar());
  if (failed(resInsertion))
    llvm_unreachable("failed to insert in loop table");

  return loop;
}

Operation *MLIRCodegen::createLoop(AffineExpr lbExpr, std::string lbId, int ub,
                                   int step, std::string iteratorId,
                                   bool parallel) {
  assert(!parallel);
  Value lb;
  if (failed(this->getLoopTable().find(lbId, lb))) {
    llvm_unreachable("Couldn't find the bound in the loop table.");
  }

  auto ubMap = AffineMap::getConstantMap(ub, builder_.getContext());
  auto lbMap = AffineMap::get(1, 0, lbExpr);

  ValueRange ubOperands = {};
  ValueRange lbOperands = ValueRange(lb);

  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
  loop.getBody()->clear();

  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineYieldOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  auto resInsertion = getLoopTable().insert(iteratorId, loop.getInductionVar());
  if (failed(resInsertion))
    llvm_unreachable("failed to insert in loop table");
  return loop;
}

Operation *MLIRCodegen::createLoop(AffineExpr lbExpr, std::string lbId,
                                   AffineExpr ubExpr, std::string ubId,
                                   int step, std::string iteratorId,
                                   bool parallel) {
  assert(!parallel);
  Value ub, lb;
  if (failed(this->getLoopTable().find(ubId, ub)))
    llvm_unreachable("Couldn't find the bound in the loop table.");
  if (failed(this->getLoopTable().find(lbId, lb)))
    llvm_unreachable("Couldn't find the bound in the loop table.");

  auto lbMap = AffineMap::get(1, 0, lbExpr);
  auto ubMap = AffineMap::get(1, 0, ubExpr);

  ValueRange ubOperands = ValueRange(ub);
  ValueRange lbOperands = ValueRange(lb);
  auto loop = builder_.create<AffineForOp>(builder_.getUnknownLoc(), lbOperands,
                                           lbMap, ubOperands, ubMap, step);
  loop.getBody()->clear();

  builder_.setInsertionPointToStart(loop.getBody());
  builder_.create<AffineYieldOp>(builder_.getUnknownLoc());
  builder_.setInsertionPointToStart(loop.getBody());

  auto resInsertion = getLoopTable().insert(iteratorId, loop.getInductionVar());
  if (failed(resInsertion))
    llvm_unreachable("failed to insert in loop table");
  return loop;
}

void MLIRCodegen::createReturn() {
  builder_.create<ReturnOp>(builder_.getUnknownLoc());
}

void MLIRCodegen::setInsertionPointAfter(Operation *op) {
  builder_.setInsertionPointAfter(op);
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

LogicalResult LoopTable::getValueAtPos(size_t pos, Value &value) const {
  if (pos > size())
    return failure();
  auto it = begin();
  std::advance(it, pos);
  value = it->second;
  return success();
}

LogicalResult LoopTable::getIdAtPos(size_t pos, std::string &valueId) const {
  if (pos > size())
    return failure();
  auto it = begin();
  std::advance(it, pos);
  valueId = it->first;
  return success();
}

LogicalResult LoopTable::lookUpPetMapping(std::string id) const {
  auto it = mappedIndVars_.find(id);
  if (it != mappedIndVars_.end())
    return success();
  return failure();
}

LogicalResult LoopTable::lookUpIslMapping(std::string id,
                                          std::string &res) const {
  for (auto it = mappedIndVars_.begin(); it != mappedIndVars_.end(); it++)
    if (it->second == id) {
      res = it->first;
      return success();
    }
  return failure();
}

void LoopTable::insertMapping(std::string idPet, std::string idIsl) {
  mappedIndVars_.insert(std::make_pair(idPet, idIsl));
}

size_t codegen::SymbolTable::size() const { return symbolTable_.size(); }

void codegen::SymbolTable::dumpImpl() const {
  outs() << "   size: " << symbolTable_.size() << "\n";
  for (auto it = symbolTable_.begin(); it != symbolTable_.end(); it++)
    outs() << "   id: " << it->first << "\n";
}

void codegen::SymbolTable::dump() const {
  outs() << "Symbol table: \n";
  dumpImpl();
}

void codegen::LoopTable::dump() const {
  outs() << "Loop Table: \n";
  dumpImpl();
  for (auto it = mappedIndVars_.begin(); it != mappedIndVars_.end(); it++)
    outs() << "   Loop id " << it->first << " --> " << it->second << "\n";
}
