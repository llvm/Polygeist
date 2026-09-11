//===- FoldSCFIf.cpp - Fold scf.if into select -----------------*- C++ -*-===//

#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::polygeist;

#define DEBUG_TYPE "fold-scf-if"

static bool hasSingleStore(Block *block) {
  llvm::SetVector<Value> memrefs;

  for (Operation &op : block->getOperations()) {
    if (!isa<affine::AffineStoreOp, memref::StoreOp>(op))
      continue;

    Value memref = op.getOperand(1);
    if (memrefs.count(memref))
      return false;

    // Store indices must be defined above the current block so that a lifted
    // store can be emitted after the if.
    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      if (llvm::any_of(storeOp.getMapOperands(), [&](Value operand) {
            return operand.getParentBlock() == block;
          }))
        return false;
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (llvm::any_of(storeOp.getIndices(), [&](Value operand) {
            return operand.getParentBlock() == block;
          }))
        return false;
    }

    memrefs.insert(memref);
  }

  return true;
}

static bool canLiftStores(Block *block) {
  bool seenStore = false;
  for (Operation &op : block->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (isa<affine::AffineStoreOp, memref::StoreOp>(op)) {
      seenStore = true;
      continue;
    }
    if (seenStore && !isMemoryEffectFree(&op))
      return false;
  }
  return true;
}

namespace {
struct MemRefStoreInfo {
  unsigned index = 0;
  Type type;
  Operation *source = nullptr;
  SmallVector<Value> operands;
  AffineMap affineMap;
  bool isAffineStore = false;
};
} // namespace

static bool getMemRefLoadInfo(Value value, MemRefStoreInfo &info) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;

  info = MemRefStoreInfo();
  info.type = value.getType();
  info.source = op;

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    info.operands.assign(loadOp.getIndices().begin(),
                         loadOp.getIndices().end());
    info.isAffineStore = false;
    return true;
  }

  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    info.operands.assign(loadOp.getMapOperands().begin(),
                         loadOp.getMapOperands().end());
    info.affineMap = loadOp.getAffineMap();
    info.isAffineStore = true;
    return true;
  }

  return false;
}

static bool getSingleStoreInfo(Operation &op, MemRefStoreInfo &info) {
  info = MemRefStoreInfo();
  info.source = &op;

  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    info.type = storeOp.getValueToStore().getType();
    info.operands.assign(storeOp.getIndices().begin(),
                         storeOp.getIndices().end());
    info.isAffineStore = false;
    return true;
  }

  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    info.type = storeOp.getValueToStore().getType();
    info.operands.assign(storeOp.getMapOperands().begin(),
                         storeOp.getMapOperands().end());
    info.affineMap = storeOp.getAffineMap();
    info.isAffineStore = true;
    return true;
  }

  return false;
}

static void getMemRefStoreInfo(Block *block,
                               llvm::MapVector<Value, MemRefStoreInfo> &info) {
  unsigned ord = 0;
  for (Operation &op : block->getOperations()) {
    if (!isa<memref::StoreOp, affine::AffineStoreOp>(op))
      continue;

    MemRefStoreInfo storeInfo;
    storeInfo.index = ord++;
    storeInfo.type = op.getOperand(0).getType();
    storeInfo.source = &op;

    if (auto storeOp = dyn_cast<memref::StoreOp>(op))
      storeInfo.operands = storeOp.getIndices();
    else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      storeInfo.operands = storeOp.getMapOperands();
      storeInfo.affineMap = storeOp.getAffineMap();
      storeInfo.isAffineStore = true;
    }

    info[op.getOperand(1)] = storeInfo;
  }
}

static bool sameStoreAddress(const MemRefStoreInfo &a,
                             const MemRefStoreInfo &b) {
  if (a.isAffineStore != b.isAffineStore)
    return false;
  if (a.operands != b.operands)
    return false;
  if (a.isAffineStore && a.affineMap != b.affineMap)
    return false;
  return true;
}

static bool hasMatchingStores(ArrayRef<Block *> blocks) {
  if (blocks.empty())
    return true;

  llvm::MapVector<Value, MemRefStoreInfo> expected;
  getMemRefStoreInfo(blocks.front(), expected);

  for (Block *block : blocks.drop_front()) {
    llvm::MapVector<Value, MemRefStoreInfo> actual;
    getMemRefStoreInfo(block, actual);

    if (expected.size() != actual.size())
      return false;

    for (auto &entry : expected) {
      auto actualIt = actual.find(entry.first);
      if (actualIt == actual.end())
        return false;
      if (!sameStoreAddress(entry.second, actualIt->second))
        return false;
    }
  }

  return true;
}

static Value getMemrefFromStore(Operation *op) {
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getMemref();
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    return storeOp.getMemref();
  return Value();
}

static Value getMemrefFromLoad(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getMemref();
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
    return loadOp.getMemref();
  return Value();
}

static bool sameLoadStoreAddress(const MemRefStoreInfo &load,
                                 const MemRefStoreInfo &store) {
  if (load.isAffineStore != store.isAffineStore)
    return false;
  if (getMemrefFromLoad(load.source) != getMemrefFromStore(store.source))
    return false;
  if (load.operands != store.operands)
    return false;
  if (load.isAffineStore && load.affineMap != store.affineMap)
    return false;
  return true;
}

static bool sameLoadAddress(const MemRefStoreInfo &a,
                            const MemRefStoreInfo &b) {
  if (a.isAffineStore != b.isAffineStore)
    return false;
  if (getMemrefFromLoad(a.source) != getMemrefFromLoad(b.source))
    return false;
  if (a.operands != b.operands)
    return false;
  if (a.isAffineStore && a.affineMap != b.affineMap)
    return false;
  return true;
}

static Value getStoredValue(Operation *op) {
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getValueToStore();
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    return storeOp.getValueToStore();
  return Value();
}

static bool isLoadLike(Operation &op) {
  return isa<memref::LoadOp, affine::AffineLoadOp>(op);
}

static bool canSpeculateForSelect(Block *block) {
  for (Operation &op : block->getOperations()) {
    if (isa<scf::YieldOp, affine::AffineYieldOp>(op))
      continue;
    if (isLoadLike(op))
      continue;
    if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
      return false;
  }
  return true;
}

static Value materializeIntegerSetCondition(Location loc, IntegerSet set,
                                            ValueRange operands, OpBuilder &b) {
  Value active;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);

  for (auto constraint : llvm::enumerate(set.getConstraints())) {
    AffineMap constraintMap =
        AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                       constraint.value(), b.getContext());
    Value applied =
        b.create<affine::AffineApplyOp>(loc, constraintMap, operands);
    auto predicate = set.isEq(constraint.index())
                         ? arith::CmpIPredicate::eq
                         : arith::CmpIPredicate::sge;
    Value ok = b.create<arith::CmpIOp>(loc, predicate, applied, zero);
    active = active ? b.create<arith::AndIOp>(loc, active, ok).getResult()
                    : ok;
  }

  if (!active)
    active = b.create<arith::ConstantIntOp>(loc, true, 1);
  return active;
}

static bool hasUnsafeInterveningEffect(Operation *begin, Operation *end) {
  for (Operation *op = begin->getNextNode(); op && op != end;
       op = op->getNextNode()) {
    if (isLoadLike(*op) || isMemoryEffectFree(op))
      continue;
    return true;
  }
  return false;
}

static bool valueMatchesCandidate(Value value, Value candidate) {
  if (value == candidate)
    return true;

  MemRefStoreInfo valueLoad, candidateLoad;
  if (!getMemRefLoadInfo(value, valueLoad) ||
      !getMemRefLoadInfo(candidate, candidateLoad))
    return false;
  return sameLoadAddress(valueLoad, candidateLoad);
}

static bool getCompareOperands(Value condition, Value &lhs, Value &rhs) {
  Operation *condOp = condition.getDefiningOp();
  if (!condOp || !isa<arith::CmpFOp, arith::CmpIOp>(condOp) ||
      condOp->getNumOperands() != 2)
    return false;
  lhs = condOp->getOperand(0);
  rhs = condOp->getOperand(1);
  return true;
}

struct LinearIndexForm {
  llvm::DenseMap<Value, int64_t> coefficients;
  int64_t constant = 0;

  bool operator==(const LinearIndexForm &other) const {
    return constant == other.constant && coefficients == other.coefficients;
  }
};

static bool addLinearValue(Value value, int64_t scale, LinearIndexForm &form) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return addLinearValue(cast.getIn(), scale, form);
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return addLinearValue(add.getLhs(), scale, form) &&
           addLinearValue(add.getRhs(), scale, form);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return addLinearValue(sub.getLhs(), scale, form) &&
           addLinearValue(sub.getRhs(), -scale, form);
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    auto integer = dyn_cast<IntegerAttr>(constant);
    if (!integer)
      return false;
    form.constant += scale * integer.getInt();
    return true;
  }
  form.coefficients[value] += scale;
  if (form.coefficients[value] == 0)
    form.coefficients.erase(value);
  return true;
}

static bool addLinearExpr(AffineExpr expression, ValueRange dims,
                          ValueRange symbols, int64_t scale,
                          LinearIndexForm &form) {
  if (auto constant = expression.dyn_cast<AffineConstantExpr>()) {
    form.constant += scale * constant.getValue();
    return true;
  }
  if (auto dim = expression.dyn_cast<AffineDimExpr>())
    return addLinearValue(dims[dim.getPosition()], scale, form);
  if (auto symbol = expression.dyn_cast<AffineSymbolExpr>())
    return addLinearValue(symbols[symbol.getPosition()], scale, form);
  auto binary = expression.dyn_cast<AffineBinaryOpExpr>();
  if (!binary)
    return false;
  if (binary.getKind() == AffineExprKind::Add)
    return addLinearExpr(binary.getLHS(), dims, symbols, scale, form) &&
           addLinearExpr(binary.getRHS(), dims, symbols, scale, form);
  if (binary.getKind() != AffineExprKind::Mul)
    return false;
  auto lhsConstant = binary.getLHS().dyn_cast<AffineConstantExpr>();
  auto rhsConstant = binary.getRHS().dyn_cast<AffineConstantExpr>();
  if (lhsConstant)
    return addLinearExpr(binary.getRHS(), dims, symbols,
                         scale * lhsConstant.getValue(), form);
  if (rhsConstant)
    return addLinearExpr(binary.getLHS(), dims, symbols,
                         scale * rhsConstant.getValue(), form);
  return false;
}

static bool sameLinearBound(Value value, AffineMap map, ValueRange operands) {
  if (map.getNumResults() != 1)
    return false;
  LinearIndexForm valueForm, mapForm;
  bool valueOK;
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>())
    valueOK = addLinearExpr(apply.getAffineMap().getResult(0), {},
                            apply.getMapOperands(), 1, valueForm);
  else
    valueOK = addLinearValue(value, 1, valueForm);
  if (!valueOK ||
      !addLinearExpr(map.getResult(0),
                     operands.take_front(map.getNumDims()),
                     operands.drop_front(map.getNumDims()), 1, mapForm))
    return false;
  return valueForm == mapForm;
}

// C frontends commonly preserve a source-level non-empty-loop check around a
// dynamic loop even though scf.for already has zero-trip semantics:
//
//   %nonempty = arith.cmpi sgt, %ub, %lb
//   scf.if %nonempty {
//     scf.for %i = %lb to %ub step %step { ... }
//   }
//
// Removing this exact wrapper is not speculation: both forms execute the loop
// iff lb < ub.  Besides simplifying the CFG, this lets raise-scf-to-affine see
// dynamic bounds that were otherwise hidden behind an scf.if region.  Keep the
// pattern deliberately narrow (no results, no else, and exactly one loop) so
// setup code or other conditionally-executed effects are never moved.
static bool foldRedundantNonEmptyLoopGuard(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 0 || ifOp.elseBlock())
    return false;

  Block *body = ifOp.thenBlock();
  if (!body)
    return false;

  SmallVector<Operation *> operations;
  for (Operation &op : body->without_terminator())
    operations.push_back(&op);
  if (operations.empty())
    return false;

  Operation &only = *operations.back();
  Value lowerBound, upperBound;
  if (auto loop = dyn_cast<scf::ForOp>(only)) {
    lowerBound = loop.getLowerBound();
    upperBound = loop.getUpperBound();
  } else if (!isa<affine::AffineForOp>(only)) {
    return false;
  }

  auto compare = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare)
    return false;

  bool sameLower = false, sameUpper = false;
  if (auto loop = dyn_cast<scf::ForOp>(only)) {
    sameLower = compare.getRhs() == loop.getLowerBound();
    sameUpper = compare.getLhs() == loop.getUpperBound();
  } else {
    auto affineLoop = cast<affine::AffineForOp>(only);
    sameLower = sameLinearBound(compare.getRhs(), affineLoop.getLowerBoundMap(),
                                affineLoop.getLowerBoundOperands());
    sameUpper = sameLinearBound(compare.getLhs(), affineLoop.getUpperBoundMap(),
                                affineLoop.getUpperBoundOperands());
  }
  bool equivalent =
      (compare.getPredicate() == arith::CmpIPredicate::sgt && sameUpper &&
       sameLower);
  if (!equivalent && compare.getPredicate() == arith::CmpIPredicate::slt) {
    if (auto loop = dyn_cast<scf::ForOp>(only))
      equivalent = compare.getLhs() == loop.getLowerBound() &&
                   compare.getRhs() == loop.getUpperBound();
    else {
      auto affineLoop = cast<affine::AffineForOp>(only);
      equivalent =
          sameLinearBound(compare.getLhs(), affineLoop.getLowerBoundMap(),
                          affineLoop.getLowerBoundOperands()) &&
          sameLinearBound(compare.getRhs(), affineLoop.getUpperBoundMap(),
                          affineLoop.getUpperBoundOperands());
    }
  }
  if (!equivalent)
    return false;

  // cgeist may compute a later dimension's bound immediately before the
  // guarded loop. Loads and effect-free scalar address arithmetic are safe to
  // speculate, and moving them makes the loop the only guarded operation.
  for (Operation *op : llvm::drop_end(operations)) {
    if ((!isLoadLike(*op) && !isMemoryEffectFree(op)) ||
        op->getNumRegions() != 0)
      return false;
  }
  for (Operation *op : llvm::drop_end(operations))
    op->moveBefore(ifOp);
  only.moveBefore(ifOp);
  ifOp.erase();
  return true;
}

// Hoist a branch-local prefix that is safe to speculate.  cgeist often emits
// dynamic-bound loads and arithmetic inside a non-empty guard, before the
// first loop.  Keeping those values in the scf.if region prevents them from
// being legal affine symbols even though they do not depend on the branch.
static bool hoistSpeculatableGuardPrefix(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 0 || ifOp.elseBlock())
    return false;

  SmallVector<Operation *> prefix;
  for (Operation &op : ifOp.thenBlock()->without_terminator()) {
    if (op.getNumRegions() != 0)
      break;
    if (!isLoadLike(op) && !isMemoryEffectFree(&op))
      break;
    prefix.push_back(&op);
  }
  if (prefix.empty())
    return false;

  for (Operation *op : prefix)
    op->moveBefore(ifOp);
  return true;
}

static LogicalResult foldGuardedStoreUpdate(scf::IfOp ifOp, OpBuilder &b) {
  if (ifOp.elseBlock() || ifOp.getNumResults() != 0)
    return failure();

  Operation *store = nullptr;
  for (Operation &op : ifOp.thenBlock()->without_terminator()) {
    if (isa<memref::StoreOp, affine::AffineStoreOp>(op)) {
      if (store)
        return failure();
      store = &op;
      continue;
    }
    if (!isLoadLike(op))
      return failure();
  }
  if (!store)
    return failure();

  MemRefStoreInfo storeInfo;
  if (!getSingleStoreInfo(*store, storeInfo))
    return failure();

  // This path does not clone the branch.  A destination materialized inside
  // it (for example, memref.get_global) would become a dangling value after
  // the if is erased.
  if (getMemrefFromStore(store).getParentBlock() == ifOp.thenBlock())
    return failure();

  for (Value operand : storeInfo.operands)
    if (operand.getParentBlock() == ifOp.thenBlock())
      return failure();

  Value cmpLhs, cmpRhs;
  if (!getCompareOperands(ifOp.getCondition(), cmpLhs, cmpRhs))
    return failure();

  Value stored = getStoredValue(store);
  Value candidate;
  Value oldValue;
  if (valueMatchesCandidate(stored, cmpLhs)) {
    candidate = cmpLhs;
    oldValue = cmpRhs;
  } else if (valueMatchesCandidate(stored, cmpRhs)) {
    candidate = cmpRhs;
    oldValue = cmpLhs;
  } else {
    return failure();
  }

  MemRefStoreInfo oldLoad;
  if (!getMemRefLoadInfo(oldValue, oldLoad) ||
      !sameLoadStoreAddress(oldLoad, storeInfo))
    return failure();

  if (oldLoad.source->getBlock() != ifOp->getBlock() ||
      hasUnsafeInterveningEffect(oldLoad.source, ifOp))
    return failure();

  OpBuilder::InsertionGuard guard(b);
  Location loc = ifOp.getLoc();
  b.setInsertionPointAfter(ifOp);
  Value selected =
      b.create<arith::SelectOp>(loc, ifOp.getCondition(), candidate, oldValue);

  if (auto storeOp = dyn_cast<memref::StoreOp>(store)) {
    b.create<memref::StoreOp>(loc, selected, storeOp.getMemref(),
                              storeOp.getIndices());
  } else {
    auto affineStoreOp = cast<affine::AffineStoreOp>(store);
    b.create<affine::AffineStoreOp>(loc, selected, affineStoreOp.getMemref(),
                                    affineStoreOp.getAffineMap(),
                                    affineStoreOp.getMapOperands());
  }

  ifOp.erase();
  return success();
}

static bool foldSingleStoreIfToSelect(scf::IfOp ifOp, OpBuilder &b) {
  if (ifOp.elseBlock() || ifOp.getNumResults() != 0)
    return false;

  Operation *store = nullptr;
  for (Operation &op : ifOp.thenBlock()->without_terminator()) {
    if (isa<memref::StoreOp, affine::AffineStoreOp>(op)) {
      if (store)
        return false;
      store = &op;
      continue;
    }
    if (!isLoadLike(op) &&
        (op.getNumRegions() != 0 || !isMemoryEffectFree(&op)))
      return false;
  }
  if (!store)
    return false;

  MemRefStoreInfo storeInfo;
  if (!getSingleStoreInfo(*store, storeInfo))
    return false;

  for (Value operand : storeInfo.operands)
    if (operand.getParentBlock() == ifOp.thenBlock())
      return false;

  OpBuilder::InsertionGuard guard(b);
  Location loc = ifOp.getLoc();
  b.setInsertionPointAfter(ifOp);

  IRMapping vmap;
  Value candidate;
  for (Operation &op : ifOp.thenBlock()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (&op == store) {
      candidate = vmap.lookupOrDefault(getStoredValue(store));
      continue;
    }
    b.clone(op, vmap);
  }
  if (!candidate)
    return false;

  Value oldValue;
  if (auto storeOp = dyn_cast<memref::StoreOp>(store)) {
    Value memref = vmap.lookupOrDefault(storeOp.getMemref());
    SmallVector<Value> indices;
    for (Value index : storeOp.getIndices())
      indices.push_back(vmap.lookupOrDefault(index));
    oldValue = b.create<memref::LoadOp>(loc, memref, indices);
    Value selected =
        b.create<arith::SelectOp>(loc, ifOp.getCondition(), candidate, oldValue);
    b.create<memref::StoreOp>(loc, selected, memref, indices);
  } else {
    auto affineStoreOp = cast<affine::AffineStoreOp>(store);
    Value memref = vmap.lookupOrDefault(affineStoreOp.getMemref());
    SmallVector<Value> mapOperands;
    for (Value operand : affineStoreOp.getMapOperands())
      mapOperands.push_back(vmap.lookupOrDefault(operand));
    oldValue = b.create<affine::AffineLoadOp>(
        loc, memref, affineStoreOp.getAffineMap(), mapOperands);
    Value selected =
        b.create<arith::SelectOp>(loc, ifOp.getCondition(), candidate, oldValue);
    b.create<affine::AffineStoreOp>(loc, selected, memref,
                                    affineStoreOp.getAffineMap(), mapOperands);
  }

  ifOp.erase();
  return true;
}

static LogicalResult liftStoreOps(scf::IfOp ifOp, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  if (!hasMatchingStores({ifOp.thenBlock(), ifOp.elseBlock()}))
    return failure();

  llvm::MapVector<Value, MemRefStoreInfo> storeInfo;
  getMemRefStoreInfo(ifOp.thenBlock(), storeInfo);

  if (storeInfo.empty())
    return failure();

  SmallVector<Type> storeTypes(storeInfo.size());
  for (auto &info : storeInfo)
    storeTypes[info.second.index] = info.second.type;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Type> resultTypes(ifOp.getResultTypes());
  resultTypes.append(storeTypes);

  scf::IfOp newIfOp = b.create<scf::IfOp>(loc, resultTypes, ifOp.getCondition(),
                                          /*withElseRegion=*/true);

  auto cloneBlock = [&](Block *target, Block *source) {
    IRMapping vmap;

    scf::YieldOp yieldOp = cast<scf::YieldOp>(source->getTerminator());
    unsigned numExistingResults = yieldOp.getNumOperands();
    SmallVector<Value> results(numExistingResults + storeInfo.size());

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(target);

    for (Operation &op : source->getOperations()) {
      if (isa<memref::StoreOp, affine::AffineStoreOp>(op)) {
        Value memref = op.getOperand(1);
        Value toStore = op.getOperand(0);
        results[storeInfo[memref].index + numExistingResults] =
            vmap.lookupOrDefault(toStore);
      } else if (!isa<scf::YieldOp>(op)) {
        b.clone(op, vmap);
      }
    }

    for (auto operand : llvm::enumerate(yieldOp.getOperands()))
      results[operand.index()] = vmap.lookupOrDefault(operand.value());

    b.create<scf::YieldOp>(loc, results);
  };

  cloneBlock(newIfOp.thenBlock(), ifOp.thenBlock());
  cloneBlock(newIfOp.elseBlock(), ifOp.elseBlock());

  b.setInsertionPointAfter(newIfOp);

  for (auto &p : storeInfo) {
    Value memref;
    MemRefStoreInfo info;
    std::tie(memref, info) = p;

    Value result = newIfOp.getResult(ifOp.getNumResults() + info.index);
    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(info.source)) {
      b.create<affine::AffineStoreOp>(loc, result, memref,
                                      storeOp.getAffineMap(), info.operands);
    } else if (isa<memref::StoreOp>(info.source)) {
      b.create<memref::StoreOp>(loc, result, memref, info.operands);
    }
  }

  for (auto result : llvm::enumerate(ifOp.getResults()))
    result.value().replaceAllUsesWith(newIfOp.getResult(result.index()));

  ifOp.erase();
  return success();
}

static bool processLiftStoreOps(func::FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](scf::IfOp ifOp) {
    if (changed)
      return;

    if (!ifOp.elseBlock() || !hasSingleStore(ifOp.thenBlock()) ||
        !hasSingleStore(ifOp.elseBlock()) ||
        !canLiftStores(ifOp.thenBlock()) || !canLiftStores(ifOp.elseBlock()))
      return;

    if (failed(liftStoreOps(ifOp, b)))
      return;

    changed = true;
  });

  return changed;
}

static bool foldScalarSCFIf(scf::IfOp ifOp, OpBuilder &b) {
  if (ifOp.getNumResults() == 0 || !ifOp.elseBlock())
    return false;
  if (!canSpeculateForSelect(ifOp.thenBlock()) ||
      !canSpeculateForSelect(ifOp.elseBlock()))
    return false;

  Location loc = ifOp.getLoc();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Value> thenResults, elseResults;

  auto cloneAfter = [&](Block *block, SmallVectorImpl<Value> &results) {
    IRMapping vmap;
    for (Operation &op : block->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        for (Value result : yieldOp.getOperands())
          results.push_back(vmap.lookupOrDefault(result));
      } else {
        b.clone(op, vmap);
      }
    }
  };

  cloneAfter(ifOp.thenBlock(), thenResults);
  cloneAfter(ifOp.elseBlock(), elseResults);

  if (thenResults.size() != ifOp.getNumResults() ||
      elseResults.size() != ifOp.getNumResults())
    return false;

  for (auto ifResult : llvm::enumerate(ifOp.getResults())) {
    Value newResult = b.create<arith::SelectOp>(
        loc, ifOp.getCondition(), thenResults[ifResult.index()],
        elseResults[ifResult.index()]);
    ifResult.value().replaceAllUsesWith(newResult);
  }

  ifOp.erase();
  return true;
}

static bool foldScalarAffineIf(affine::AffineIfOp ifOp, OpBuilder &b) {
  if (ifOp.getNumResults() == 0 || !ifOp.hasElse())
    return false;
  if (!canSpeculateForSelect(ifOp.getThenBlock()) ||
      !canSpeculateForSelect(ifOp.getElseBlock()))
    return false;

  Location loc = ifOp.getLoc();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(ifOp);

  Value condition = materializeIntegerSetCondition(
      loc, ifOp.getIntegerSet(), ifOp.getOperands(), b);

  SmallVector<Value> thenResults, elseResults;

  auto cloneAfter = [&](Block *block, SmallVectorImpl<Value> &results) {
    IRMapping vmap;
    for (Operation &op : block->getOperations()) {
      if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(op)) {
        for (Value result : yieldOp.getOperands())
          results.push_back(vmap.lookupOrDefault(result));
      } else {
        b.clone(op, vmap);
      }
    }
  };

  cloneAfter(ifOp.getThenBlock(), thenResults);
  cloneAfter(ifOp.getElseBlock(), elseResults);

  if (thenResults.size() != ifOp.getNumResults() ||
      elseResults.size() != ifOp.getNumResults())
    return false;

  for (auto ifResult : llvm::enumerate(ifOp.getResults())) {
    Value newResult = b.create<arith::SelectOp>(
        loc, condition, thenResults[ifResult.index()],
        elseResults[ifResult.index()]);
    ifResult.value().replaceAllUsesWith(newResult);
  }

  ifOp.erase();
  return true;
}

static bool foldSCFIf(scf::IfOp ifOp, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  LLVM_DEBUG(llvm::dbgs() << "Working on scf.if:\n" << ifOp << "\n");

  if (foldRedundantNonEmptyLoopGuard(ifOp))
    return true;

  if (hoistSpeculatableGuardPrefix(ifOp))
    return true;

  // Fold scalar store-update idioms such as softmax/reduce-max:
  //   if (%candidate > %old) store %candidate, %slot
  // into:
  //   %selected = arith.select %cond, %candidate, %old
  //   store %selected, %slot
  // This is intentionally narrower than generic store speculation: the
  // implicit else must be the previously loaded value from the same address.
  if (succeeded(foldGuardedStoreUpdate(ifOp, b)))
    return true;

  if (foldSingleStoreIfToSelect(ifOp, b))
    return true;

  if (foldScalarSCFIf(ifOp, b))
    return true;

  if (!hasSingleStore(ifOp.thenBlock()) ||
      (ifOp.elseBlock() && !hasSingleStore(ifOp.elseBlock())))
    return false;

  // Replacing control flow with select speculates both sides. Keep this path
  // narrow by refusing stores, calls, and nested regions.
  if (!canSpeculateForSelect(ifOp.thenBlock()) ||
      (ifOp.elseBlock() && !canSpeculateForSelect(ifOp.elseBlock())))
    return false;

  if (ifOp.getNumResults() == 0)
    return false;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(ifOp);

  SmallVector<Value> thenResults, elseResults;

  auto cloneAfter = [&](Block *block, SmallVectorImpl<Value> &results) {
    IRMapping vmap;
    for (Operation &op : block->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        for (Value result : yieldOp.getOperands())
          results.push_back(vmap.lookupOrDefault(result));
      } else {
        b.clone(op, vmap);
      }
    }
  };

  cloneAfter(ifOp.thenBlock(), thenResults);

  if (ifOp.elseBlock()) {
    cloneAfter(ifOp.elseBlock(), elseResults);

    for (auto ifResult : llvm::enumerate(ifOp.getResults())) {
      Value newResult = b.create<arith::SelectOp>(
          loc, ifOp.getCondition(), thenResults[ifResult.index()],
          elseResults[ifResult.index()]);
      ifResult.value().replaceAllUsesWith(newResult);
    }
  }

  ifOp.erase();
  return true;
}

static bool processFold(func::FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](scf::IfOp ifOp) {
    if (changed)
      return;

    changed = foldSCFIf(ifOp, b);
  });

  return changed;
}

static bool processAffineFold(func::FuncOp f, OpBuilder &b) {
  bool changed = false;

  f.walk([&](affine::AffineIfOp ifOp) {
    if (changed)
      return;

    changed = foldScalarAffineIf(ifOp, b);
  });

  return changed;
}

namespace {
struct FoldSCFIf : public FoldSCFIfBase<FoldSCFIf> {
  void runOnOperation() override {
    Operation *op = getOperation();
    SmallVector<func::FuncOp> funcs;

    if (auto func = dyn_cast<func::FuncOp>(op))
      funcs.push_back(func);
    else
      op->walk([&](func::FuncOp func) { funcs.push_back(func); });

    for (func::FuncOp func : funcs) {
      if (func->hasAttr("scop.ignored"))
        continue;

      OpBuilder builder(func.getContext());

      while (processLiftStoreOps(func, builder))
        ;

      OpPassManager pm(func.getOperationName());
      pm.addPass(affine::createAffineScalarReplacementPass());
      if (failed(runPipeline(pm, func)))
        return signalPassFailure();

      while (processFold(func, builder))
        ;
      while (processAffineFold(func, builder))
        ;
    }
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createFoldSCFIfPass() {
  return std::make_unique<FoldSCFIf>();
}
} // namespace polygeist
} // namespace mlir
