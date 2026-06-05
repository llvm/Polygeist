#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-linalg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;
using namespace linalg;

// Also want to add support for affine.for ( ) { linalg.generic } -> bigger
// linalg.generic Also probably want to try to do { linalg.generc1();
// linalg.generic2(); } -> bigger linalg.generic()

/*

affine.for() {
    affine.for() {
    }
    affine.for() {
    }
}

*/
struct Condition {
  bool ifTrue;
  AffineIfOp op;
  Condition(bool ifTrue, AffineIfOp op) : ifTrue(ifTrue), op(op) {}
};

bool isLinearInIndex(AffineExpr expr, size_t idx) {
  if (!expr.isFunctionOfDim(idx)) {
    return true;
  }

  if (expr.getKind() == AffineExprKind::DimId) {
    return true;
  }

  if (expr.getKind() == AffineExprKind::Add) {
    auto binop = expr.cast<AffineBinaryOpExpr>();
    return isLinearInIndex(binop.getLHS(), idx) &&
           isLinearInIndex(binop.getRHS(), idx);
  }
  if (expr.getKind() == AffineExprKind::Mul) {
    auto binop = expr.cast<AffineBinaryOpExpr>();
    return (isLinearInIndex(binop.getLHS(), idx) &&
            !binop.getRHS().isFunctionOfDim(idx)) ||
           (isLinearInIndex(binop.getRHS(), idx) &&
            !binop.getLHS().isFunctionOfDim(idx));
  }

  return false;
}

bool isLinearInIndex(AffineMap map, size_t idx) {
  for (auto expr : map.getResults()) {
    if (!isLinearInIndex(expr, idx))
      return false;
  }
  return true;
}

AffineExpr shiftDimsDown1(AffineExpr expr, unsigned numDims, unsigned offset) {
  SmallVector<AffineExpr, 4> dims;
  for (unsigned idx = 0; idx < offset; ++idx)
    dims.push_back(getAffineDimExpr(idx, expr.getContext()));
  for (unsigned idx = offset; idx < numDims; ++idx)
    dims.push_back(getAffineDimExpr(idx - 1, expr.getContext()));
  return expr.replaceDimsAndSymbols(dims, {});
}

// This is reducing the number of input dims in expression by 1
AffineMap shiftDimsDown1(AffineMap expr, unsigned numDim, unsigned offset) {
  assert(offset <= expr.getNumDims());
  return AffineMap::get(expr.getNumDims() - 1, expr.getNumSymbols(),
                        llvm::map_to_vector<4>(expr.getResults(),
                                               [&](AffineExpr e) {
                                                 return shiftDimsDown1(
                                                     e, expr.getNumDims(),
                                                     offset);
                                               }),
                        expr.getContext());
}

// Helper function to check if an operation dominates the target region
bool dominatesTarget(Operation* op, Region* targetRegion) {
    return op->getParentRegion()->isAncestor(targetRegion);
}

Value recursiveCloneWithDominanceCheck(
    OpBuilder& builder, 
    Value value, 
    Region* targetRegion,
    IRMapping& mapping,
    DenseSet<Operation*>& processedOps) {
    
    // If value is already mapped, return the mapped value
    if (mapping.contains(value)) {
        return mapping.lookup(value);
    }
    
    // Handle block arguments
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getParentBlock()->getParent()->isAncestor(targetRegion)) {
            mapping.map(value, value);
            return value;
        } else {
            llvm::errs() << "Non-dominating block argument encountered\n";
            return nullptr;
        }
    }
    
    Operation* defOp = value.getDefiningOp();
    if (!defOp) {
        return value;
    }
    
    // Check if this operation dominates the target region
    if (dominatesTarget(defOp, targetRegion)) {
        // Operation dominates, use it directly
        mapping.map(value, value);
        return value;
    }
    
    // Avoid processing the same operation multiple times
    if (processedOps.contains(defOp)) {
        // Operation was already processed, should be in mapping
        auto resultNum = cast<OpResult>(value).getResultNumber();
        auto mappedOp = mapping.lookup(defOp->getResult(0)).getDefiningOp();
        auto clonedValue = mappedOp->getResult(resultNum);
        mapping.map(value, clonedValue);
        return clonedValue;
    }
    
    // Check if operation is safe to clone
    if (!isReadOnly(defOp)) {
        llvm::errs() << "Cannot clone non-read-only operation: " << *defOp << "\n";
        return nullptr;
    }
    
    processedOps.insert(defOp);
    
    // Recursively process ALL operands first to populate the mapping
    for (Value operand : defOp->getOperands()) {
        Value clonedOperand = recursiveCloneWithDominanceCheck(
            builder, operand, targetRegion, mapping, processedOps);
        if (!clonedOperand) {
            return nullptr;
        }
        // clonedOperand is automatically added to mapping by recursive call
    }
    
    // Now clone the operation using the populated mapping
    Operation* clonedOp = builder.clone(*defOp, mapping);
    
    // The clone automatically maps all results, so we can just return what we need
    auto resultNum = cast<OpResult>(value).getResultNumber();
    return clonedOp->getResult(resultNum);
}

// Check if the affine apply is a constant and return the constant value
std::optional<int64_t> getConstantFromAffineApply(AffineApplyOp applyOp) {
    AffineMap map = applyOp.getAffineMap();
    
    // Must have no dimensions and no symbols
    if (map.getNumDims() != 0 || map.getNumSymbols() != 0) {
        return std::nullopt;
    }
    
    // Must have exactly one result that is a constant
    if (map.getNumResults() != 1) {
        return std::nullopt;
    }
    
    // Check if the single result is a constant expression
    AffineExpr result = map.getResult(0);
    if (auto constExpr = result.dyn_cast<AffineConstantExpr>()) {
        return constExpr.getValue();
    }
    
    return std::nullopt;
}

// Given an affine map `oldmap`, memref `val`, and corresponding input values
// (which are a list of indicies, then symbols), and a set of loop indices
// `indices` produce the following:
//  1. A (potentially new) memref value `newval` which does not have any
//  dependence on `indices`
//     and
//  2. an affine map `newmap` which takes size(indices) values (`indices`) and
//  produces indices into `newval` such that
//     indexing `newval[map(indices)]` produces the same result as indexing the
//     original map.
// check_reduction is set true, when passed from store/linalg.generic's output
// variable. And it is returned true, only if index was not encountered in
// oldmap operands and check_reduction was set true.
Value remap_in_affine_dim(bool &legal, OpBuilder &builder, AffineMap oldmap,
                          Value memref_val, Value index, Value bound, AffineApplyOp lower_bound,
                          int firstNDims, ValueRange oldmap_operands,
                          Value origmemref, bool &check_reduction) {

  LLVM_DEBUG(llvm::dbgs() << "\n=== remap_in_affine_dim ===\n");
  LLVM_DEBUG(llvm::dbgs() << "  oldmap: " << oldmap << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  firstNDims: " << firstNDims << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  check_reduction (input): " << check_reduction << "\n");

  int lower_bound_val = getConstantFromAffineApply(lower_bound).value_or(0);
  LLVM_DEBUG(llvm::dbgs() << "  lower_bound_val: " << lower_bound_val << "\n");

  assert(oldmap_operands.size() ==
         oldmap.getNumSymbols() + oldmap.getNumDims());
  // Operands which don't correspond to indices
  SmallVector<Value> operands_without_indices;
  ssize_t dimidx = -1;
  for (auto [i, v] : llvm::enumerate(oldmap_operands)) {
    if (v == nullptr) {
      assert(i < firstNDims);
      continue;
    }
    assert(i >= firstNDims);
    if (v != index) {
      // Check if the symbol value is read-only or defined in a scope where it
      // is always visible.
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        // check if it dominates the current scope
        if (ba.getParentBlock()->getParent()->isAncestor(
                builder.getBlock()->getParent()))
          operands_without_indices.push_back(v);
        else {
          assert(false);
          legal = false;
          return nullptr;
        }
      } else {
        auto op = v.getDefiningOp();
        // check if this dominates the current scope
        if (op->getParentRegion()->isAncestor(
                builder.getBlock()->getParent())) {
          operands_without_indices.push_back(v);
        } else if (isReadOnly(op)) {
          // if not, check if it is readnone
          // Technically this isn't quite sufficient yet, and does require that
          // the operands to this op are also able to be hoisted, but for now we
          // will assume this
          auto op2 = builder.clone(*op);
          operands_without_indices.push_back(
              op2->getResult(cast<OpResult>(v).getResultNumber()));
        } else {
          // if so clone it in the right scope
          // otherwise set illegal and don't continue
          assert(false);
          legal = false;
          return nullptr;
        }
      }
    } else
      dimidx = i;
  }
  if ((dimidx == -1) && (check_reduction))
    check_reduction = true;
  else
    check_reduction = false;

  LLVM_DEBUG(llvm::dbgs() << "  dimidx: " << dimidx << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  check_reduction (output): " << check_reduction << "\n");

  // Raising an outer loop around an existing linalg.generic prepends a new
  // iterator dimension: old `linalg.index 0` becomes index 1, etc. Keep the
  // submap in that same logical order. Previously this appended the new
  // dimension after the existing inner dimensions, which made lowered
  // im2col-style layouts use `(w, h, c)` storage while the body used
  // `(c, h, w)` indices.
  SmallVector<AffineExpr> dimReplacements;
  size_t validSims = 0;
  size_t nextInnerDim = 1;
  AffineExpr newLoopDim =
      builder.getAffineDimExpr(0) + builder.getAffineConstantExpr(lower_bound_val);
  for (int i = 0; i < oldmap.getNumDims(); i++) {
    if (i < firstNDims) {
      assert(i != dimidx);
      dimReplacements.push_back(builder.getAffineDimExpr(nextInnerDim));
      nextInnerDim++;
    } else if (i == dimidx) {
      dimReplacements.push_back(newLoopDim);
    } else {
      // TODO: Why are we using symbol here instead of dim?
      dimReplacements.push_back(builder.getAffineSymbolExpr(validSims));
      validSims++;
    }
  }

  SmallVector<AffineExpr> symReplacements;
  for (int i = 0; i < oldmap.getNumSymbols(); i++) {
    if (i + oldmap.getNumDims() == dimidx) {
      symReplacements.push_back(newLoopDim);
    } else {
      symReplacements.push_back(builder.getAffineSymbolExpr(validSims));
      validSims++;
    }
  }
  if (validSims != operands_without_indices.size()) {
    llvm::errs() << " oldmap: " << oldmap << "\n";
    llvm::errs() << " dimidx=" << dimidx << "\n";
    llvm::errs() << " index: " << index << "\n";
    llvm::errs() << "  oldmap_operands: size=" << oldmap_operands.size()
                 << "\n";
    for (auto op : oldmap_operands) {
      if (op) {
        llvm::errs() << "  -" << op << " &" << op.getAsOpaquePointer() << "\n";
      } else {
        llvm::errs() << "  -"
                     << "null"
                     << " &nullptr\n";
      }
    }
    llvm::errs() << " validSims: " << validSims << "\n";
    llvm::errs() << " operands_without_indices: size="
                 << operands_without_indices.size() << "\n";
    for (auto op : operands_without_indices) {
      llvm::errs() << "  -" << op << " &" << op.getAsOpaquePointer() << "\n";
    }
  }
  assert(validSims == operands_without_indices.size());
  auto map2 = oldmap.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                           firstNDims + 1/*Number of dims in new map*/,
                                           operands_without_indices.size() /*Number of symbols in new map*/);
  
  LLVM_DEBUG(llvm::dbgs() << "  new map (map2): " << map2 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  nextInnerDim: " << nextInnerDim
                          << ", validSims: " << validSims << "\n");

  SmallVector<Value> idx_sizes;
  idx_sizes.push_back(bound);
  for (size_t i = 0; i < firstNDims; i++) {
    // memref.dimOp captures the size of the memref
    if (auto submap = origmemref.getDefiningOp<polygeist::SubmapOp>())
      idx_sizes.push_back(submap.getSizes()[i]);
    else
      llvm_unreachable("Won't reach this case");
    // idx_sizes.push_back(builder.create<memref::DimOp>(origmemref.getLoc(),
    // origmemref, i));
  }

  legal = true;
  SmallVector<int64_t> sizes(idx_sizes.size(), mlir::ShapedType::kDynamic);
  for (auto sz : idx_sizes) {
    DenseSet<Operation*> processedOps;
    IRMapping mapping;
    auto clonedOp = recursiveCloneWithDominanceCheck(builder, sz, builder.getBlock()->getParent(), mapping, processedOps);
    if (!clonedOp) {
      legal = false;
      return nullptr;
    }
    operands_without_indices.push_back(clonedOp);
  }

  //for (auto sz : idx_sizes) {
  //  // Check if the symbol value is read-only or defined in a scope where it is
  //  // always visible.
  //  if (auto ba = dyn_cast<BlockArgument>(sz)) {
  //    // check if it dominates the current scope
  //    if (ba.getParentBlock()->getParent()->isAncestor(
  //            builder.getBlock()->getParent()))
  //      operands_without_indices.push_back(sz);
  //    else {
  //      llvm::errs() << " value is a non-dominating block arg: " << sz << "\n";
  //      legal = false;
  //      assert(false);
  //      return nullptr;
  //    }
  //  } else {
  //    auto op = sz.getDefiningOp();
  //    // check if this dominates the current scope
  //    if (op->getParentRegion()->isAncestor(builder.getBlock()->getParent())) {
  //      operands_without_indices.push_back(sz);
  //    } else if (isReadOnly(op)) {
  //      // if not, check if it is readnone
  //      // Technically this isn't quite sufficient yet, and does require that
  //      // the operands to this op are also able to be hoisted, but for now we
  //      // will assume this
  //      // We need to clone the op along and check if it's operands are dominating or not, else do a recursive clone
  //      auto op2 = builder.clone(*op);
  //      operands_without_indices.push_back(
  //          op2->getResult(cast<OpResult>(sz).getResultNumber()));
  //    } else {
  //      llvm::errs() << " op is not readonly: " << *op << "\n";
  //      // if so clone it in the right scope
  //      // otherwise set illegal and don't continue
  //      legal = false;
  //      assert(false);
  //      return nullptr;
  //    }
  //  }
  //}
  auto ty = MemRefType::get(
      sizes, cast<MemRefType>(memref_val.getType()).getElementType());

  ////TODO: Can we have a case where stride is not 1?
  //Value stride = builder.create<arith::ConstantIndexOp>(memref_val.getLoc(), 1);

  //// Create a subview op using lower bound, stride and size
  //// Convert AffineApplyOp to its result Value and wrap in ValueRange
  //Value lowerBoundValue = lower_bound.getResult();
  //auto subViewOp = builder.create<memref::SubViewOp>(
  //    memref_val.getLoc(),                              // Location
  //    memref_val,                       // Source memref
  //    ValueRange{lowerBoundValue},      // Offsets (array)
  //    ValueRange{bound},                // Sizes (array)
  //    ValueRange{stride}                // Strides (array)
  //);

  //Value subview = subViewOp.getResult();

  auto result = builder.create<polygeist::SubmapOp>(
      memref_val.getLoc(), ty, memref_val, operands_without_indices, map2);
  
  LLVM_DEBUG(llvm::dbgs() << "  Created SubmapOp with type: " << ty << "\n");
  LLVM_DEBUG(llvm::dbgs() << "=== remap_in_affine_dim END ===\n\n");
  
  return result;
}

// store A[...]
// val = load A[...]

/*  prevA :
    store A
    val is now prevA
*/

/*

f(%memref )

%memref = ...

affine.for {

    %inp = .. subview %memref [ ... ]

    linalg.generic %inp #map {
      body()
    }
}


->


affine.for j {

    linalg.generic %memref #map2(j) {
      body()
    }
}




#map2 = #map with the indexing done to %inp





%memref = .. subview %memref_base [ ... ]

linalg.generic %[[[memref]]] [[[[#map]]]]([[[[operands]]]]) {
  body()
}

->


output_memref = memref_base
output_map    = subvmap()

 compose
# uts are memref, map, and operands
# outputs are o
memref[map(operands)] ==== output_memref[output_map(output_operands)]



bas= memref<40x40>

B

u

tput_memref, output_map and output_operands
# possible intermediate is ...

getLinalgArgMap(memref, map, operands to map [e.g. input symbols/dims])
  if memref is alloca/unknown/etc
    return memref/map/operands
  else
    memref = subview memref_base[map2(operands2)]

    return memref_base   and a new output_map such that
      memref_base[output_map(output_operands)] === memref[map(operands)]





*/

// Suppose we have a memref expression E=input[affine.map(operands)]
//     if input = memref.subview A[starts, offsets]
//    can we rewrite E as A[affine.map2(operands2)]
//    We update lgMap and lgOperands in place with this coresponding map2 and
//    operands2
LogicalResult getLinalgArgMap(Operation *loop, Value &input, AffineMap &lgMap,
                              SmallVector<Value> &lgOperands) {
  OpBuilder builder(loop->getContext());

  LLVM_DEBUG(llvm::dbgs() << "\n=== getLinalgArgMap ===\n");
  LLVM_DEBUG(llvm::dbgs() << "  Initial lgMap: " << lgMap << "\n");

  while (Operation *defOp = input.getDefiningOp()) {

    assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
    // If the input is defined outside of the loop, we are finished.
    if (!loop->isAncestor(defOp)) {
      LLVM_DEBUG(llvm::dbgs() << "  Input defined outside loop, breaking\n");
      break;
    }

    if (auto SM = dyn_cast<polygeist::SubmapOp>(defOp)) {
      auto submap = SM.getMap();

      LLVM_DEBUG(llvm::dbgs() << "  Found SubmapOp with map: " << submap << "\n");

      // TODO: Do we achieve anything with this compose?
      // As lgMap in our case is 1 to 1 identity map
      auto composeMap = submap.compose(lgMap);
      
      LLVM_DEBUG(llvm::dbgs() << "  Composed map: " << composeMap << "\n");

      SmallVector<Value> operands0;

      // First the dims
      for (size_t i = 0; i < lgMap.getNumDims(); i++)
        operands0.push_back(lgOperands[i]);

      // Then the symbols of submap
      for (size_t i = 0; i < submap.getNumSymbols(); i++)
        operands0.push_back(SM.getSymbols()[i]);

      // Then the symbols of lgMap
      for (size_t i = 0; i < lgMap.getNumSymbols(); i++)
        operands0.push_back(lgOperands[i + lgMap.getNumDims()]);

      lgMap = composeMap;
      lgOperands = operands0;
      input = SM.getBase();
      assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
      continue;
    }

    // if (auto SV = dyn_cast<memref::SubViewOp>(defOp)) {

    //  // TODO update map with the new indexing from here

    //  // Create affine map
    //  //   i. Track number of running dims and symbols
    //  //  ii. shift dims and symbols to generate shifted expressions.
    //  // Extract corresponding operands
    //  // Use affineMap::get with numOperands and numSymbols along with shifted
    //  // expressions to get a map. Use affine map simplify to simplify this

    //  SmallVector<AffineExpr> startExprs;
    //  SmallVector<AffineExpr> strideExprs;
    //  SmallVector<Value> dimOperands;
    //  SmallVector<Value> symOperands;
    //  for (auto &&[first, second] : llvm::zip(SV.getOffsets(),
    //  SV.getStrides())) {
    //    for (auto &&[index, val] : llvm::enumerate(SmallVector<Value>({first,
    //    second}))) {
    //      auto &exprOutput = (index == 0) ? startExprs : strideExprs;
    //      // Only support constants, symbols, or affine apply as offsets
    //      if (auto cop = val.getDefiningOp<arith::ConstantIntOp>()) {
    //        exprOutput.push_back(builder.getAffineConstantExpr(cop.value()));
    //        continue;
    //      } else if (auto cop = val.getDefiningOp<arith::ConstantIndexOp>()) {
    //        exprOutput.push_back(builder.getAffineConstantExpr(cop.value()));
    //        continue;
    //      }
    //      if (auto ba = dyn_cast<BlockArgument>(val)) {
    //        Block *parentBlock = ba.getOwner();
    //        if (isa<AffineForOp,
    //        AffineParallelOp>(parentBlock->getParentOp())) {
    //          exprOutput.push_back(
    //              builder.getAffineDimExpr(dimOperands.size()));
    //          dimOperands.push_back(ba);
    //          continue;

    //        }
    //      }

    //      auto valOp = val.getDefiningOp();
    //      // Defined outside loop, consider it a symbol [for now]
    //      //if (!valOp || loop->isAncestor(defOp)) {
    //      if (valOp&&!loop->isAncestor(defOp)) {
    //        exprOutput.push_back(
    //            builder.getAffineSymbolExpr(symOperands.size()));
    //        symOperands.push_back(val);
    //        continue;
    //      }

    //      //TODO: Maybe it's a case to add, but are we sure we need it for
    //      starts and offsets
    //      // and not for operands
    //      if (auto apply = dyn_cast<AffineApplyOp>(valOp)) {
    //        auto map = apply.getAffineMap();
    //        auto *scope = affine::getAffineScope(valOp)->getParentOp();
    //        DominanceInfo DI(scope);
    //        auto map_operands = apply.getOperands();
    //        //fully2ComposeAffineMapAndOperands(builder, &map, &map_operands,
    //        DI);
    //// Instead of using loop step we are using 1 (Assumption as the stride
    /// size)
    //        auto newexpr = map.shiftDims(dimOperands.size())
    //                           .shiftSymbols(symOperands.size());

    //        for (auto expr : newexpr.getResults()) {
    //          exprOutput.push_back(expr);
    //        }

    //        for (size_t i = 0; i < map.getNumDims(); i++)
    //          dimOperands.push_back(apply.getOperands()[i]);

    //        for (size_t i = 0; i < map.getNumSymbols(); i++)
    //          symOperands.push_back(apply.getOperands()[i +
    //          map.getNumDims()]);

    //        continue;
    //      }

    //      //return failure();
    //    }
    //  }

    //  SmallVector<AffineExpr> inputExprs;
    //  for (auto expr : lgMap.shiftDims(dimOperands.size())
    //                       .shiftSymbols(symOperands.size()).getResults()) {
    //    inputExprs.push_back(expr);
    //  }
    //  for (size_t i = 0; i < lgMap.getNumDims(); i++)
    //    dimOperands.push_back(lgOperands[i]);

    //  for (size_t i = 0; i < lgMap.getNumSymbols(); i++)
    //    symOperands.push_back(lgOperands[i + lgMap.getNumDims()]);

    //  SmallVector<AffineExpr> mergedExprs;
    //  for (auto && [start, stride, idx] :
    //       llvm::zip(startExprs, strideExprs, inputExprs)) {
    //    mergedExprs.push_back(start + idx * stride);
    //  }

    //  lgMap =
    //      AffineMap::get(dimOperands.size(), symOperands.size(), mergedExprs,
    //      loop->getContext());
    //  lgOperands.clear();
    //  lgOperands.insert(lgOperands.begin(), dimOperands.begin(),
    //  dimOperands.end());
    //  lgOperands.insert(lgOperands.begin()+lgOperands.size(),
    //  symOperands.begin(), symOperands.end()); input = SV.getSource(); break;
    //}

    // return failure();
  }
  assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
  
  LLVM_DEBUG(llvm::dbgs() << "  Final lgMap: " << lgMap << "\n");
  LLVM_DEBUG(llvm::dbgs() << "=== getLinalgArgMap END ===\n\n");
  
  return success();
}

//===----------------------------------------------------------------------===//
// Group C — distribute an affine.for whose body has multiple "chunks"
//           (each linalg.generic and each nested affine.for is a chunk).
//
// Match precondition: either
//   (a) the loop was promoted from an affine.parallel (so it carries
//       `polygeist.was_parallel`) — iterations are independent, so it's legal
//       to run all of chunk-1 across iterations, then all of chunk-2, etc.; or
//   (b) the loop is sequential but cross-chunk fission is provably safe: every
//       root memref shared across multiple chunks (with at least one writer)
//       is indexed by the outer IV in the same composed dim across all of
//       those chunks. The check below builds an AccessInfo per
//       affine.load/store, memref.load/store, and linalg.generic operand (via
//       the polygeist.submap chain) and verifies the iv-binding consistency.
//
// After this rewrite each new sibling loop has a homogeneous body that
// AffineForOpRaising can handle.
//===----------------------------------------------------------------------===//

namespace {
struct AccessInfo {
  Value rootMemref;
  // Root-dim positions that are bound to the outer IV via identity (same SSA
  // value as the outer IV appears as the dim operand / submap symbol that
  // feeds this root-dim).
  SmallVector<unsigned, 4> ivBoundRootDims;
  bool isWrite;
};

// For a memref value reached by an access (the direct memref of an affine
// load/store, or the linalg.generic operand which is typically a submap),
// follow at most one polygeist.submap layer to the root, and compute which
// root-dim positions are bound to `outerIV` via identity (a single dim/symbol
// expression that names `outerIV`). Returns std::nullopt if the structure is
// too complex to analyze conservatively (chained submaps, non-trivial
// expressions involving the IV, etc.) — caller must treat that as unsafe.
static std::optional<AccessInfo> analyzeAccessThroughSubmap(
    Value memref, AffineMap accessMap, ValueRange accessOperands, bool isWrite,
    Value outerIV) {
  AccessInfo info;
  info.isWrite = isWrite;

  if (auto submap = memref.getDefiningOp<polygeist::SubmapOp>()) {
    // Chained submaps require full composition; bail conservatively for now.
    if (submap.getBase().getDefiningOp<polygeist::SubmapOp>())
      return std::nullopt;
    info.rootMemref = submap.getBase();
    AffineMap m = submap.getMap();
    ValueRange syms = submap.getSymbols();
    // Each result of `m` is one root-dim. If it names symbol s and syms[s] is
    // the outer IV, mark this root-dim as iv-bound.
    for (unsigned d = 0, e = m.getNumResults(); d < e; ++d) {
      AffineExpr expr = m.getResult(d);
      if (auto sym = expr.dyn_cast<AffineSymbolExpr>()) {
        unsigned sIdx = sym.getPosition();
        if (sIdx < syms.size() && syms[sIdx] == outerIV)
          info.ivBoundRootDims.push_back(d);
      }
      // Any non-trivial expression involving outerIV: if expr references a
      // symbol whose binding is outerIV but isn't a pure SymbolExpr, treat as
      // unanalyzable.
      else {
        bool referencesIv = false;
        expr.walk([&](AffineExpr sub) {
          if (auto s = sub.dyn_cast<AffineSymbolExpr>()) {
            unsigned sIdx = s.getPosition();
            if (sIdx < syms.size() && syms[sIdx] == outerIV)
              referencesIv = true;
          }
        });
        if (referencesIv) return std::nullopt;
      }
    }
    return info;
  }

  // Direct memref access via affine map.
  if (!accessMap) return std::nullopt;
  info.rootMemref = memref;
  for (unsigned d = 0, e = accessMap.getNumResults(); d < e; ++d) {
    AffineExpr expr = accessMap.getResult(d);
    if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
      unsigned dIdx = dim.getPosition();
      if (dIdx < accessOperands.size() && accessOperands[dIdx] == outerIV)
        info.ivBoundRootDims.push_back(d);
    } else {
      bool referencesIv = false;
      expr.walk([&](AffineExpr sub) {
        if (auto dimSub = sub.dyn_cast<AffineDimExpr>()) {
          unsigned dIdx = dimSub.getPosition();
          if (dIdx < accessOperands.size() && accessOperands[dIdx] == outerIV)
            referencesIv = true;
        }
      });
      if (referencesIv) return std::nullopt;
    }
  }
  return info;
}

// Walk a chunk's ops (transitively, into nested regions) and collect
// AccessInfo for every memref access op. Returns false if any access is
// unanalyzable (caller must bail).
static bool collectChunkAccesses(ArrayRef<Operation *> chunk, Value outerIV,
                                 SmallVectorImpl<AccessInfo> &out) {
  bool unanalyzable = false;
  auto visit = [&](Operation *op) {
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      auto info = analyzeAccessThroughSubmap(
          load.getMemref(), load.getAffineMap(),
          ValueRange(load.getMapOperands()), /*isWrite=*/false, outerIV);
      if (!info) { unanalyzable = true; return WalkResult::interrupt(); }
      out.push_back(*info);
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      auto info = analyzeAccessThroughSubmap(
          store.getMemref(), store.getAffineMap(),
          ValueRange(store.getMapOperands()), /*isWrite=*/true, outerIV);
      if (!info) { unanalyzable = true; return WalkResult::interrupt(); }
      out.push_back(*info);
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      AccessInfo info;
      info.rootMemref = load.getMemref();
      info.isWrite = false;
      for (unsigned d = 0, e = load.getIndices().size(); d < e; ++d)
        if (load.getIndices()[d] == outerIV)
          info.ivBoundRootDims.push_back(d);
      out.push_back(info);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      AccessInfo info;
      info.rootMemref = store.getMemref();
      info.isWrite = true;
      for (unsigned d = 0, e = store.getIndices().size(); d < e; ++d)
        if (store.getIndices()[d] == outerIV)
          info.ivBoundRootDims.push_back(d);
      out.push_back(info);
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      for (Value input : generic.getInputs()) {
        auto info = analyzeAccessThroughSubmap(input, AffineMap(), ValueRange(),
                                               /*isWrite=*/false, outerIV);
        if (!info) { unanalyzable = true; return WalkResult::interrupt(); }
        out.push_back(*info);
      }
      for (Value output : generic.getOutputs()) {
        auto info = analyzeAccessThroughSubmap(output, AffineMap(), ValueRange(),
                                               /*isWrite=*/true, outerIV);
        if (!info) { unanalyzable = true; return WalkResult::interrupt(); }
        out.push_back(*info);
      }
    }
    // SubmapOp setup and read-none arith are not accesses themselves.
    return WalkResult::advance();
  };
  for (Operation *op : chunk) {
    op->walk(visit);
    if (unanalyzable) return false;
  }
  return true;
}

// For each shared root memref across chunks with at least one writer, every
// access from any chunk that touches it must (a) bind the outer IV to at
// least one root-dim, and (b) bind it to the same dim-set across chunks.
// Otherwise distributing reorders cross-iteration accesses to address-overlapping
// cells.
static bool
chunksDistributionSafe(ArrayRef<SmallVector<Operation *, 4>> chunks,
                       Value outerIV) {
  SmallVector<SmallVector<AccessInfo>, 4> perChunk(chunks.size());
  for (unsigned i = 0; i < chunks.size(); ++i) {
    if (!collectChunkAccesses(chunks[i], outerIV, perChunk[i])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Distribute REJECTED: unanalyzable access in chunk " << i
                 << "\n");
      return false;
    }
  }
  for (unsigned p = 0; p < chunks.size(); ++p) {
    for (unsigned q = p + 1; q < chunks.size(); ++q) {
      for (const AccessInfo &accP : perChunk[p]) {
        for (const AccessInfo &accQ : perChunk[q]) {
          if (accP.rootMemref != accQ.rootMemref) continue;
          if (!accP.isWrite && !accQ.isWrite) continue;
          if (accP.ivBoundRootDims.empty() || accQ.ivBoundRootDims.empty()) {
            LLVM_DEBUG(llvm::dbgs() << "Distribute REJECTED: shared memref "
                                       "access not bound to outer IV\n");
            return false;
          }
          if (accP.ivBoundRootDims != accQ.ivBoundRootDims) {
            LLVM_DEBUG(llvm::dbgs() << "Distribute REJECTED: shared memref "
                                       "binds outer IV to different root-dims "
                                       "across chunks\n");
            return false;
          }
        }
      }
    }
  }
  return true;
}
} // end anonymous namespace

struct DistributeAffineForOnLinalgGeneric
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool isParallel = forOp->hasAttr("polygeist.was_parallel");
    // Can't distribute loops with iter_args.
    if (forOp.getNumResults() != 0) return failure();

    Block *body = forOp.getBody();
    if (body->empty()) return failure();

    // Anchor-based chunking: each side-effecting op (linalg.generic,
    // affine.store, memref.store, nested affine.for) is an anchor. Its
    // chunk is itself plus the SSA def-use closure of its operands within
    // the body. Chunks must be disjoint (no shared deps); body order
    // determines emit order.

    // Step 1: collect anchors (in body order).
    SmallVector<Operation *> anchors;
    for (Operation &op : *body) {
      if (isa<affine::AffineYieldOp>(op)) continue;
      if (isa<linalg::GenericOp, affine::AffineStoreOp, memref::StoreOp,
              affine::AffineForOp>(op))
        anchors.push_back(&op);
    }
    if (anchors.size() <= 1) return failure();

    // Step 2: compute each anchor's SSA dep closure within the body. If two
    // anchors share a body-local dependency, we can't cleanly split — fail.
    DenseMap<Operation *, unsigned> opToChunk;
    Value iv = forOp.getInductionVar();
    for (unsigned i = 0; i < anchors.size(); ++i) {
      SmallVector<Operation *> work;
      work.push_back(anchors[i]);
      while (!work.empty()) {
        Operation *op = work.pop_back_val();
        auto it = opToChunk.find(op);
        if (it != opToChunk.end()) {
          if (it->second != i) {
            LLVM_DEBUG(llvm::dbgs() << "Distribute REJECTED: shared dependency between chunks\n");
            return failure();
          }
          continue;
        }
        opToChunk[op] = i;
        for (Value operand : op->getOperands()) {
          if (operand == iv) continue;
          Operation *defOp = operand.getDefiningOp();
          if (!defOp) continue;             // block arg / outer-scope
          if (defOp->getBlock() != body) continue;  // outside this body
          work.push_back(defOp);
        }
      }
    }

    // Step 3: collect chunks by chunkIdx, preserving body order.
    SmallVector<SmallVector<Operation *, 4>> chunks(anchors.size());
    for (Operation &op : *body) {
      if (isa<affine::AffineYieldOp>(op)) continue;
      auto it = opToChunk.find(&op);
      if (it == opToChunk.end()) {
        // Op not reachable from any anchor — pure, dead, or feeds an unknown
        // sink. Conservatively bail rather than drop it.
        LLVM_DEBUG(llvm::dbgs() << "Distribute REJECTED: op not in any chunk's closure\n");
        return failure();
      }
      chunks[it->second].push_back(&op);
    }

    // Safety gate: parallel-loop fast path, otherwise cross-chunk dep check.
    if (!isParallel && !chunksDistributionSafe(chunks, iv)) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Distributing affine.for into " << chunks.size()
                            << " sibling loops"
                            << (isParallel ? " (was_parallel)" : " (dep-check)")
                            << "\n");

    // For each chunk, clone the affine.for with just that chunk's ops.
    rewriter.setInsertionPoint(forOp);
    for (auto &chunk : chunks) {
      auto newFor = rewriter.create<affine::AffineForOp>(
          forOp.getLoc(),
          forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
          forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
          forOp.getStep());
      // Only carry the parallel mark forward when the input had it. The
      // dep-check fallback path operates on sequential loops; the sibling
      // loops it produces are equally sequential.
      if (isParallel)
        newFor->setAttr("polygeist.was_parallel", rewriter.getUnitAttr());

      Block *newBody = newFor.getBody();
      // newBody already has a default affine.yield from the builder.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(newBody);

      IRMapping mapping;
      mapping.map(iv, newFor.getInductionVar());
      for (Operation *op : chunk)
        rewriter.clone(*op, mapping);
      // Leave the builder-inserted affine.yield alone (it terminates the body).
    }

    rewriter.eraseOp(forOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PrivatizeScratchAllocaForLoop
//
// Looks for a 0-D scalar `memref.alloca` (defined in the enclosing function,
// outside the loop) that is used as per-iteration scratch by the loop body —
// i.e., every iteration starts by overwriting the scalar before reading it,
// and nothing outside the loop reads it after the loop. Expands the alloca
// to `memref<? x T>` with one slot per loop iteration and rewrites every
// in-loop use to address `new_alloca[iv]` instead of `alloca[]`.
//
// After this rewrite, all accesses to the scratch are bound to the outer
// IV at root-dim 0, which is exactly what the dep-check in
// DistributeAffineForOnLinalgGeneric needs to fire on the loop.
//
// Constraints (kept tight for v1):
//  - Loop has constant lb 0 (so `iv` can be used as a direct index).
//  - Loop has no iter_args.
//  - Alloca type is `memref<T>` (0-D scalar).
//  - The first use of the alloca inside the loop body is a write.
//  - The alloca has no uses after the loop.
//===----------------------------------------------------------------------===//

namespace {
// Does this op write to `alloca` without first reading from it?
static bool isInitWriteForScalarAlloca(Operation *op, Value alloca) {
  if (auto store = dyn_cast<affine::AffineStoreOp>(op))
    return store.getMemref() == alloca;
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getMemref() == alloca;
  return false;
}

// Find the first use of `alloca` in body order; return null if none.
static Operation *firstUseInBody(Value alloca, Block *body) {
  for (Operation &op : *body)
    for (Value v : op.getOperands())
      if (v == alloca) return &op;
  return nullptr;
}

// Returns true iff `user` is executed strictly before `loopOp` in the program
// flow, accounting for the possibility that they live in different (but
// nested) blocks.
static bool isBeforeLoopInProgramOrder(Operation *user, Operation *loopOp) {
  DenseMap<Block *, Operation *> loopBlockToAncestor;
  for (Operation *l = loopOp; l; l = l->getParentOp())
    loopBlockToAncestor[l->getBlock()] = l;
  for (Operation *u = user; u; u = u->getParentOp()) {
    auto it = loopBlockToAncestor.find(u->getBlock());
    if (it == loopBlockToAncestor.end()) continue;
    if (u == it->second) return false; // same op — neither before nor after
    return u->isBeforeInBlock(it->second);
  }
  return false;
}

// Verify the alloca is unused past `loopOp`.
static bool noUsesAfterLoop(Value alloca, Operation *loopOp) {
  for (Operation *user : alloca.getUsers()) {
    if (loopOp->isAncestor(user)) continue; // inside the loop — fine
    if (isBeforeLoopInProgramOrder(user, loopOp)) continue; // before — fine
    return false;
  }
  return true;
}
} // anonymous namespace

struct PrivatizeScratchAllocaForLoop
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (forOp.getNumResults() != 0) return failure();
    if (!forOp.hasConstantLowerBound() || forOp.getConstantLowerBound() != 0)
      return failure();

    // We need the loop's iteration count as an SSA Value to size the new
    // alloca. For constant ub, materialize a constant; otherwise emit an
    // affine.apply at the loop's site.
    Block *body = forOp.getBody();
    Value iv = forOp.getInductionVar();

    // Find candidate allocas: any operand inside the body whose defining op
    // is a `memref.alloca` outside the loop with 0-D scalar type.
    SmallVector<memref::AllocaOp> candidates;
    DenseSet<Operation *> seen;
    body->walk([&](Operation *op) {
      for (Value v : op->getOperands()) {
        auto allocaOp = v.getDefiningOp<memref::AllocaOp>();
        if (!allocaOp) continue;
        if (forOp->isAncestor(allocaOp)) continue; // inside this loop already
        if (!seen.insert(allocaOp).second) continue;
        auto mrt = dyn_cast<MemRefType>(allocaOp.getType());
        if (!mrt || mrt.getRank() != 0) continue;
        if (allocaOp->getNumOperands() != 0) continue; // dynamic-shape alloca: skip
        candidates.push_back(allocaOp);
      }
    });
    if (candidates.empty()) return failure();

    // Filter candidates: first in-body use is a write, all in-loop users are
    // among the rewriteable set, no uses after loop, and the alloca lives
    // in some ancestor block of `forOp` so we can place the sized
    // replacement at the same scope (and have AffineForOpRaising later
    // lift enclosing loops without dominance issues).
    SmallVector<memref::AllocaOp> good;
    for (memref::AllocaOp a : candidates) {
      Operation *firstUse = firstUseInBody(a, body);
      if (!firstUse) continue;
      if (!isInitWriteForScalarAlloca(firstUse, a)) continue;
      if (!noUsesAfterLoop(a, forOp)) continue;
      bool allHandled = true;
      for (Operation *user : a->getUsers()) {
        if (!forOp->isAncestor(user)) continue;
        if (!isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp,
                 memref::StoreOp, polygeist::SubmapOp>(user)) {
          allHandled = false;
          break;
        }
      }
      if (!allHandled) continue;
      good.push_back(a);
    }
    if (good.empty()) return failure();

    AffineMap idxMap = AffineMap::get(/*dimCount=*/1, /*symCount=*/0,
                                      rewriter.getAffineDimExpr(0),
                                      rewriter.getContext());

    for (memref::AllocaOp oldAlloca : good) {
      // Find the ancestor of `forOp` that lives in the same block as
      // `oldAlloca`. That's where we want to insert: same block as the old
      // alloca, just before the outermost enclosing loop. This keeps the
      // new alloca at the scratch's original scope so AffineForOpRaising
      // can later lift the enclosing loops without hitting dominance
      // failures on the size operand.
      Block *allocaBlock = oldAlloca->getBlock();
      Operation *insertionAnchor = forOp.getOperation();
      while (insertionAnchor && insertionAnchor->getBlock() != allocaBlock)
        insertionAnchor = insertionAnchor->getParentOp();
      if (!insertionAnchor) continue; // shouldn't happen given precondition
      rewriter.setInsertionPoint(insertionAnchor);
      AffineMap ubMap = forOp.getUpperBoundMap();
      Value tripCount;
      if (forOp.hasConstantUpperBound()) {
        tripCount = rewriter.create<arith::ConstantIndexOp>(
            forOp.getLoc(), forOp.getConstantUpperBound());
      } else {
        tripCount = rewriter.create<affine::AffineApplyOp>(
            forOp.getLoc(), ubMap,
            SmallVector<Value>(forOp.getUpperBoundOperands()));
      }
      MemRefType oldTy = cast<MemRefType>(oldAlloca.getType());
      auto newTy = MemRefType::get({ShapedType::kDynamic}, oldTy.getElementType());
      auto newAlloca = rewriter.create<memref::AllocaOp>(oldAlloca.getLoc(),
                                                         newTy, tripCount);

      // Rewrite every in-loop use of oldAlloca.
      SmallVector<Operation *> users(oldAlloca->getUsers().begin(),
                                     oldAlloca->getUsers().end());
      for (Operation *user : users) {
        if (!forOp->isAncestor(user)) continue;
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(user);
        if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
          auto newLoad = rewriter.create<affine::AffineLoadOp>(
              load.getLoc(), newAlloca, idxMap, ValueRange{iv});
          rewriter.replaceOp(load, newLoad.getResult());
        } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
          rewriter.create<affine::AffineStoreOp>(
              store.getLoc(), store.getValue(), newAlloca, idxMap,
              ValueRange{iv});
          rewriter.eraseOp(store);
        } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
          auto newLoad = rewriter.create<memref::LoadOp>(
              load.getLoc(), newAlloca, ValueRange{iv});
          rewriter.replaceOp(load, newLoad.getResult());
        } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
          rewriter.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                            newAlloca, ValueRange{iv});
          rewriter.eraseOp(store);
        } else if (auto submap = dyn_cast<polygeist::SubmapOp>(user)) {
          // Original submap: takes 0-D scalar base + (viewSize) operands +
          // 0 symbols. Rewrite to take 1-D base + (iv, viewSize) operands +
          // 1 extra symbol (s_iv) that selects new_alloca[iv]. The result
          // expression for the inner-most root-dim becomes s_iv; the view
          // shape (and hence later linalg semantics) is unchanged.
          AffineMap oldMap = submap.getMap();
          unsigned numDims = oldMap.getNumDims();
          unsigned numSyms = oldMap.getNumSymbols();
          // New map has numDims dims, numSyms+1 symbols. s_iv is symbol
          // position numSyms. Result is a single expression: s_iv (the
          // address into new_alloca). Note: the old map's results were
          // 0-rank (no result expressions, since old base was 0-D). The new
          // base is 1-D, so the new map has exactly one result.
          AffineExpr sIv = rewriter.getAffineSymbolExpr(numSyms);
          AffineMap newMap = AffineMap::get(numDims, numSyms + 1, {sIv},
                                            rewriter.getContext());
          // SubmapOp builder takes (loc, resultType, base, indices_and_sizes,
          // map) — indices_and_sizes is [syms..., sizes...]. Append iv as a
          // new trailing symbol so it pairs with the new s_iv we added.
          SmallVector<Value> indicesAndSizes;
          for (Value s : submap.getSymbols()) indicesAndSizes.push_back(s);
          indicesAndSizes.push_back(iv);
          for (Value sz : submap.getSizes()) indicesAndSizes.push_back(sz);
          auto newSubmap = rewriter.create<polygeist::SubmapOp>(
              submap.getLoc(), submap.getType(), newAlloca, indicesAndSizes,
              newMap);
          rewriter.replaceOp(submap, newSubmap.getResult());
        } else {
          // Unhandled user. Bail entire pattern by deleting the new alloca
          // and returning failure.
          // (Other uses we've already rewritten above will still be live;
          // the simplest recovery is to refuse the rewrite up front. Since
          // we're inside a greedy driver, returning failure here without a
          // clean rollback would leave inconsistent IR. So instead, we
          // checked-cast above and bail before any rewrite for unknown
          // users.)
          // — but for safety: we already early-bailed in the precondition
          // pass below. Reaching this should be impossible.
          llvm_unreachable("unhandled alloca user in privatization");
        }
      }
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// PrivatizeRowScratchAllocaForLoop
//
// Rank-1 (1-D row) extension of PrivatizeScratchAllocaForLoop. Recognises
// per-iteration scratch row buffers ("scratch row carries"): an outer
// `affine.for L` has a `memref.alloca` of static rank-1 `memref<N x T>`
// defined OUTSIDE L, where each iteration of L writes the full row before
// any read and nothing outside L observes the buffer.
//
// Canonical example (NPB MG psinv/resid/rprj3):
//     %r1 = memref.alloca() : memref<35xf64>     // outside both loops
//     affine.for %i3 ... {
//       affine.for %i2 ... {                     // <-- L (this pattern)
//         affine.for %i1 = 0 to N { affine.store v, %r1[%i1] }   // fill
//         affine.for %i1 = 1 to N-1 { ... %r1[%i1-1] + %r1[%i1] + %r1[%i1+1] ... }
//       }
//     }
// Rewrite expands `r1` to `memref<? x N x T>` sized by L's trip count
// and emits ONE `memref.subview new[%iv, 0] [1, N] [1, 1] -> rank-1`
// at L's body entry that all in-loop users share. Each iteration of L
// then writes a disjoint slice, the dep check sees no cross-iteration
// conflict, and downstream Distribute / AffineForOpRaising can lift L.
//
// KNOWN PIPELINE INTEGRATION ISSUE: the strided result type of
// `memref.subview` (with dynamic offset) makes `AffineForOpRaising`'s
// polyhedral analysis blow up in practical time on mg_psinv-shaped
// inputs. See [[row-scratch-privatization-attempt]] for diagnosis. The
// pattern is enabled here to surface the failure modes for diagnosis,
// not as a finished feature.
//===----------------------------------------------------------------------===//

#define PRIV_ROW_DBG(X) llvm::errs() << "[PrivRow] " << X << "\n"

namespace {
// Walk `body` recursively in pre-order and return the first op that
// substantively touches `alloca` — reads or writes. View-creation ops
// (memref.subview, polygeist.submap) are skipped because they only
// reshape the address.
static Operation *firstTouchInBody(Value alloca, Region &body) {
  Operation *found = nullptr;
  body.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (found) return WalkResult::interrupt();
    if (isa<memref::SubViewOp, polygeist::SubmapOp>(op))
      return WalkResult::advance();
    for (Value v : op->getOperands()) {
      if (v == alloca) { found = op; return WalkResult::interrupt(); }
    }
    return WalkResult::advance();
  });
  return found;
}

// Returns true iff `op` writes `alloca` (store / affine.store / a
// linalg.generic that has `alloca` in its `outs`).
static bool isWriteOfAlloca(Operation *op, Value alloca) {
  if (auto s = dyn_cast<affine::AffineStoreOp>(op))
    return s.getMemref() == alloca;
  if (auto s = dyn_cast<memref::StoreOp>(op))
    return s.getMemref() == alloca;
  if (auto g = dyn_cast<linalg::GenericOp>(op))
    for (Value o : g.getOutputs())
      if (o == alloca) return true;
  return false;
}
} // anonymous namespace

struct PrivatizeRowScratchAllocaForLoop
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (forOp.getNumResults() != 0) return failure();
    // Pattern-firing marker: once we've privatized for this loop, don't
    // re-fire — the new alloca is rank-2 and wouldn't match anyway, but
    // this short-circuits the candidate walk on every greedy re-visit.
    if (forOp->hasAttr("polygeist.row_privatized")) return failure();

    Block *body = forOp.getBody();
    Value iv = forOp.getInductionVar();

    // Collect rank-1 static allocas defined outside this loop.
    SmallVector<memref::AllocaOp> candidates;
    DenseSet<Operation *> seen;
    body->walk([&](Operation *op) {
      for (Value v : op->getOperands()) {
        auto allocaOp = v.getDefiningOp<memref::AllocaOp>();
        if (!allocaOp) continue;
        if (forOp->isAncestor(allocaOp)) continue;
        if (!seen.insert(allocaOp).second) continue;
        auto mrt = dyn_cast<MemRefType>(allocaOp.getType());
        if (!mrt || mrt.getRank() != 1) continue;
        if (mrt.isDynamicDim(0)) continue;
        if (allocaOp->getNumOperands() != 0) continue;
        candidates.push_back(allocaOp);
      }
    });
    if (candidates.empty()) return failure();

    // Helper: innermost-enclosing-loop check.
    auto innerContainsAllUses = [&](affine::AffineForOp inner,
                                     Value alloca) -> bool {
      for (Operation *user : alloca.getUsers())
        if (!inner->isAncestor(user)) return false;
      return true;
    };

    SmallVector<memref::AllocaOp> good;
    for (memref::AllocaOp a : candidates) {
      Operation *firstUse = firstTouchInBody(a.getResult(),
                                               forOp.getRegion());
      if (!firstUse) continue;
      if (!isWriteOfAlloca(firstUse, a.getResult())) continue;
      if (!noUsesAfterLoop(a, forOp)) continue;

      bool allHandled = true;
      for (Operation *user : a->getUsers()) {
        if (!forOp->isAncestor(user)) continue;
        if (!isa<linalg::GenericOp, memref::SubViewOp, polygeist::SubmapOp,
                 affine::AffineLoadOp, affine::AffineStoreOp,
                 memref::LoadOp, memref::StoreOp>(user)) {
          allHandled = false;
          break;
        }
      }
      if (!allHandled) continue;

      // Innermost-loop check: defer to nested affine.for if it already
      // contains every user of alloca.
      bool isInnermost = true;
      forOp.getBody()->walk([&](affine::AffineForOp inner) {
        if (inner == forOp) return WalkResult::advance();
        if (innerContainsAllUses(inner, a.getResult())) {
          isInnermost = false;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!isInnermost) continue;

      good.push_back(a);
    }
    if (good.empty()) return failure();

    for (memref::AllocaOp oldAlloca : good) {
      Block *allocaBlock = oldAlloca->getBlock();
      Operation *insertionAnchor = forOp.getOperation();
      while (insertionAnchor && insertionAnchor->getBlock() != allocaBlock)
        insertionAnchor = insertionAnchor->getParentOp();
      if (!insertionAnchor) continue;
      rewriter.setInsertionPoint(insertionAnchor);

      Value tripCount;
      if (forOp.hasConstantUpperBound()) {
        tripCount = rewriter.create<arith::ConstantIndexOp>(
            forOp.getLoc(), forOp.getConstantUpperBound());
      } else {
        tripCount = rewriter.create<affine::AffineApplyOp>(
            forOp.getLoc(), forOp.getUpperBoundMap(),
            SmallVector<Value>(forOp.getUpperBoundOperands()));
      }

      MemRefType oldTy = cast<MemRefType>(oldAlloca.getType());
      int64_t N = oldTy.getShape()[0];
      auto newTy = MemRefType::get({ShapedType::kDynamic, N},
                                    oldTy.getElementType());
      auto newAlloca = rewriter.create<memref::AllocaOp>(
          oldAlloca.getLoc(), newTy, tripCount);

      // ONE subview at forOp's body entry, shared by all in-loop users.
      Value rowView;
      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        SmallVector<OpFoldResult> offsets;
        offsets.push_back(iv);
        offsets.push_back(rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        sizes.push_back(rewriter.getIndexAttr(1));
        sizes.push_back(rewriter.getIndexAttr(N));
        SmallVector<OpFoldResult> strides;
        strides.push_back(rewriter.getIndexAttr(1));
        strides.push_back(rewriter.getIndexAttr(1));
        auto resTy = memref::SubViewOp::inferRankReducedResultType(
            {N}, newTy, offsets, sizes, strides).cast<MemRefType>();
        rowView = rewriter.create<memref::SubViewOp>(
            oldAlloca.getLoc(), resTy, newAlloca, offsets, sizes, strides);
      }

      // Rewrite every in-loop user.
      SmallVector<Operation *> users(oldAlloca->getUsers().begin(),
                                      oldAlloca->getUsers().end());
      for (Operation *user : users) {
        if (!forOp->isAncestor(user)) continue;
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(user);

        if (auto gen = dyn_cast<linalg::GenericOp>(user)) {
          rewriter.startRootUpdate(gen);
          for (auto &operand : gen->getOpOperands())
            if (operand.get() == oldAlloca.getResult())
              operand.set(rowView);
          rewriter.finalizeRootUpdate(gen);
          continue;
        }
        if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
          auto newSv = rewriter.create<memref::SubViewOp>(
              sv.getLoc(), sv.getType(), rowView,
              sv.getMixedOffsets(), sv.getMixedSizes(), sv.getMixedStrides());
          rewriter.replaceOp(sv, newSv.getResult());
          continue;
        }
        if (auto sm = dyn_cast<polygeist::SubmapOp>(user)) {
          rewriter.startRootUpdate(sm);
          sm->setOperand(0, rowView);
          rewriter.finalizeRootUpdate(sm);
          continue;
        }
        if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
          rewriter.replaceOp(load,
              rewriter.create<affine::AffineLoadOp>(
                  load.getLoc(), rowView, load.getAffineMap(),
                  load.getMapOperands()).getResult());
          continue;
        }
        if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
          rewriter.create<affine::AffineStoreOp>(
              store.getLoc(), store.getValue(), rowView,
              store.getAffineMap(), store.getMapOperands());
          rewriter.eraseOp(store);
          continue;
        }
        if (auto load = dyn_cast<memref::LoadOp>(user)) {
          rewriter.replaceOp(load,
              rewriter.create<memref::LoadOp>(
                  load.getLoc(), rowView, load.getIndices()).getResult());
          continue;
        }
        if (auto store = dyn_cast<memref::StoreOp>(user)) {
          rewriter.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                             rowView, store.getIndices());
          rewriter.eraseOp(store);
          continue;
        }
        llvm_unreachable("unhandled user in row-scratch privatization");
      }
      rewriter.eraseOp(oldAlloca);
    }

    forOp->setAttr("polygeist.row_privatized", rewriter.getUnitAttr());
    return success();
  }
};

// Shift every `linalg.index` op nested in `region` by `shift`. Used when an
// outer loop is being raised and prepends `shift` new iterator dims to an
// inner linalg's iteration space: each existing `linalg.index N` becomes
// `linalg.index N + shift`.
static void shiftLinalgIndexDims(Region &region, unsigned shift) {
  if (shift == 0) return;
  region.walk([&](linalg::IndexOp idxOp) {
    idxOp.setDim(idxOp.getDim() + shift);
  });
}

// Group A — triangular-bound support helpers.
// Returns true iff every operand of `operands` is an SSA value defined strictly
// outside of `loop` (i.e., loop-invariant w.r.t. `loop`). This is the safety
// criterion for using an outer-scope-derived bound as an in-body mask.
static bool allOperandsAreLoopInvariantWrt(ValueRange operands,
                                           affine::AffineForOp loop) {
  for (Value v : operands) {
    if (Operation *defOp = v.getDefiningOp()) {
      if (loop->isAncestor(defOp)) return false;
    } else if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      Operation *parent = blockArg.getOwner()->getParentOp();
      if (!parent) return false;
      if (parent == loop.getOperation()) return false;
      if (loop->isAncestor(parent)) return false;
    } else {
      return false;
    }
  }
  return true;
}

// Bound-mask info captured at loop acceptance time and consumed at body-build
// time to emit a `linalg.index + affine.apply + cmpi + select` guard.
struct BoundMaskInfo {
  bool needed = false;
  AffineMap origMap;
  SmallVector<Value> origOperands;
};

static bool affineStoresProvablyDisjoint(affine::AffineStoreOp lhs,
                                         affine::AffineStoreOp rhs) {
  if (lhs.getMemref() != rhs.getMemref())
    return true;

  AffineMap lhsMap = lhs.getAffineMap();
  AffineMap rhsMap = rhs.getAffineMap();
  if (lhsMap.getNumResults() != rhsMap.getNumResults())
    return false;

  for (auto pair : llvm::zip(lhsMap.getResults(), rhsMap.getResults())) {
    auto lhsConst = std::get<0>(pair).dyn_cast<AffineConstantExpr>();
    auto rhsConst = std::get<1>(pair).dyn_cast<AffineConstantExpr>();
    if (lhsConst && rhsConst && lhsConst.getValue() != rhsConst.getValue())
      return true;
  }

  return false;
}

static bool storesProvablyDisjoint(Operation *lhs, Operation *rhs) {
  if (auto lhsAffine = dyn_cast<affine::AffineStoreOp>(lhs)) {
    if (auto rhsAffine = dyn_cast<affine::AffineStoreOp>(rhs))
      return affineStoresProvablyDisjoint(lhsAffine, rhsAffine);
  }

  if (auto lhsStore = dyn_cast<memref::StoreOp>(lhs)) {
    if (auto rhsStore = dyn_cast<memref::StoreOp>(rhs))
      return lhsStore.getMemref() != rhsStore.getMemref();
  }

  return false;
}

static bool onlyFeedsNestedGenericThroughReadNone(Value value, Operation *scope,
                                                  Operation *nestedGeneric,
                                                  DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return true;

  for (Operation *user : value.getUsers()) {
    if (!scope->isAncestor(user))
      return false;
    if (user == nestedGeneric || nestedGeneric->isAncestor(user))
      continue;
    if (!isReadNone(user))
      return false;
    for (Value result : user->getResults())
      if (!onlyFeedsNestedGenericThroughReadNone(result, scope, nestedGeneric,
                                                seen))
        return false;
  }
  return true;
}

struct PromotedScalarLoad {
  Value input;
  AffineMap indexingMap;
};

static bool sameAffineLoadStoreAddress(affine::AffineLoadOp load,
                                       affine::AffineStoreOp store) {
  return load.getMemref() == store.getMemref() &&
         load.getAffineMap() == store.getAffineMap() &&
         load.getMapOperands() == store.getMapOperands();
}

static Value getOperandDimSize(OpBuilder &builder, Location loc, Value operand,
                               unsigned dim) {
  if (auto submap = operand.getDefiningOp<polygeist::SubmapOp>())
    return submap.getSizes()[dim];
  return linalg::createOrFoldDimOp(builder, loc, operand, dim);
}

static LogicalResult
collectNestedGenericLoopSizes(linalg::GenericOp generic, OpBuilder &builder,
                              SmallVectorImpl<Value> &loopSizes) {
  loopSizes.assign(generic.getNumLoops(), Value());

  SmallVector<Value> operands;
  operands.append(generic.getInputs().begin(), generic.getInputs().end());
  operands.append(generic.getOutputs().begin(), generic.getOutputs().end());

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != operands.size())
    return failure();

  for (auto indexedOperand : llvm::enumerate(operands)) {
    AffineMap map = maps[indexedOperand.index()];
    if (!map.isProjectedPermutation())
      return failure();

    Value operand = indexedOperand.value();
    auto operandType = dyn_cast<MemRefType>(operand.getType());
    if (!operandType)
      return failure();
    if (map.getNumResults() != operandType.getRank())
      return failure();

    for (auto indexedExpr : llvm::enumerate(map.getResults())) {
      auto dimExpr = indexedExpr.value().dyn_cast<AffineDimExpr>();
      if (!dimExpr)
        continue;
      unsigned loopDim = dimExpr.getPosition();
      if (loopDim >= loopSizes.size())
        return failure();
      if (!loopSizes[loopDim])
        loopSizes[loopDim] = getOperandDimSize(
            builder, generic.getLoc(), operand, indexedExpr.index());
    }
  }

  for (Value loopSize : loopSizes)
    if (!loopSize)
      return failure();
  return success();
}

// Hybrid raiser for loop bodies that are semantically elementwise stores but
// cannot be expressed as pure linalg ins/outs because the value computation
// contains guarded memory reads (for example im2col padding:
// `scf.if oob then 0 else memref.load input[idx]`). MLIR allows such a region
// inside linalg.generic, so keep the guarded load in the payload and only raise
// the output iteration space to linalg. This gives downstream matchers a stable
// `linalg.generic` anchor without speculating the load past its bounds check.
struct HybridAffineForOpRaising : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const final {
    if (loop.getNumResults() != 0)
      return failure();
    if (!loop.hasConstantLowerBound() || loop.getConstantLowerBound() != 0)
      return failure();
    if (loop.getStep() != 1)
      return failure();

    Block *loopBody = loop.getBody();
    Operation *terminator = loopBody->getTerminator();

    affine::AffineStoreOp targetStore;
    SmallVector<affine::AffineLoadOp> outputLoads;
    bool hasHybridPayload = false;
    bool illegal = false;

    loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == loop)
        return WalkResult::advance();

      if (isa<affine::AffineYieldOp, scf::YieldOp>(op))
        return WalkResult::advance();

      if (isa<affine::AffineForOp, affine::AffineParallelOp,
              linalg::GenericOp>(op)) {
        illegal = true;
        return WalkResult::interrupt();
      }

      if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        if (store->getParentOp() != loop || targetStore) {
          illegal = true;
          return WalkResult::interrupt();
        }
        targetStore = store;
        return WalkResult::advance();
      }

      if (isa<memref::StoreOp, CallOpInterface>(op)) {
        illegal = true;
        return WalkResult::interrupt();
      }

      if (isa<scf::IfOp, memref::LoadOp>(op)) {
        hasHybridPayload = true;
        return WalkResult::advance();
      }

      if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        outputLoads.push_back(load);
        return WalkResult::advance();
      }

      if (isReadNone(op))
        return WalkResult::advance();

      illegal = true;
      return WalkResult::interrupt();
    });

    if (illegal || !targetStore || !hasHybridPayload)
      return failure();
    if (targetStore->getNextNode() != terminator)
      return failure();

    for (affine::AffineLoadOp outputLoad : outputLoads)
      if (!sameAffineLoadStoreAddress(outputLoad, targetStore))
        return failure();

    Value storedValue = targetStore.getValueToStore();

    AffineMap ubMap = loop.getUpperBoundMap();
    SmallVector<Value> ubOperands(loop.getUpperBoundOperands());
    AffineMap lbMap = loop.getLowerBoundMap();
    SmallVector<Value> lbOperands(loop.getLowerBoundOperands());
    if (!ubMap || ubMap.getNumResults() != 1 || !lbMap ||
        lbMap.getNumResults() != 1)
      return failure();

    auto ubValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), ubMap, ubOperands);
    auto lbValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), lbMap, lbOperands);
    auto loopSize =
        rewriter.create<arith::SubIOp>(loop.getLoc(), ubValue, lbValue);

    bool legal = true;
    bool checkReduction = true;
    size_t firstNDims = 0;
    Value newOutput = remap_in_affine_dim(
        legal, rewriter, targetStore.getAffineMap(), targetStore.getMemref(),
        loop.getInductionVar(), loopSize, lbValue, firstNDims,
        targetStore.getMapOperands(), targetStore.getMemref(), checkReduction);
    if (!legal)
      return failure();

    SmallVector<Value> inputs;
    SmallVector<Value> outputs{newOutput};
    SmallVector<AffineMap> affineMaps{
        rewriter.getMultiDimIdentityMap(firstNDims + 1)};
    SmallVector<utils::IteratorType> iteratorTypes{
        checkReduction ? utils::IteratorType::reduction
                       : utils::IteratorType::parallel};

    StringAttr empty = StringAttr::get(loop.getContext());
    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loop.getLoc(), TypeRange(), inputs, outputs, affineMaps, iteratorTypes,
        empty, empty);

    rewriter.setInsertionPointToStart(loopBody);
    auto idx = rewriter.create<linalg::IndexOp>(loop.getLoc(), 0);
    rewriter.replaceAllUsesWith(loop.getInductionVar(), idx);

    auto &genericBody = genericOp.getRegion();
    genericBody.takeBody(loop.getRegion());

    Block *newBody = &genericBody.front();
    newBody->eraseArguments(0, newBody->getNumArguments());
    Value outputArg =
        newBody->addArgument(targetStore.getValueToStore().getType(),
                             targetStore.getLoc());
    for (affine::AffineLoadOp outputLoad : outputLoads) {
      if (storedValue == outputLoad.getResult())
        storedValue = outputArg;
      rewriter.replaceOp(outputLoad, outputArg);
    }

    rewriter.eraseOp(targetStore);
    rewriter.eraseOp(newBody->getTerminator());
    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<linalg::YieldOp>(loop.getLoc(), storedValue);

    rewriter.eraseOp(loop);
    return success();
  }
};

struct AffineForOpRaising : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const final {

    LLVM_DEBUG(llvm::dbgs() << "\n========================================\n");
    LLVM_DEBUG(llvm::dbgs() << "=== AffineForOpRaising::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "========================================\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing loop:\n" << loop << "\n\n");

    auto module = loop->getParentOfType<ModuleOp>();

    // Don't handle accumulations in registers for the moment, we can have
    // a separate pattern move them into memref's
    if (loop.getNumResults() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Loop has results\n\n");
      return failure();
    }

    SmallVector<std::pair<std::vector<Condition>, AffineLoadOp>> loads;
    SmallVector<std::pair<std::vector<Condition>, AffineStoreOp>> stores;
    SmallVector<std::pair<std::vector<Condition>, GenericOp>> linalgGenerics;
    bool check_reduction;

    // TODO Also collect all the linalg generics!

    // Check that the only operations within the region are either:
    //      affine.load, affine.store, affine.if, affine.yield
    // Additionally, for each load/store, remember what conditions are
    // required for that load or store to execute.
    auto result = loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == loop)
        return WalkResult::advance();
      // TODO extend this, any non-memory operation is also legal here.
      // mul, add, etc (we can just check propety)
      if (isa<AffineYieldOp, AffineIfOp>(op)) {
        return WalkResult::advance();
      }
      if (isa<AffineLoadOp, AffineStoreOp>(op) || isa<GenericOp>(op)) {
        Operation *cur = op->getParentOp();
        std::vector<Condition> conditions;
        while (cur != loop) {
          auto ifstmt = dyn_cast<AffineIfOp>(cur);
          if (!ifstmt) {
            return WalkResult::interrupt();
          }
          bool ifTrue =
              ifstmt.getThenRegion().isAncestor(cur->getParentRegion());
          conditions.emplace_back(ifTrue, ifstmt);
          cur = ifstmt->getParentOp();
        }
        if (auto linalgGeneric = dyn_cast<GenericOp>(op)) {
          linalgGenerics.emplace_back(conditions, linalgGeneric);
          // Treat a nested linalg.generic as a single payload op for this
          // wrapping step. Its region may legally contain guarded loads after
          // HybridAffineForOpRaising, and those operations should not be
          // re-classified as top-level affine loop accesses here.
          return WalkResult::skip();
        } else if (auto load = dyn_cast<AffineLoadOp>(op)) {
          loads.emplace_back(conditions, load);
        } else {
          auto store = cast<AffineStoreOp>(op);
          stores.emplace_back(conditions, store);
        }
        return WalkResult::advance();
      }
      // IsReadNone takes care of apply and subview too?
      if (isReadNone(op)) {
        return WalkResult::advance();
      }
      return WalkResult::interrupt();
    });

    if (result.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Walk was interrupted (invalid operations found)\n\n");
      return failure();
    }

    if (!(linalgGenerics.size() == 1 || linalgGenerics.size() == 0)) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: More than one linalg generic\n\n");
      return failure();
    }
    if ((linalgGenerics.size() == 1) && !stores.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Linalg generic exists with stores\n\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Pattern recognition complete:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Loads: " << loads.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Stores: " << stores.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  LinalgGenerics: " << linalgGenerics.size() << "\n\n");

    DominanceInfo DI(loop);

    // Check that all of the stores do not alias the loaded values (otherwise we
    // could get an incorrect result)
    // TODO we can extend this and handle things like reductions, but we're
    // going to start easy for now
    // TODO
    DenseMap<AffineLoadOp, AffineStoreOp> stores_map;
    for (auto &&[_, store] : stores) {
      for (auto &&[_, load] : loads) {
        if (mayAlias(load.getMemref(), store.getMemref())) {
          // We have one exception in this case -- if the load and store are
          // from the exact same location, it is permitted.
          if (load.getMemref() == store.getMemref() &&
              load.getAffineMap() == store.getAffineMap() &&
              load.getIndices() == store.getIndices() &&
              DI.dominates((Operation *)load, (Operation *)store)) {
            // Example case where load does not dominate stores - if the load
            // was conditional. Or, store followed by load? Q. Can't we still
            // overlook the aliasing?
            stores_map[load] = store;
            continue;
          }
          //return failure();
        }
      }
      for (auto &&[_, store2] : stores) {
        if (store == store2)
          continue;
        if (mayAlias(store.getMemref(), store2.getMemref()) &&
            !storesProvablyDisjoint(store.getOperation(),
                                    store2.getOperation())) {
          return failure();
        }
      }
    }
    // Check that any other loads / stores do not alias with any linalg generics
    // We're going to need to upgrade the defn of mayAlias for subviews (aka
    // mayAlias(subview, x) -> mayAlias(operand(subview), x))

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> affineMaps;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<PromotedScalarLoad> promotedScalarLoads;

    // if (loop.getStep() != 1) {
    //     return failure();
    // }

    // Group A — triangular-bound support.
    BoundMaskInfo lbMaskInfo, ubMaskInfo;

    AffineMap ubMap = loop.getUpperBoundMap();
    SmallVector<Value> ubOperands(loop.getUpperBoundOperands());
    if (!ubMap || ubMap.getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Invalid upper bound map\n\n");
      return failure();
    }

    AffineMap lbMap = loop.getLowerBoundMap();
    SmallVector<Value> lbOperands(loop.getLowerBoundOperands());
    if (!lbMap || lbMap.getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Invalid lower bound map\n\n");
      return failure();
    }

    // Non-constant lower bound (e.g. `for k = i+1 to m`): substitute lb = 0
    // for iteration sizing and emit an in-body mask `index >= origLb(captures)`.
    if (!loop.hasConstantLowerBound()) {
      if (!allOperandsAreLoopInvariantWrt(lbOperands, loop)) {
        LLVM_DEBUG(llvm::dbgs() << "REJECTED: lb operands are not loop-invariant w.r.t. this loop\n\n");
        return failure();
      }
      lbMaskInfo.needed = true;
      lbMaskInfo.origMap = lbMap;
      lbMaskInfo.origOperands.assign(lbOperands.begin(), lbOperands.end());
      lbMap = AffineMap::get(/*dimCount=*/0, /*symCount=*/0,
                              rewriter.getAffineConstantExpr(0),
                              rewriter.getContext());
      lbOperands.clear();
      LLVM_DEBUG(llvm::dbgs() << "Captured non-constant lb for mask emission\n");
    }

    // Non-constant upper bound (e.g. `for j = 0 to i+1`): if any of the ub
    // operands is an IV of an enclosing affine.for, replace it with that
    // outer loop's (ub - 1) so the resulting size becomes outer-scope-
    // dominating. This is necessary for the outer loop to later wrap this
    // inner linalg.generic. Emit a body mask `index < origUb(captures)` so
    // the iterations we'd otherwise execute past the original ub are gated.
    if (!loop.hasConstantUpperBound() &&
        allOperandsAreLoopInvariantWrt(ubOperands, loop)) {
      // Check whether any operand is an IV of an enclosing affine.for.
      bool anyOuterIv = false;
      SmallVector<Value> maxUbOperands;
      maxUbOperands.reserve(ubOperands.size());
      for (Value op : ubOperands) {
        if (auto blockArg = dyn_cast<BlockArgument>(op)) {
          Operation *parentOp = blockArg.getOwner()->getParentOp();
          if (auto outerFor = dyn_cast<affine::AffineForOp>(parentOp)) {
            // Build (outerFor.ub - 1) at the same site this loop currently is.
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(loop);
            Value outerUb = rewriter.create<affine::AffineApplyOp>(
                loop.getLoc(), outerFor.getUpperBoundMap(),
                SmallVector<Value>(outerFor.getUpperBoundOperands()));
            Value c1 = rewriter.create<arith::ConstantIndexOp>(loop.getLoc(), 1);
            Value outerUbMinus1 = rewriter.create<arith::SubIOp>(
                loop.getLoc(), outerUb, c1);
            maxUbOperands.push_back(outerUbMinus1);
            anyOuterIv = true;
            continue;
          }
        }
        maxUbOperands.push_back(op);
      }
      if (anyOuterIv) {
        ubMaskInfo.needed = true;
        ubMaskInfo.origMap = ubMap;
        ubMaskInfo.origOperands.assign(ubOperands.begin(), ubOperands.end());
        // Use max-substituted operands for iteration-domain sizing.
        ubOperands = std::move(maxUbOperands);
        LLVM_DEBUG(llvm::dbgs() << "Captured non-constant ub for mask emission (max-substituted)\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Loop bounds:\n");
    LLVM_DEBUG(llvm::dbgs() << "  lbMap: " << lbMap << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  ubMap: " << ubMap << "\n");

    //auto ub = loop.getSingleUpperBound();
    //if (!ub)
    //  return failure();

    //auto lb = loop.getSingleLowerBound();
    //if (!lb)
    //  return failure();

    //if (!loop.hasConstantUpperBound()) {
    //  return failure();
    //}

    // Retrieve the step size
    int64_t step = loop.getStep();

    // Get the single result expressions
    AffineExpr ubExpr = ubMap.getResult(0);
    auto ubValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), ubMap, ubOperands);

    AffineExpr lbExpr = lbMap.getResult(0);
    auto lbValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), lbMap, lbOperands);

    //// Ensure the bounds are constant expressions
    //auto ubConst = ubExpr.dyn_cast<AffineConstantExpr>();
    //auto lbConst = lbExpr.dyn_cast<AffineConstantExpr>();
    //if (!ubConst || !lbConst)
    //  return failure();

    // Compute the loop size
    // int64_t loopSize = ubConst.getValue() - lbConst.getValue();
    auto loopSize = rewriter.create<SubIOp>(loop.getLoc(), ubValue, lbValue);

    // Value loopSize = rewriter.create<arith::ConstantIndexOp>(loop.getLoc(),
    // loop.getConstantUpperBound());//rewriter.create<arith::SubIOp>(loop.getLoc(),
    // *ub, *lb);

    LLVM_DEBUG(llvm::dbgs() << "\n--- Processing Linalg Generics ---\n");
    
    for (auto &&[conds, lg] : linalgGenerics) {

      LLVM_DEBUG(llvm::dbgs() << "Processing linalg.generic:\n" << lg << "\n");

      // This captures the indexing map attribute from the linalg.generic being
      // processed
      ArrayAttr indexingMapsAttr = lg.getIndexingMaps();

      int idx = 0;
      // Iterate over input arguments
      LLVM_DEBUG(llvm::dbgs() << "  Processing " << lg.getInputs().size() << " inputs\n");
      for (const Value input : lg.getInputs()) {
        // Is this needed?
        if (conds.size() != 0) {
          LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Input has conditions\n");
          return failure();
        }

        // TODO: Implement this
        // lgMap comes from offset of memref.subview,
        // lgOperands comes from operands of memref.subview

        const AffineMap lgMap0 =
            cast<AffineMapAttr>(indexingMapsAttr[idx]).getAffineMap();
        AffineMap lgMap = lgMap0;
        
        LLVM_DEBUG(llvm::dbgs() << "  Input " << idx << " indexing map: " << lgMap << "\n");
        SmallVector<Value> lgOperands;
        for (int i = 0; i < lgMap.getNumDims(); i++) {
          lgOperands.push_back(nullptr);
        }

        Value lgMemref = input;

        // At input, this contains, current input (i.e. probably a subview)
        // an  lgMap which is obtained from LG's indexing map for corresponding
        // input lgOperands contains current input (i.e probably a subview)

        // Gives output ...

        assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
        auto result = getLinalgArgMap(loop, lgMemref, lgMap, lgOperands);

        if (!result.succeeded())
          return failure();

        bool legal = true;

        // Takes input's/output's, affineMap of load/store (here lgMap ?),
        // induction variable corresponding to the loop
        // Memref corresponding the the memory accessed (in this case subview ?)
        // loopSize, lower and upper bounds
        // Get operands for load/store (here ?) to find dependent dim

        // Gives output newMemref which is a subviewOp,
        // newAffineMap which is the LG's indexing map corresponding this
        // inp/output

        // This takes load and store maps and then creates
        // affine.apply+subview+linalg.generic For this case: LG within ForOp -
        // Inputs should be : load map extracted from subviewOp
        // Returns LG with indexingMap and subview  with affine.apply - which
        // are correct

        // TODO: Or is it num dims?
        // size_t firstNDims = lgMap.getResults().size();
        size_t firstNDims = lgMap.getNumDims();
        check_reduction = false;
        
        LLVM_DEBUG(llvm::dbgs() << "  Calling remap_in_affine_dim for input " << idx << "\n");
        
        auto newMemref = remap_in_affine_dim(
            legal, rewriter, lgMap, lgMemref, loop.getInductionVar(), loopSize, lbValue,
            firstNDims, ValueRange(lgOperands), input, check_reduction);
        if (!legal) {
          LLVM_DEBUG(llvm::dbgs() << "  REJECTED: remap_in_affine_dim returned illegal for input\n");
          return failure();
        }

        auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims + 1);

        // TODO: need to mergre previous indexing maps and new affine maps
        affineMaps.push_back(newAffineMap);
        inputs.push_back(newMemref);
        idx++;
      }

      // Iterate over output arguments
      LLVM_DEBUG(llvm::dbgs() << "  Processing " << lg.getOutputs().size() << " outputs\n");
      for (const Value output : lg.getOutputs()) {
        // Is this needed?
        if (conds.size() != 0)
          return failure();

        const AffineMap lgMap0 =
            cast<AffineMapAttr>(indexingMapsAttr[idx]).getAffineMap();
        AffineMap lgMap = lgMap0;

        SmallVector<Value> lgOperands;
        for (int i = 0; i < lgMap.getNumDims(); i++) {
          lgOperands.push_back(nullptr);
        }
        Value lgMemref = output;

        auto result = getLinalgArgMap(loop, lgMemref, lgMap, lgOperands);

        if (!result.succeeded())
          return failure();

        bool legal = true;

        size_t firstNDims = lgMap.getNumDims();
        check_reduction = true;
        
        LLVM_DEBUG(llvm::dbgs() << "  Calling remap_in_affine_dim for output " << (idx - lg.getInputs().size()) << "\n");
        
        auto newMemref = remap_in_affine_dim(
            legal, rewriter, lgMap, lgMemref, loop.getInductionVar(), loopSize, lbValue,
            firstNDims, ValueRange(lgOperands), output, check_reduction);
        if (!legal) {
          LLVM_DEBUG(llvm::dbgs() << "  REJECTED: remap_in_affine_dim returned illegal for output\n");
          return failure();
        }

        auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims + 1);
        // TODO: need to merge previous indexing maps and new affine maps
        affineMaps.push_back(newAffineMap);
        outputs.push_back(newMemref);
      }
    }

    // current spec is going to be indexed off of the loop var in isolation
    LLVM_DEBUG(llvm::dbgs() << "\n--- Processing Loads ---\n");
    
    for (auto &&[conds, load] : loads) {
      LLVM_DEBUG(llvm::dbgs() << "Processing load: " << load << "\n");
      
      // Only support unconditional loads for the moment
      if (conds.size() != 0) {
        LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Load has conditions\n");
        return failure();
      }

      if (stores_map.find(load) != stores_map.end()) {
        // We have a store that represents this load.
        continue;
      }

      if (linalgGenerics.size() == 1) {
        // Darknet's GEMM uses the shape `for i; for k; a = A[i,k];
        // for j; C[i,j] += a * B[k,j]`. After the `j` loop has been raised,
        // the `k` wrapper contains one scalar affine.load plus one nested
        // linalg.generic. Promote that scalar load to a broadcast linalg input
        // instead of rejecting the mixed load + nested-generic body.
        auto nestedGeneric = linalgGenerics[0].second;
        if (load->getParentOp() != loop) {
          LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Load is not top-level in the wrapper loop\n");
          return failure();
        }
        for (Value output : nestedGeneric.getOutputs()) {
          if (load.getMemref() == output) {
            LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Promoted load aliases nested output by identity\n");
            return failure();
          }
        }
        DenseSet<Value> seen;
        if (!onlyFeedsNestedGenericThroughReadNone(
                load.getResult(), loop.getOperation(), nestedGeneric, seen)) {
          LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Load has non-generic/non-readnone users\n");
          return failure();
        }

        size_t firstNDims = 0;
        bool legal = true;
        bool promotedLoadReductionCheck = false;
        auto newMemref = remap_in_affine_dim(
            legal, rewriter, load.getAffineMap(), load.getMemref(),
            loop.getInductionVar(), loopSize, lbValue, firstNDims,
            load.getMapOperands(), load.getMemref(),
            promotedLoadReductionCheck);

        if (!legal)
          return failure();

        auto newMemrefType = cast<MemRefType>(newMemref.getType());
        if (nestedGeneric.getNumLoops() != 0) {
          SmallVector<Value> innerLoopSizes;
          if (failed(collectNestedGenericLoopSizes(nestedGeneric, rewriter,
                                                   innerLoopSizes)))
            return failure();

          SmallVector<Value> broadcastSizes;
          broadcastSizes.push_back(loopSize);
          broadcastSizes.append(innerLoopSizes.begin(), innerLoopSizes.end());

          SmallVector<int64_t> broadcastShape(
              broadcastSizes.size(), ShapedType::kDynamic);
          auto broadcastType = MemRefType::get(
              broadcastShape, newMemrefType.getElementType());
          auto broadcastMap = AffineMap::get(
              /*dimCount=*/broadcastSizes.size(), /*symbolCount=*/0,
              rewriter.getAffineDimExpr(0), rewriter.getContext());
          newMemref = rewriter.create<polygeist::SubmapOp>(
              load.getLoc(), broadcastType, newMemref, broadcastSizes,
              broadcastMap);
        }

        auto newAffineMap =
            rewriter.getMultiDimIdentityMap(nestedGeneric.getNumLoops() + 1);
        promotedScalarLoads.push_back(PromotedScalarLoad{newMemref,
                                                         newAffineMap});
        continue;
      }

      size_t firstNDims = 0;
      bool legal = true;

      check_reduction = false;
      auto newMemref = remap_in_affine_dim(
          legal, rewriter, load.getAffineMap(), load.getMemref(),
          loop.getInductionVar(), loopSize, lbValue, firstNDims, load.getMapOperands(),
          load.getMemref(), check_reduction);

      if (!legal)
        return failure();

      auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims + 1);
      affineMaps.push_back(newAffineMap);
      inputs.push_back(newMemref);
    }
    // TODO Push all of the inputs to the linalg generics (modifying maps as
    // needed)

    // SmallVector<Value> outputs;
    //  Store we may need to reindex into a splat potentially later, but for now
    //  we'll be lazy
    LLVM_DEBUG(llvm::dbgs() << "\n--- Processing Stores ---\n");
    
    for (auto &&[conds, store] : stores) {
      LLVM_DEBUG(llvm::dbgs() << "Processing store: " << store << "\n");
      
      // Only support unconditional loads for the moment
      if (conds.size() != 0) {
        LLVM_DEBUG(llvm::dbgs() << "  REJECTED: Store has conditions\n");
        return failure();
      }

      bool legal = true;

      size_t firstNDims = 0;

      check_reduction = true;
      auto newMemref = remap_in_affine_dim(
          legal, rewriter, store.getAffineMap(), store.getMemref(),
          loop.getInductionVar(), loopSize, lbValue, firstNDims, store.getMapOperands(),
          store.getMemref(), check_reduction);

      if (!legal) {
        return failure();
      }

      auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims + 1);
      affineMaps.push_back(newAffineMap);
      outputs.push_back(newMemref);
    }
    // TODO Push all of the outputs to the linalg generics

    if (!promotedScalarLoads.empty()) {
      SmallVector<Value> promotedInputs;
      SmallVector<AffineMap> promotedMaps;
      for (const PromotedScalarLoad &promoted : promotedScalarLoads) {
        promotedInputs.push_back(promoted.input);
        promotedMaps.push_back(promoted.indexingMap);
      }
      inputs.insert(inputs.begin(), promotedInputs.begin(),
                    promotedInputs.end());
      affineMaps.insert(affineMaps.begin(), promotedMaps.begin(),
                        promotedMaps.end());
    }

    SmallVector<utils::IteratorType> iteratorTypes;
    // TODO if linalg generic exists, make this iterator type prepend to the
    // existing iterators

    // TODO: Just store check is not sufficient, there has to be a check for
    // bool is_parallel = stores_map.size() == 0;
    //  TODO determine if linalg generic, whether to create parallel or
    //  reduction by looking at memory patterns of maps

    if (linalgGenerics.size() == 1) {
      // determine whether now we write to ourselves
    }

    iteratorTypes.push_back(check_reduction ? utils::IteratorType::reduction
                                            : utils::IteratorType::parallel);

    LLVM_DEBUG(llvm::dbgs() << "\n--- Creating linalg.generic ---\n");
    LLVM_DEBUG(llvm::dbgs() << "Iterator type for this loop: " 
               << (check_reduction ? "reduction" : "parallel") << "\n");

    if (linalgGenerics.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Extending iterator types from nested linalg.generic\n");
      for (auto attr : linalgGenerics[0].second.getIteratorTypesArray())
        iteratorTypes.push_back(attr);
    }

    LLVM_DEBUG(llvm::dbgs() << "Total iterator types: " << iteratorTypes.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Total inputs: " << inputs.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Total outputs: " << outputs.size() << "\n");

    StringAttr empty = StringAttr::get(loop.getContext());
    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loop.getLoc(), TypeRange(), inputs, outputs, affineMaps, iteratorTypes,
        empty, empty);

    // TODO if doing the linalg generic case, ignore a lot of the below and
    // instead of injecting the old body of the affine.for, move the inner
    // linalg.generic body and also add a new induction variable
    auto blk = &*loop.getRegion().begin();
    rewriter.setInsertionPointToStart(blk);

    // This index will replace the use of the affine index
    auto idx = rewriter.create<linalg::IndexOp>(loop.getLoc(),
                                                0);
    rewriter.replaceAllUsesWith(loop.getInductionVar(), idx);

    auto &body = genericOp.getRegion();
    body.takeBody(loop.getRegion());

    blk->eraseArguments(0, blk->getNumArguments());

    for (auto &&[conds, load] : loads) {
      if (stores_map.find(load) != stores_map.end()) {
        // We have a store that represents this load.
        continue;
      }
      auto arg = blk->addArgument(load.getType(), load.getLoc());
      rewriter.replaceOp(load, arg);
    }
    for (auto &&[conds, store] : stores) {
      auto arg =
          blk->addArgument(store.getValueToStore().getType(), store.getLoc());

      SmallVector<AffineLoadOp> inverted;
      for (auto &&[map_load, map_store] : stores_map) {
        if (map_store == store) {
          inverted.push_back(map_load);
        }
      }
      for (size_t i = 0; i < inverted.size(); i++) {
        stores_map.erase(inverted[i]);
        auto tmp = inverted[i];
        inverted[i] = nullptr;
        rewriter.replaceOp(tmp, arg);
      }
    }

    SmallVector<Value> toreturn;

    for (auto genPair : linalgGenerics) {
      auto genOp = genPair.second;
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genOp);
      auto &genBlock = genOp->getRegion(0).front();
      auto term = genBlock.getTerminator();
      mlir::IRMapping map;
      for (auto arg : genBlock.getArguments()) {
        auto arg2 = blk->addArgument(arg.getType(), arg.getLoc());
        map.map(arg, arg2);
      }
      for (auto &op : genBlock.without_terminator()) {
        Operation *cloned = rewriter.clone(op, map);
        // The outer loop being raised prepends one new iter dim (index 0).
        // Shift any cloned linalg.index dim numbers by 1 so they keep
        // referring to the inner iter they referenced before extension.
        if (auto idxOp = dyn_cast<linalg::IndexOp>(cloned)) {
          idxOp.setDim(idxOp.getDim() + 1);
        }
      }
      for (auto op : term->getOperands()) {
        toreturn.push_back(map.lookupOrDefault(op));
      }
      // llvm::errs() << genOp->getParentOfType<func::FuncOp>() << "\n";
      rewriter.eraseOp(genOp);
    }

    for (auto &&[conds, store] : stores) {
      toreturn.push_back(store.getValueToStore());
      rewriter.eraseOp(store);
    }

    rewriter.eraseOp(blk->getTerminator());
    rewriter.setInsertionPointToEnd(blk);

    // Group A — emit in-body mask when the loop had a non-constant lb and/or
    // ub. Gate each store-derived yield by the combined condition; fall back
    // to the corresponding output block arg when inactive.
    if (lbMaskInfo.needed || ubMaskInfo.needed) {
      Value idx = rewriter.create<linalg::IndexOp>(loop.getLoc(), /*dim=*/0);
      Value active;
      if (lbMaskInfo.needed) {
        Value lbVal = rewriter.create<affine::AffineApplyOp>(
            loop.getLoc(), lbMaskInfo.origMap, lbMaskInfo.origOperands);
        Value lbOk = rewriter.create<arith::CmpIOp>(
            loop.getLoc(), arith::CmpIPredicate::sge, idx, lbVal);
        active = lbOk;
      }
      if (ubMaskInfo.needed) {
        Value ubVal = rewriter.create<affine::AffineApplyOp>(
            loop.getLoc(), ubMaskInfo.origMap, ubMaskInfo.origOperands);
        Value ubOk = rewriter.create<arith::CmpIOp>(
            loop.getLoc(), arith::CmpIPredicate::slt, idx, ubVal);
        active = active
                     ? rewriter.create<arith::AndIOp>(loop.getLoc(), active, ubOk).getResult()
                     : ubOk;
      }

      // The last `stores.size()` entries of `toreturn` correspond to the
      // store-derived yields; the last `stores.size()` block args of `blk`
      // are the output operand block-args (representing the existing
      // accumulator/output value at this iteration).
      unsigned nArgs = blk->getNumArguments();
      unsigned nStores = stores.size();
      if (nStores > 0 && nArgs >= nStores && toreturn.size() >= nStores) {
        unsigned firstStoreArg = nArgs - nStores;
        unsigned firstStoreYield = toreturn.size() - nStores;
        for (unsigned i = 0; i < nStores; ++i) {
          Value oldAcc = blk->getArgument(firstStoreArg + i);
          Value gated = rewriter.create<arith::SelectOp>(
              loop.getLoc(), active, toreturn[firstStoreYield + i], oldAcc);
          toreturn[firstStoreYield + i] = gated;
        }
      }
    }

    rewriter.create<linalg::YieldOp>(loop.getLoc(), toreturn);

    auto func = loop->getParentOfType<func::FuncOp>();
    rewriter.eraseOp(loop);

    LLVM_DEBUG(llvm::dbgs() << "\n=== AffineForOpRaising SUCCESS ===\n");
    LLVM_DEBUG(llvm::dbgs() << "========================================\n\n");

    // return success!
    return success();
  }
};

struct AffineParallelFission : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::dbgs() << "\n=== AffineParallelFission ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing affine.parallel:\n" << parallelOp << "\n");

    auto module = parallelOp->getParentOfType<ModuleOp>();
    // Collect all top-level nested loops (affine.parallel or affine.for)
    SmallVector<Operation*> nestedLoops;
    Block *body = parallelOp.getBody();
    
    for (auto &op : body->without_terminator()) {
      if (isa<AffineParallelOp, AffineForOp>(op)) {
        nestedLoops.push_back(&op);
      } else {
        // Only allow pure nested loops - reject any other operations
        return failure();
      }
    }
    
    // Need at least 2 nested loops to perform fission
    if (nestedLoops.size() < 2) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Less than 2 nested loops (found " 
                 << nestedLoops.size() << ")\n\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found " << nestedLoops.size() << " nested loops to fission\n");
    
    // Convert reductions ArrayAttr to ArrayRef<AtomicRMWKind>
    SmallVector<arith::AtomicRMWKind> reductionKinds;
    for (auto attr : parallelOp.getReductions()) {
      auto enumAttr = cast<arith::AtomicRMWKindAttr>(attr);
      reductionKinds.push_back(enumAttr.getValue());
    }
    
    // Convert steps to ArrayRef<int64_t>
    SmallVector<int64_t> stepValues;
    for (auto step : parallelOp.getSteps()) {
      stepValues.push_back(step);
    }
    
    for (Operation *nestedLoop : nestedLoops) {
      
      // Create new parallel loops for each nested loop
      rewriter.setInsertionPoint(parallelOp);
    
      // Create a new outer parallel loop with same bounds
      auto newParallelOp = rewriter.create<AffineParallelOp>(
          parallelOp.getLoc(),
          parallelOp.getResultTypes(),
          reductionKinds,
          SmallVector<AffineMap>{parallelOp.getLowerBoundsMap()},
          parallelOp.getLowerBoundsOperands(),
          SmallVector<AffineMap>{parallelOp.getUpperBoundsMap()},
          parallelOp.getUpperBoundsOperands(),
          stepValues
      );
      
      // Move the nested loop into the new outer loop
      Block *newBody = newParallelOp.getBody();
      // Remove the existing terminator
      rewriter.eraseOp(newBody->getTerminator());
      
      // Set insertion point to the new body before cloning
      rewriter.setInsertionPointToEnd(newBody);
      
      // Clone the nested loop into the new body
      IRMapping mapping;
      // Map the induction variables (use getIVs() instead of getInductionVars())
      for (auto [oldIV, newIV] : llvm::zip(parallelOp.getIVs(),
                                           newParallelOp.getIVs())) {
        mapping.map(oldIV, newIV);
      }
      
      // Clone the operation (it will be automatically inserted at the current insertion point)
      rewriter.clone(*nestedLoop, mapping);
      
      // Ensure insertion point is at the end of the outer parallel loop's body
      rewriter.setInsertionPointToEnd(newBody);
      
      // Add the terminator back
      rewriter.create<AffineYieldOp>(parallelOp.getLoc());
    }
    
    // Remove the original parallel loop
    rewriter.eraseOp(parallelOp);
    
    return success();
  }

private:
  // Helper to check if an operation has no side effects that would 
  // prevent loop fission
  bool isMemoryOrControlFlowNeutral(Operation *op) const {
    // Allow constants, arithmetic, and other side-effect-free ops
    if (isa<arith::ConstantOp>(op)) return true;
    if (op->hasTrait<OpTrait::ConstantLike>()) return true;
    
    // Check if it's a pure operation (no memory effects)
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      effectInterface.getEffects(effects);
      return effects.empty();
    }
    
    // Conservative: if we can't prove it's safe, assume it's not
    return false;
  }
};

struct AffineParallelToFor : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== AffineParallelToFor ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing affine.parallel:\n" << parallelOp << "\n");
    
    // Skip if there are reductions - they need special handling
    if (!parallelOp.getReductions().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Has reductions\n\n");
      return failure();
    }
    
    // Skip if there are result types - parallel loops with returns need special handling
    if (!parallelOp.getResultTypes().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Has result types\n\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Converting parallel loop with " 
               << parallelOp.getIVs().size() << " induction variables\n");
    
    Location loc = parallelOp.getLoc();
    
    // Get the bounds and steps
    auto lowerBounds = parallelOp.getLowerBoundsMap();
    auto upperBounds = parallelOp.getUpperBoundsMap();
    auto steps = parallelOp.getSteps();
    auto lowerOperands = parallelOp.getLowerBoundsOperands();
    auto upperOperands = parallelOp.getUpperBoundsOperands();
    auto ivs = parallelOp.getIVs();
    
    // Start building nested for loops from outermost to innermost
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(parallelOp);
    
    // Create nested affine.for loops
    SmallVector<AffineForOp> forOps;
    SmallVector<Value> newIVs;
    
    for (unsigned i = 0; i < ivs.size(); ++i) {
      // Extract bounds for this dimension
      auto lbMap = lowerBounds.getSliceMap(i, 1);
      auto ubMap = upperBounds.getSliceMap(i, 1);
      int64_t step = steps[i];
      
      auto forOp = rewriter.create<AffineForOp>(
          loc,
          lowerOperands, lbMap,
          upperOperands, ubMap,
          step
      );
      // Mark this loop as known-parallel (came from affine.parallel). Group C
      // loop-distribution uses this as a precondition for safe fission.
      forOp->setAttr("polygeist.was_parallel", rewriter.getUnitAttr());

      forOps.push_back(forOp);
      newIVs.push_back(forOp.getInductionVar());
      
      // Set insertion point for next loop or body
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    
    // Move the body content from parallel to innermost for loop
    Block *parallelBody = parallelOp.getBody();
    Block *targetBody = forOps.empty() ? nullptr : forOps.back().getBody();
    
    if (!targetBody) {
      return failure();
    }
    
    // Create mapping for induction variables
    IRMapping mapping;
    for (auto [parallelIV, newIV] : llvm::zip(ivs, newIVs)) {
      mapping.map(parallelIV, newIV);
    }
    
    // Clone operations from parallel body to for body (excluding terminator)
    for (auto &op : parallelBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    
    // Remove the original parallel loop
    rewriter.eraseOp(parallelOp);
    
    LLVM_DEBUG(llvm::dbgs() << "=== AffineParallelToFor SUCCESS ===\n\n");
    
    return success();
  }
};

// namespace {
// struct RaiseAffineToLinalg
//     : public AffineRaiseToLinalgBase<RaiseAffineToLinalg> {

//   std::shared_ptr<const FrozenRewritePatternSet> patterns;

//   LogicalResult initialize(MLIRContext *context) override {
//     RewritePatternSet owningPatterns(context);
//     for (auto *dialect : context->getLoadedDialects())
//       dialect->getCanonicalizationPatterns(owningPatterns);
//     for (RegisteredOperationName op : context->getRegisteredOperations())
//       op.getCanonicalizationPatterns(owningPatterns, context);

//     owningPatterns.insert<AffineForOpRaising>(&getContext());

//     patterns = std::make_shared<FrozenRewritePatternSet>(
//         std::move(owningPatterns));
//     return success();
//   }
//   void runOnOperation() override {
//     GreedyRewriteConfig config;
//     (void)applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
//   }
// };
// } // namespace

namespace {
struct RaiseAffineToLinalgPipeline
    : public AffineRaiseToLinalgPipelineBase<RaiseAffineToLinalgPipeline> {
  void runOnOperation() override;
};
} // namespace

void RaiseAffineToLinalgPipeline::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\n****************************************\n");
  LLVM_DEBUG(llvm::dbgs() << "*** RaiseAffineToLinalgPipeline START ***\n");
  LLVM_DEBUG(llvm::dbgs() << "****************************************\n\n");

  // Create a nested pass manager to run the pipeline on functions
  OpPassManager pm(getOperation()->getName());
  
  // Create a nested pass manager for function operations
  OpPassManager &funcPM = pm.nest<func::FuncOp>();

  // Convert if/else scalar choices and matching stores to arith.select before
  // the affine-to-linalg raise. This handles control-flow-shaped expressions
  // that the linalg raiser can represent inside a generic body.
  funcPM.addPass(createFoldSCFIfPass());
  
  // Add affine-parallelize pass first (runs on func.func)
  funcPM.addPass(mlir::affine::createAffineParallelizePass());
  
  // Add our raise-affine-to-linalg pass second (also runs on func.func)
  funcPM.addPass(createRaiseAffineToLinalgPass());
  
  // Canonicalize after raise-to-linalg to eliminate submaps and other patterns
  //funcPM.addPass(createCanonicalizerPass());
  
  // Run the pipeline
  LLVM_DEBUG(llvm::dbgs() << "Running pipeline...\n");
  if (failed(runPipeline(pm, getOperation()))) {
    // Warn but don't fail the pass - convergence issues shouldn't kill output
    LLVM_DEBUG(llvm::dbgs() << "WARNING: Pipeline didn't converge completely\n");
    getOperation()->emitWarning("Pipeline didn't converge completely, but continuing anyway");
  }

  LLVM_DEBUG(llvm::dbgs() << "\n****************************************\n");
  LLVM_DEBUG(llvm::dbgs() << "*** RaiseAffineToLinalgPipeline END ***\n");
  LLVM_DEBUG(llvm::dbgs() << "****************************************\n\n");
}

namespace {
struct RaiseAffineToLinalg
    : public AffineRaiseToLinalgBase<RaiseAffineToLinalg> {
  void runOnOperation() override;
};
} // namespace

void RaiseAffineToLinalg::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\n****************************************\n");
  LLVM_DEBUG(llvm::dbgs() << "*** RaiseAffineToLinalg START ***\n");
  LLVM_DEBUG(llvm::dbgs() << "****************************************\n\n");

  GreedyRewriteConfig config;
  
  // Step 1: Apply fission pattern first
  {
    LLVM_DEBUG(llvm::dbgs() << "### Step 1: Applying AffineParallelFission ###\n");
    RewritePatternSet fissionPatterns(&getContext());
    fissionPatterns.insert<AffineParallelFission>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(fissionPatterns), config))) {
      LLVM_DEBUG(llvm::dbgs() << "WARNING: AffineParallelFission didn't converge\n");
      getOperation()->emitWarning("AffineParallelFission didn't converge, continuing anyway");
    }
    LLVM_DEBUG(llvm::dbgs() << "### Step 1 Complete ###\n\n");
  }
  
  // Step 2: Apply parallel-to-for conversion
  {
    LLVM_DEBUG(llvm::dbgs() << "### Step 2: Applying AffineParallelToFor ###\n");
    RewritePatternSet parallelToForPatterns(&getContext());
    parallelToForPatterns.insert<AffineParallelToFor>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(parallelToForPatterns), config))) {
      LLVM_DEBUG(llvm::dbgs() << "WARNING: AffineParallelToFor didn't converge\n");
      getOperation()->emitWarning("AffineParallelToFor didn't converge, continuing anyway");
    }
    LLVM_DEBUG(llvm::dbgs() << "### Step 2 Complete ###\n\n");
  }
  
  // Step 3: Apply distribution then raising patterns. Distribute runs at
  // higher benefit so loops whose bodies have mixed chunks (Group C/D)
  // get split into sibling homogeneous-body loops before being raised.
  {
    LLVM_DEBUG(llvm::dbgs() << "### Step 3: Applying Distribute + AffineForOpRaising ###\n");
    RewritePatternSet raisingPatterns(&getContext());
    raisingPatterns.add<PrivatizeScratchAllocaForLoop>(&getContext(), /*benefit=*/3);
    // NOT REGISTERED: PrivatizeRowScratchAllocaForLoop is implemented above
    // but is currently not wired into the pipeline because its rewrite
    // (memref.subview-based row selection) causes AffineForOpRaising to
    // stall on the strided dynamic-offset result type. See
    // notes/row_scratch_privatization_failures.md and
    // memory/row_scratch_privatization_attempt.md for the diagnosis and
    // the planned fix (switch to polygeist.submap-based row selection,
    // mirroring the rank-0 sibling). When that fix lands, uncomment the
    // line below to re-enable.
    // raisingPatterns.add<PrivatizeRowScratchAllocaForLoop>(&getContext(), /*benefit=*/3);
    raisingPatterns.add<HybridAffineForOpRaising>(&getContext(), /*benefit=*/2);
    raisingPatterns.add<DistributeAffineForOnLinalgGeneric>(&getContext(), /*benefit=*/2);
    raisingPatterns.add<AffineForOpRaising>(&getContext(), /*benefit=*/1);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(raisingPatterns), config))) {
      LLVM_DEBUG(llvm::dbgs() << "WARNING: Distribute+Raising didn't converge\n");
      getOperation()->emitWarning("Distribute+Raising didn't converge, continuing anyway");
    }
    LLVM_DEBUG(llvm::dbgs() << "### Step 3 Complete ###\n\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "****************************************\n");
  LLVM_DEBUG(llvm::dbgs() << "*** RaiseAffineToLinalg END ***\n");
  LLVM_DEBUG(llvm::dbgs() << "****************************************\n\n");
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalg>();
}

std::unique_ptr<Pass> createRaiseAffineToLinalgPipelinePass() {
  return std::make_unique<RaiseAffineToLinalgPipeline>();
}
} // namespace polygeist
} // namespace mlir
