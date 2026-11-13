//===- DelinearizeMemrefAccesses.cpp - Recover n-D views of flat buffers --===//
//
// C code often stores a multi-dimensional array in a flat buffer and
// linearizes subscripts by hand: `C[i*ldc + j]`, `A[(i*N + j)*M + k]`.
// cgeist turns such a buffer into a rank-1 memref whose affine maps multiply
// loop IVs by symbols (`d1 + d0 * s0`).  Those products are outside the
// Presburger fragment used by affine dependence analysis, so
// --affine-parallelize cannot prove the loops parallel, loop fission is
// refused, and raising to Linalg leaves the outer loops as scalar loops.
//
// This pass recovers the n-D structure.  For each rank-1 memref that is only
// accessed through affine.load / affine.store it
//
//  1. writes every subscript as a sum of terms  c * m * v,  where m is a
//     product of symbols (or 1) and v an index value (or 1);
//  2. derives one stride chain S_0 > S_1 > ... > S_{n-1} = 1 shared by all
//     accesses.  Strides are either symbol products in which consecutive
//     strides differ by exactly one symbol (C99-style VLAs, `i*N*M + j*M + k`)
//     or integer constants that divide each other (fixed-size arrays); the
//     extents are D_k = S_{k-1} / S_k;
//  3. splits every subscript into per-dimension indices idx_k with
//     subscript = sum_k idx_k * S_k (offsets such as `(i-1)*S + j + 1` are
//     assigned to the dimension whose stride they carry);
//  4. proves, from the iteration domain of the enclosing affine ops, the
//     conditions of enclosing scf.if ops (e.g. a BLAS argument check
//     `if (lda < MAX(1, m)) return;`) and dominating `__builtin_assume`
//     calls (e.g. the BLAS contract `__builtin_assume(lda >= MAX(1, m))`),
//     0 <= idx_k < D_k for every k >= 1 and idx_0 >= 0;
//  5. replaces the buffer by a memref.reinterpret_cast to a row-major
//     memref<D_0 x ... x D_{n-1}> with strides [S_0, ..., 1] and rewrites
//     every access to the n-D view.  D_0 is the smallest extent covering
//     every accessed row, computed from the loop bounds.
//
// Step 4 makes the rewrite exact: with every inner index in range, two
// accesses touch the same element iff their n-D indices are equal, so
// dependence analysis on the view is sound.  The rewrite is all-or-nothing
// per buffer: a buffer with any other use, or with an access that does not
// fit the chain or cannot be proved in range, is left untouched (a partially
// rewritten buffer would let affine analysis treat the view and the base as
// unrelated memrefs).  The view assumes the allocation holds every row it
// covers, i.e. that the buffer really is the dense row-major array its
// subscripts describe.  `__builtin_assume` calls are consumed: after the
// rewrite they are erased (dropping an assumption is always sound), so no
// call to the undefined builtin is left behind.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "delinearize-memref-accesses"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// A product of symbol values, kept sorted so that equal products compare
/// equal.  The empty product is 1.
using Monomial = SmallVector<Value, 2>;

static void sortMonomial(Monomial &m) {
  llvm::sort(m, [](Value a, Value b) {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  });
}

/// `sub` divides `m` (multiset inclusion); on success `rest` = m / sub.
static bool divides(const Monomial &sub, const Monomial &m, Monomial &rest) {
  rest.clear();
  SmallVector<bool> used(m.size(), false);
  for (Value s : sub) {
    bool found = false;
    for (unsigned i = 0; i < m.size(); ++i)
      if (!used[i] && m[i] == s) {
        used[i] = found = true;
        break;
      }
    if (!found)
      return false;
  }
  for (unsigned i = 0; i < m.size(); ++i)
    if (!used[i])
      rest.push_back(m[i]);
  return true;
}

/// One term c * syms * var of a subscript; `var` is null for none.
struct Term {
  int64_t coeff;
  Monomial syms;
  Value var;
};

/// A linear expression  sum c_v * v + cst  over index values.
struct LinearExpr {
  llvm::MapVector<Value, int64_t> coeffs;
  int64_t cst = 0;

  void add(Value v, int64_t c) {
    if (!v) {
      cst += c;
      return;
    }
    coeffs[v] += c;
    if (coeffs[v] == 0)
      coeffs.erase(v);
  }
  void add(const LinearExpr &o, int64_t scale = 1) {
    for (auto [v, c] : o.coeffs)
      add(v, c * scale);
    cst += o.cst * scale;
  }
};

static std::optional<int64_t> constantIndex(Value v) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  return std::nullopt;
}

/// Expand an affine expression into a list of terms.  Fails on mod/div and
/// on products of two index variables.
static LogicalResult expandTerms(AffineExpr e, ValueRange dims, ValueRange syms,
                                 SmallVectorImpl<Term> &out) {
  switch (e.getKind()) {
  case AffineExprKind::Constant:
    out.push_back({e.cast<AffineConstantExpr>().getValue(), {}, Value()});
    return success();
  case AffineExprKind::DimId:
    out.push_back({1, {}, dims[e.cast<AffineDimExpr>().getPosition()]});
    return success();
  case AffineExprKind::SymbolId: {
    Value s = syms[e.cast<AffineSymbolExpr>().getPosition()];
    if (auto c = constantIndex(s))
      out.push_back({*c, {}, Value()});
    else
      out.push_back({1, {s}, Value()});
    return success();
  }
  case AffineExprKind::Add: {
    auto bin = e.cast<AffineBinaryOpExpr>();
    if (failed(expandTerms(bin.getLHS(), dims, syms, out)))
      return failure();
    return expandTerms(bin.getRHS(), dims, syms, out);
  }
  case AffineExprKind::Mul: {
    auto bin = e.cast<AffineBinaryOpExpr>();
    SmallVector<Term> lhs, rhs;
    if (failed(expandTerms(bin.getLHS(), dims, syms, lhs)) ||
        failed(expandTerms(bin.getRHS(), dims, syms, rhs)))
      return failure();
    for (const Term &a : lhs)
      for (const Term &b : rhs) {
        if (a.var && b.var)
          return failure();
        Term t{a.coeff * b.coeff, a.syms, a.var ? a.var : b.var};
        t.syms.append(b.syms.begin(), b.syms.end());
        out.push_back(std::move(t));
      }
    return success();
  }
  default:
    return failure();
  }
}

/// Canonicalize: sort monomials and merge like terms.
static void normalizeTerms(SmallVectorImpl<Term> &terms) {
  SmallVector<Term> merged;
  for (Term &t : terms) {
    sortMonomial(t.syms);
    auto it = llvm::find_if(merged, [&](const Term &m) {
      return m.var == t.var && m.syms == t.syms;
    });
    if (it != merged.end())
      it->coeff += t.coeff;
    else
      merged.push_back(t);
  }
  terms.clear();
  for (Term &t : merged)
    if (t.coeff != 0)
      terms.push_back(std::move(t));
}

/// An affine.load or affine.store of the candidate buffer.
struct Access {
  Operation *op;
  AffineMap map;
  SmallVector<Value> dimOperands, symOperands;
  SmallVector<Term> terms;
  SmallVector<LinearExpr, 4> idx; // per dimension, filled by splitAccess
};

/// The recovered shape.  `strides[k]` is either a symbol product (symbolic
/// mode) or an integer (constant mode); `extents[k]` (k >= 1) is the single
/// symbol or integer S_{k-1}/S_k.
struct Shape {
  bool symbolic;
  SmallVector<Monomial> symStrides;
  SmallVector<int64_t> cstStrides;
  SmallVector<LinearExpr, 4> extents; // extents[0] unused
  unsigned rank() const {
    return symbolic ? symStrides.size() : cstStrides.size();
  }
};

static std::optional<Shape> deriveShape(ArrayRef<Access> accesses) {
  Shape shape;
  SmallVector<Monomial> mons;
  bool anySymbolic = false;
  for (const Access &a : accesses)
    for (const Term &t : a.terms)
      if (t.var && !t.syms.empty()) {
        anySymbolic = true;
        if (!llvm::is_contained(mons, t.syms))
          mons.push_back(t.syms);
      }
  if (anySymbolic) {
    shape.symbolic = true;
    mons.push_back({});
    llvm::sort(mons, [](const Monomial &a, const Monomial &b) {
      return a.size() > b.size();
    });
    shape.extents.push_back({});
    for (unsigned k = 1; k < mons.size(); ++k) {
      Monomial rest;
      // Consecutive strides must differ by exactly one symbol so that every
      // extent is a single value and the range proof stays linear.
      if (!divides(mons[k], mons[k - 1], rest) || rest.size() != 1)
        return std::nullopt;
      LinearExpr d;
      d.add(rest[0], 1);
      shape.extents.push_back(d);
    }
    shape.symStrides = mons;
    return shape;
  }
  SmallVector<int64_t> strides;
  for (const Access &a : accesses)
    for (const Term &t : a.terms)
      if (t.var && std::abs(t.coeff) > 1 &&
          !llvm::is_contained(strides, std::abs(t.coeff)))
        strides.push_back(std::abs(t.coeff));
  if (strides.empty())
    return std::nullopt;
  strides.push_back(1);
  llvm::sort(strides, std::greater<int64_t>());
  shape.symbolic = false;
  shape.extents.push_back({});
  for (unsigned k = 1; k < strides.size(); ++k) {
    if (strides[k - 1] % strides[k] != 0)
      return std::nullopt;
    LinearExpr d;
    d.cst = strides[k - 1] / strides[k];
    shape.extents.push_back(d);
  }
  shape.cstStrides = strides;
  return shape;
}

/// Assign every non-constant term of an access to a dimension.  Pure integer
/// offsets are returned in `cstOffset` for the caller to distribute.
static LogicalResult splitAccess(Access &a, const Shape &shape,
                                 int64_t &cstOffset) {
  unsigned n = shape.rank();
  a.idx.assign(n, LinearExpr());
  cstOffset = 0;
  for (const Term &t : a.terms) {
    if (!t.var && t.syms.empty()) {
      cstOffset += t.coeff;
      continue;
    }
    bool placed = false;
    for (unsigned k = 0; k < n && !placed; ++k) {
      if (shape.symbolic) {
        Monomial rest;
        if (!divides(shape.symStrides[k], t.syms, rest))
          continue;
        // The per-dimension index must stay linear: var * symbol is not.
        if (t.var ? !rest.empty() : rest.size() > 1)
          continue;
        a.idx[k].add(t.var ? t.var : (rest.empty() ? Value() : rest[0]),
                     t.coeff);
        placed = true;
      } else {
        int64_t s = shape.cstStrides[k];
        if (t.coeff % s != 0 || t.syms.size() > 1)
          continue;
        a.idx[k].add(t.var ? t.var : t.syms[0], t.coeff / s);
        placed = true;
      }
    }
    if (!placed)
      return failure();
  }
  return success();
}

/// Ways of distributing an integer offset over the dimensions: all of it in
/// the innermost index, or a mixed-radix split with floor or truncating
/// division (`(i-1)*100 + j` folds to `i*100 + j - 100`, which must become
/// row i-1, while `i*100 + j - 1` must stay column j-1).
static SmallVector<SmallVector<int64_t>>
offsetSplits(int64_t c, const Shape &shape) {
  unsigned n = shape.rank();
  SmallVector<SmallVector<int64_t>> splits;
  SmallVector<int64_t> inner(n, 0);
  inner[n - 1] = c;
  splits.push_back(inner);
  if (shape.symbolic || c == 0)
    return splits;
  for (bool floorDiv : {true, false}) {
    SmallVector<int64_t> s(n, 0);
    int64_t rem = c;
    for (unsigned k = 0; k + 1 < n; ++k) {
      int64_t stride = shape.cstStrides[k];
      int64_t q = rem / stride;
      if (floorDiv && rem % stride != 0 && rem < 0)
        --q;
      s[k] = q;
      rem -= q * stride;
    }
    s[n - 1] = rem;
    if (!llvm::is_contained(splits, s))
      splits.push_back(s);
  }
  return splits;
}

/// An integer value as an index-typed constraint column, or a constant: an
/// index value itself, or the result of an existing `arith.index_cast` of an
/// integer value (signed casts preserve order, so facts carry over).
static std::optional<std::pair<Value, int64_t>> factOperand(Value v) {
  if (auto c = constantIndex(v))
    return std::make_pair(Value(), *c);
  if (auto ci = v.getDefiningOp<arith::ConstantIntOp>())
    return std::make_pair(Value(), ci.value());
  if (v.getType().isIndex())
    return std::make_pair(v, int64_t(0));
  if (!v.getType().isSignlessInteger())
    return std::nullopt;
  for (Operation *user : v.getUsers())
    if (auto cast = dyn_cast<arith::IndexCastOp>(user))
      return std::make_pair(cast.getResult(), int64_t(0));
  return std::nullopt;
}

/// Facts `a - b + k >= 0` (or `== 0` when `isEq`) implied by an scf.if
/// condition that holds (`holds`) or fails at `op`.  Handles arith.cmpi on
/// integer/index values, `arith.andi` of conditions in the holding branch,
/// and max/min written as `arith.select (cmpi a, b), a, b` on the larger side.
struct Fact {
  Value a, b;           // null = 0
  int64_t aConst = 0, bConst = 0, k = 0;
  bool isEq = false;
};

static void conditionFacts(Value cond, bool holds, SmallVectorImpl<Fact> &out) {
  if (auto andi = cond.getDefiningOp<arith::AndIOp>()) {
    if (holds) {
      conditionFacts(andi.getLhs(), true, out);
      conditionFacts(andi.getRhs(), true, out);
    }
    return;
  }
  if (auto ori = cond.getDefiningOp<arith::OrIOp>()) {
    if (!holds) {    // !(a || b) == !a && !b
      conditionFacts(ori.getLhs(), false, out);
      conditionFacts(ori.getRhs(), false, out);
    }
    return;
  }
  if (auto ext = cond.getDefiningOp<arith::ExtUIOp>()) {
    conditionFacts(ext.getIn(), holds, out);
    return;
  }
  auto cmp = cond.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return;
  using P = arith::CmpIPredicate;
  P pred = cmp.getPredicate();
  if (!holds) {
    switch (pred) {
    case P::slt: pred = P::sge; break;
    case P::sle: pred = P::sgt; break;
    case P::sgt: pred = P::sle; break;
    case P::sge: pred = P::slt; break;
    default: return;
    }
  }
  Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
  // Normalise to  big >= small + k  (sge: k = 0, sgt: k = 1).
  Value big, small;
  int64_t k;
  switch (pred) {
  case P::sge: big = lhs; small = rhs; k = 0; break;
  case P::sgt: big = lhs; small = rhs; k = 1; break;
  case P::sle: big = rhs; small = lhs; k = 0; break;
  case P::slt: big = rhs; small = lhs; k = 1; break;
  case P::eq: {
    auto a = factOperand(lhs), b = factOperand(rhs);
    if (a && b)
      out.push_back({a->first, b->first, a->second, b->second, 0, true});
    return;
  }
  default:
    return;
  }
  // big >= max(x, y) implies big >= x and big >= y;  min(x, y) >= small
  // implies x >= small and y >= small.
  SmallVector<Value, 2> bigs{big}, smalls{small};
  auto selectArms = [](Value v, bool wantMax, SmallVectorImpl<Value> &arms) {
    auto sel = v.getDefiningOp<arith::SelectOp>();
    if (!sel)
      return false;
    auto c = sel.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!c)
      return false;
    Value t = sel.getTrueValue(), f = sel.getFalseValue();
    bool tIsLhsMax = (c.getPredicate() == P::sgt || c.getPredicate() == P::sge);
    bool tIsLhsMin = (c.getPredicate() == P::slt || c.getPredicate() == P::sle);
    bool isMax = ((tIsLhsMax && t == c.getLhs() && f == c.getRhs()) ||
                  (tIsLhsMin && t == c.getRhs() && f == c.getLhs()));
    bool isMin = ((tIsLhsMin && t == c.getLhs() && f == c.getRhs()) ||
                  (tIsLhsMax && t == c.getRhs() && f == c.getLhs()));
    if ((wantMax && !isMax) || (!wantMax && !isMin))
      return false;
    arms.assign({t, f});
    return true;
  };
  SmallVector<Value, 2> arms;
  if (selectArms(small, /*wantMax=*/true, arms))
    smalls = arms;
  arms.clear();
  if (selectArms(big, /*wantMax=*/false, arms))
    bigs = arms;
  for (Value bg : bigs)
    for (Value sm : smalls) {
      auto a = factOperand(bg), b = factOperand(sm);
      if (a && b)
        out.push_back({a->first, b->first, a->second, b->second, -k, false});
    }
}

/// `__builtin_assume(cond)` calls of a function (cgeist emits them as calls
/// to an external function taking the zero-extended condition).
static SmallVector<func::CallOp> assumeCalls(func::FuncOp f) {
  SmallVector<func::CallOp> out;
  f.walk([&](func::CallOp c) {
    if (c.getCallee() == "__builtin_assume" && c.getNumOperands() == 1)
      out.push_back(c);
  });
  return out;
}

/// Prove  lo <= e  and, if `hi` is given,  e <= hi - 1  over the iteration
/// domain of the affine ops enclosing `op`, using the conditions of enclosing
/// scf.if ops and of `assumes` that dominate `op` as additional facts.
static bool provedInRange(Operation *op, const LinearExpr &e, int64_t lo,
                          const LinearExpr *hi,
                          ArrayRef<func::CallOp> assumes = {},
                          DominanceInfo *dom = nullptr) {
  SmallVector<Operation *> enclosing;
  getEnclosingAffineOps(*op, &enclosing);
  FlatAffineValueConstraints cst;
  if (failed(getIndexSet(enclosing, &cst)))
    return false;
  auto ensureColumn = [&](Value v) {
    unsigned pos;
    if (!cst.findVar(v, &pos))
      cst.appendSymbolVar(ValueRange{v});
  };
  for (auto &kv : e.coeffs)
    ensureColumn(kv.first);
  if (hi)
    for (auto &kv : hi->coeffs)
      ensureColumn(kv.first);
  // Guard facts: op runs only in the branch of each enclosing scf.if whose
  // condition (or its negation, in the else branch) therefore holds.
  SmallVector<Fact> facts;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(p)) {
      bool inThen = ifOp.getThenRegion().isAncestor(op->getParentRegion());
      conditionFacts(ifOp.getCondition(), inThen, facts);
    }
    if (isa<func::FuncOp>(p))
      break;
  }
  for (func::CallOp c : assumes)
    if (dom && dom->properlyDominates(c.getOperation(), op))
      conditionFacts(c.getOperand(0), /*holds=*/true, facts);
  for (const Fact &f : facts) {
    if (f.a)
      ensureColumn(f.a);
    if (f.b)
      ensureColumn(f.b);
    SmallVector<int64_t> r(cst.getNumCols(), 0);
    unsigned pos;
    if (f.a && cst.findVar(f.a, &pos))
      r[pos] += 1;
    if (f.b && cst.findVar(f.b, &pos))
      r[pos] -= 1;
    r.back() += f.aConst - f.bConst + f.k;
    if (f.isEq)
      cst.addEquality(r);
    else
      cst.addInequality(r);
  }
  auto row = [&](const LinearExpr &x, int64_t scale,
                 SmallVectorImpl<int64_t> &r) {
    for (auto [v, c] : x.coeffs) {
      unsigned pos;
      bool found = cst.findVar(v, &pos);
      assert(found && "column was appended above");
      (void)found;
      r[pos] += scale * c;
    }
    r.back() += scale * x.cst;
  };
  // Violation of the lower bound:  lo - 1 - e >= 0.
  {
    FlatAffineValueConstraints c = cst;
    SmallVector<int64_t> r(c.getNumCols(), 0);
    row(e, -1, r);
    r.back() += lo - 1;
    c.addInequality(r);
    if (!c.isEmpty())
      return false;
  }
  if (!hi)
    return true;
  // Violation of the upper bound:  e - hi >= 0.
  FlatAffineValueConstraints c = cst;
  SmallVector<int64_t> r(c.getNumCols(), 0);
  row(e, 1, r);
  row(*hi, -1, r);
  c.addInequality(r);
  return c.isEmpty();
}

/// Turn an affine bound expression into a LinearExpr over its operands.
static std::optional<LinearExpr> linearize(AffineExpr e, ValueRange dims,
                                           ValueRange syms) {
  SmallVector<Term> terms;
  if (failed(expandTerms(e, dims, syms, terms)))
    return std::nullopt;
  normalizeTerms(terms);
  LinearExpr out;
  for (const Term &t : terms) {
    if (t.var && !t.syms.empty())
      return std::nullopt;
    if (t.syms.size() > 1)
      return std::nullopt;
    out.add(t.var ? t.var : (t.syms.empty() ? Value() : t.syms[0]), t.coeff);
  }
  return out;
}

/// An inclusive upper bound of `e` over the iteration domain, expressed in
/// loop-invariant values: each affine.for IV is replaced by its upper bound
/// minus one (positive coefficient) or its lower bound (negative), innermost
/// first, until no IV is left.  Any result of a min/max bound map is a valid
/// choice.  Fails on IVs of other loop kinds.
static std::optional<LinearExpr> upperBound(LinearExpr e) {
  for (unsigned guard = 0; guard < 64; ++guard) {
    Value iv;
    int64_t c = 0;
    for (auto [v, coeff] : e.coeffs) {
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        if (getForInductionVarOwner(v)) {
          iv = v;
          c = coeff;
          break;
        }
        if (!isa<func::FuncOp>(arg.getOwner()->getParentOp()))
          return std::nullopt; // IV of a loop we cannot bound
      }
    }
    if (!iv)
      return e;
    AffineForOp forOp = getForInductionVarOwner(iv);
    AffineBound b = c > 0 ? forOp.getUpperBound() : forOp.getLowerBound();
    AffineMap m = b.getMap();
    if (m.getNumResults() == 0)
      return std::nullopt;
    SmallVector<Value> operands(b.getOperands().begin(), b.getOperands().end());
    ValueRange dims = ValueRange(operands).take_front(m.getNumDims());
    ValueRange syms = ValueRange(operands).drop_front(m.getNumDims());
    auto bound = linearize(m.getResult(0), dims, syms);
    if (!bound)
      return std::nullopt;
    if (c > 0)
      bound->cst -= 1; // exclusive upper bound
    e.coeffs.erase(iv);
    e.add(*bound, c);
  }
  return std::nullopt;
}

/// Materialize sum c_v * v + cst as index arithmetic at the builder point.
static Value materialize(OpBuilder &b, Location loc, const LinearExpr &e) {
  if (e.cst == 0 && e.coeffs.size() == 1 && e.coeffs.front().second == 1)
    return e.coeffs.front().first;
  SmallVector<Value> operands;
  AffineExpr expr = getAffineConstantExpr(e.cst, b.getContext());
  for (auto [v, c] : e.coeffs) {
    expr = expr + getAffineSymbolExpr(operands.size(), b.getContext()) * c;
    operands.push_back(v);
  }
  auto map = AffineMap::get(0, operands.size(), expr);
  return b.create<AffineApplyOp>(loc, map, operands);
}

struct DelinearizeMemrefAccessesPass
    : public polygeist::DelinearizeMemrefAccessesBase<
          DelinearizeMemrefAccessesPass> {
  void runOnOperation() override {
    getOperation()->walk([&](func::FuncOp f) {
      if (f.isExternal())
        return;
      SmallVector<Value> candidates;
      Block &entry = f.getBody().front();
      for (BlockArgument arg : entry.getArguments())
        candidates.push_back(arg);
      for (Operation &op : entry)
        if (isa<memref::AllocOp, memref::AllocaOp>(op))
          candidates.push_back(op.getResult(0));
      DominanceInfo dom(f);
      assumes = assumeCalls(f);
      for (Value buf : candidates)
        (void)delinearize(f, buf, dom);
      // Assumptions are consumed (see the file comment).
      for (func::CallOp c : assumes) {
        Value v = c.getOperand(0);
        c.erase();
        while (Operation *d = v.getDefiningOp()) {
          if (!d->use_empty() || !isa<arith::ExtUIOp, arith::CmpIOp>(d))
            break;
          v = isa<arith::ExtUIOp>(d) ? d->getOperand(0) : Value();
          d->erase();
          if (!v)
            break;
        }
      }
      assumes.clear();
    });
    if (auto decl = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            getOperation(), StringAttr::get(&getContext(), "__builtin_assume")))
      if (decl.isExternal() &&
          SymbolTable::symbolKnownUseEmpty(decl, getOperation()))
        decl.erase();
  }

  SmallVector<func::CallOp> assumes;

  LogicalResult delinearize(func::FuncOp f, Value buf, DominanceInfo &dom) {
    auto type = dyn_cast<MemRefType>(buf.getType());
    if (!type || type.getRank() != 1 || !type.getLayout().isIdentity() ||
        buf.use_empty())
      return failure();

    // 1. Every use is an affine access of this buffer.
    SmallVector<Access, 4> accesses;
    for (Operation *user : buf.getUsers()) {
      Access a;
      a.op = user;
      if (auto ld = dyn_cast<AffineLoadOp>(user)) {
        a.map = ld.getAffineMap();
        auto ops = ld.getMapOperands();
        a.dimOperands.assign(ops.begin(), ops.begin() + a.map.getNumDims());
        a.symOperands.assign(ops.begin() + a.map.getNumDims(), ops.end());
      } else if (auto st = dyn_cast<AffineStoreOp>(user)) {
        if (st.getValueToStore() == buf)
          return failure();
        a.map = st.getAffineMap();
        auto ops = st.getMapOperands();
        a.dimOperands.assign(ops.begin(), ops.begin() + a.map.getNumDims());
        a.symOperands.assign(ops.begin() + a.map.getNumDims(), ops.end());
      } else {
        LLVM_DEBUG(llvm::dbgs() << "skip " << buf << ": non-affine use "
                                << *user << "\n");
        return failure();
      }
      if (a.map.getNumResults() != 1 ||
          failed(expandTerms(a.map.getResult(0), a.dimOperands, a.symOperands,
                             a.terms)))
        return failure();
      normalizeTerms(a.terms);
      accesses.push_back(std::move(a));
    }

    // 2. One stride chain for the whole buffer.
    std::optional<Shape> shape = deriveShape(accesses);
    if (!shape || shape->rank() < 2)
      return failure();
    unsigned n = shape->rank();

    // 3 + 4. Split each access and prove every index in range.
    bool multiDim = false;
    for (Access &a : accesses) {
      int64_t c;
      if (failed(splitAccess(a, *shape, c)))
        return failure();
      SmallVector<LinearExpr, 4> base = a.idx;
      bool proved = false;
      for (const SmallVector<int64_t> &split : offsetSplits(c, *shape)) {
        a.idx = base;
        for (unsigned k = 0; k < n; ++k)
          a.idx[k].cst += split[k];
        bool ok = provedInRange(a.op, a.idx[0], 0, nullptr, assumes, &dom);
        for (unsigned k = 1; ok && k < n; ++k)
          ok = provedInRange(a.op, a.idx[k], 0, &shape->extents[k], assumes, &dom);
        if (ok) {
          proved = true;
          break;
        }
      }
      if (!proved) {
        LLVM_DEBUG(llvm::dbgs() << "skip " << buf << ": cannot prove "
                                << *a.op << " in range\n");
        return failure();
      }
      unsigned varying = llvm::count_if(a.idx, [](const LinearExpr &x) {
        return llvm::any_of(x.coeffs, [](auto &kv) {
          return getForInductionVarOwner(kv.first) != nullptr;
        });
      });
      multiDim |= varying >= 2;
    }
    // Nothing to recover if no access walks two dimensions at once.
    if (!multiDim)
      return failure();

    // 5a. Leading extent: one past the largest row any access touches.
    SmallVector<LinearExpr, 4> rowBounds;
    for (Access &a : accesses) {
      auto ub = upperBound(a.idx[0]);
      if (!ub)
        return failure();
      ub->cst += 1;
      rowBounds.push_back(*ub);
    }

    // The view goes before the first entry-block op that holds an access and
    // must see every value it is built from.
    Block &entry = f.getBody().front();
    Operation *insertBefore = nullptr;
    for (Access &a : accesses) {
      Operation *top = entry.findAncestorOpInBlock(*a.op);
      if (!top)
        return failure();
      if (!insertBefore || top->isBeforeInBlock(insertBefore))
        insertBefore = top;
    }
    auto available = [&](Value v) {
      return dom.properlyDominates(v, insertBefore);
    };
    if (!available(buf))
      return failure();
    for (const LinearExpr &rb : rowBounds)
      for (auto &kv : rb.coeffs)
        if (!available(kv.first))
          return failure();
    for (unsigned k = 1; k < n; ++k)
      for (auto &kv : shape->extents[k].coeffs)
        if (!available(kv.first))
          return failure();

    // 5b. Build the view.
    OpBuilder b(insertBefore);
    Location loc = buf.getLoc();
    MLIRContext *ctx = f.getContext();
    SmallVector<Value> rowOperands;
    SmallVector<AffineExpr> rowExprs{getAffineConstantExpr(0, ctx)};
    for (const LinearExpr &rb : rowBounds) {
      AffineExpr e = getAffineConstantExpr(rb.cst, ctx);
      for (auto [v, c] : rb.coeffs) {
        auto it = llvm::find(rowOperands, v);
        unsigned pos = it - rowOperands.begin();
        if (it == rowOperands.end())
          rowOperands.push_back(v);
        e = e + getAffineSymbolExpr(pos, ctx) * c;
      }
      if (!llvm::is_contained(rowExprs, e))
        rowExprs.push_back(e);
    }
    auto rowMap = AffineMap::get(0, rowOperands.size(), rowExprs, ctx);
    Value rows = b.create<AffineMaxOp>(loc, rowMap, rowOperands);

    SmallVector<OpFoldResult> sizes(n), strides(n);
    SmallVector<int64_t> shapeDims(n, ShapedType::kDynamic);
    sizes[0] = rows;
    if (auto c = getConstantIntValue(rows))
      shapeDims[0] = *c;
    strides[n - 1] = b.getIndexAttr(1);
    Value runningStride; // dynamic stride below the current dimension
    int64_t staticStride = 1;
    for (int k = n - 1; k >= 1; --k) {
      const LinearExpr &d = shape->extents[k];
      if (d.coeffs.empty()) {
        sizes[k] = b.getIndexAttr(d.cst);
        shapeDims[k] = d.cst;
      } else {
        sizes[k] = materialize(b, loc, d);
      }
      // stride[k-1] = extent[k] * stride[k]
      if (!runningStride && d.coeffs.empty()) {
        staticStride *= d.cst;
        strides[k - 1] = b.getIndexAttr(staticStride);
        continue;
      }
      Value dk = d.coeffs.empty()
                     ? b.create<arith::ConstantIndexOp>(loc, d.cst).getResult()
                     : sizes[k].get<Value>();
      if (!runningStride && staticStride == 1) {
        runningStride = dk;
      } else {
        Value prev = runningStride
                         ? runningStride
                         : b.create<arith::ConstantIndexOp>(loc, staticStride)
                               .getResult();
        runningStride = b.create<arith::MulIOp>(loc, dk, prev);
      }
      strides[k - 1] = runningStride;
    }
    // Identity-layout result: the strides are exactly the row-major ones.
    auto viewType = MemRefType::get(shapeDims, type.getElementType(),
                                    MemRefLayoutAttrInterface(),
                                    type.getMemorySpace());
    Value view = b.create<memref::ReinterpretCastOp>(
        loc, viewType, buf, b.getIndexAttr(0), sizes, strides);

    // 5c. Rewrite every access to the view, reusing its operands.
    for (Access &a : accesses) {
      SmallVector<AffineExpr> results;
      for (const LinearExpr &x : a.idx) {
        AffineExpr e = getAffineConstantExpr(x.cst, ctx);
        for (auto [v, c] : x.coeffs) {
          AffineExpr ve;
          if (auto it = llvm::find(a.dimOperands, v); it != a.dimOperands.end())
            ve = getAffineDimExpr(it - a.dimOperands.begin(), ctx);
          else if (auto it = llvm::find(a.symOperands, v);
                   it != a.symOperands.end())
            ve = getAffineSymbolExpr(it - a.symOperands.begin(), ctx);
          else
            llvm_unreachable("index value must come from the access");
          e = e + ve * c;
        }
        results.push_back(e);
      }
      auto map = AffineMap::get(a.map.getNumDims(), a.map.getNumSymbols(),
                                results, ctx);
      SmallVector<Value> operands(a.dimOperands);
      operands.append(a.symOperands);
      OpBuilder ab(a.op);
      if (auto ld = dyn_cast<AffineLoadOp>(a.op)) {
        auto nl = ab.create<AffineLoadOp>(ld.getLoc(), view, map, operands);
        ld.getResult().replaceAllUsesWith(nl.getResult());
      } else {
        auto st = cast<AffineStoreOp>(a.op);
        ab.create<AffineStoreOp>(st.getLoc(), st.getValueToStore(), view, map,
                                 operands);
      }
      a.op->erase();
    }
    LLVM_DEBUG(llvm::dbgs() << "delinearized " << buf << " -> " << view
                            << "\n");
    return success();
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createDelinearizeMemrefAccessesPass() {
  return std::make_unique<DelinearizeMemrefAccessesPass>();
}
} // namespace polygeist
} // namespace mlir
