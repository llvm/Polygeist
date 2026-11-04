#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <deque>

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;

/// Carries the `hls.*` marks of a memory access (CGCall.cc's
/// hls.ap_compound_assign on a store) to the access that replaces it.
static void copyHLSMarks(Operation *from, Operation *to) {
  for (NamedAttribute a : from->getDiscardableAttrs())
    if (a.getName().getValue().starts_with("hls."))
      to->setAttr(a.getName(), a.getValue());
}
using namespace mlir::arith;
using namespace polygeist;
using namespace mlir::affine;

bool isReadOnly(Operation *op);

bool isValidSymbolInt(Value value, bool recur = true);
bool isValidSymbolInt(Operation *defOp, bool recur) {
  Attribute operandCst;
  if (matchPattern(defOp, m_Constant(&operandCst)))
    return true;

  if (recur) {
    if (isa<SelectOp, IndexCastOp, IndexCastUIOp, AddIOp, MulIOp, DivSIOp,
            DivUIOp, RemSIOp, RemUIOp, SubIOp, CmpIOp, TruncIOp, ExtUIOp,
            ExtSIOp>(defOp))
      if (llvm::all_of(defOp->getOperands(), [&](Value v) {
            bool b = isValidSymbolInt(v, recur);
            // if (!b)
            //	LLVM_DEBUG(llvm::dbgs() << "illegal isValidSymbolInt: "
            //<< value << " due to " << v << "\n");
            return b;
          }))
        return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      if (isValidSymbolInt(ifOp.getCondition(), recur)) {
        if (llvm::all_of(
                ifOp.thenBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }) &&
            llvm::all_of(
                ifOp.elseBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }))
          return true;
      }
    }
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(defOp)) {
      if (llvm::all_of(ifOp.getOperands(),
                       [&](Value o) { return isValidSymbolInt(o, recur); }))
        if (llvm::all_of(
                ifOp.getThenBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }) &&
            llvm::all_of(
                ifOp.getElseBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }))
          return true;
    }
  }
  return false;
}

// isValidSymbol, even if not index
bool isValidSymbolInt(Value value, bool recur) {
  // Check that the value is a top level value.
  if (affine::isTopLevelValue(value))
    return true;

  if (auto *defOp = value.getDefiningOp()) {
    if (isValidSymbolInt(defOp, recur))
      return true;
    return affine::isValidSymbol(value, affine::getAffineScope(defOp));
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Exact raising of integer casts and arithmetic
//===----------------------------------------------------------------------===//
//
// An affine expression computes with mathematical integers. An iN value is a
// different integer read signed (index_cast, extsi) or unsigned
// (index_castui, extui), and the arith ops on iN compute modulo 2^N. Raising
// replaces an index by an affine expression over the values it is computed
// from, which is exact only while every step keeps the value:
//  - a sign / zero extension of an iN, read with that signedness (an extsi
//    read unsigned only when its input is provably non-negative);
//  - a truncation index -> iN (index_cast / index_castui of an index) only
//    when the index value provably fits the iN as it is read;
//  - an add / sub / mul only when its exact result provably lies in the
//    range of its type as it is read (its operands read the same way; a
//    constant operand as whichever of the integers its bits stand for modulo
//    2^N keeps the result in that range: `k -= 5` of a uint8_t k is the i8
//    add of -5, read unsigned 251, and is k - 5 for k >= 5);
//  - a div / rem of any width, index included, only when it is the affine
//    floordiv / mod: a provably non-negative dividend (read as the map reads
//    it) and a positive divisor. C's / and % truncate toward zero, floordiv
//    and mod round down: (i - 5) / 2 is -2 for i = 0, (i - 5) floordiv 2 is
//    -3. This is rounding, not overflow, so the width threshold below does
//    not apply to it.
// The ranges come from the definitions (IntRangeOracle: constants,
// extensions, the bounds of affine.for / scf.for induction variables, sums
// and products of them). The raised expression reads each narrow input with
// the signedness the proof used, through an index_cast / index_castui of it.
//
// What is raised without a proof (the frontend's facts, Passes.h):
//  - an add / sub / mul of at least 32 bits that cgeist marked polygeist.nsw:
//    C signed arithmetic at least as wide as int, whose overflow is undefined
//    behaviour, so its result read signed is the exact result. The range
//    oracle reads such an op, read signed, as that exact result (the value
//    its raised expression computes). Unmarked arithmetic of any width
//    (unsigned int and unsigned long arithmetic, which wraps modulo 2^N:
//    `(i * 0x7FFFFFFFu) / 0x7FFFFFFFu` is 0 for i = 3; arithmetic on promoted
//    narrow operands; ops the canonicalizer rebuilt without the mark) needs
//    the proof; index arithmetic never does;
//  - an index_cast index -> iN marked polygeist.nsw (polygeist.nuw): the
//    index value fits iN read signed (unsigned); CanonicalizeFor marks the
//    counter of a loop it raised by assuming that the loop terminates;
//  - integer casts at least 32 bits wide (truncations index -> i32 / i64,
//    extensions, constants): the counters CanonicalizeFor creates for loops
//    whose counter is as wide as int are assumed not to wrap
//    (CanonicalizeFor, counterStaysInRange), and C reads every narrower
//    integer through a conversion to at least int.
// Both facts are about the value read signed (a cast: the value of its
// input). Read unsigned (through an extui / index_castui), a negative iN
// value is 2^N more: (size_t)(unsigned)(i - 3) is 4294967293 for i = 0, where
// the raised i - 3 is -3, and (unsigned long)(unsigned)i of an int counter
// i = -3 is 4294967293 as well. A marked op, a wide sign extension and a
// wide truncation of an index read unsigned narrower than 64 bits are
// therefore raised (and an index -> i32 -> index round trip read unsigned
// folded, roundTripKeepsValue) only when the range oracle proves that signed
// value non-negative (unsignedReadKeepsValue); a truncation the frontend
// marked polygeist.nuw needs no proof. A 64-bit value read unsigned needs none
// either: it reaches an index only through an index_castui, which keeps its
// bits, so its index value is its signed value (an unsigned 64-bit value of
// 2^63 or more has no index value of its own; see raisableRange), and a
// wide divui / remui reads its operands signed (operandsUnsigned).
// Narrower arithmetic and casts do occur in the IR and wrap in ordinary
// programs: the `+=` / `-=` of a narrow C integer (built in its own width, so
// that a counter update stays an iN add), _BitInt / ap_int arithmetic, narrow
// loop counters (`uint8_t k = 10; k -= i` is 266 - i for i > 10;
// `(uint16_t)i` of an int8_t i = -8 is 65528).
//
// -affine-cfg-assume-nowrap-width=N (N > 0) restores the width rule of the
// earlier versions, for comparison: casts and add / sub / mul at least N bits
// wide (narrower than 64) are raised without a proof, marked or not, and
// narrower ones only with it (32: the former default; 64: every integer
// narrower than index needs the proof); it also raises them read unsigned
// without the non-negativity proof, as those versions did.

static llvm::cl::opt<unsigned> AssumeNoWrapWidth(
    "affine-cfg-assume-nowrap-width", llvm::cl::init(0), llvm::cl::Hidden,
    llvm::cl::desc(
        "0 (default): AffineCFG raises an integer add / sub / mul to an affine "
        "expression without proving that it keeps the value only when the "
        "frontend marked it polygeist.nsw (and it is at least 32 bits wide), "
        "and integer casts at least 32 bits wide; N > 0: casts and arithmetic "
        "at least N bits wide, marked or not (32: the former default; 64: "
        "every integer narrower than index needs the proof)"));

/// The width from which casts are raised without a proof.
static unsigned castNoWrapWidth() {
  return AssumeNoWrapWidth ? (unsigned)AssumeNoWrapWidth : 32u;
}

namespace {
using WideInt = __int128;
/// A closed interval of mathematical integers.
struct IntRange {
  WideInt lo, hi;
};
} // namespace

/// Magnitude beyond which an IntRange is unbounded.
static constexpr WideInt kRangeInf = (WideInt)1 << 100;

static IntRange unboundedRange() { return {-kRangeInf, kRangeInf}; }

static IntRange clampRange(WideInt lo, WideInt hi) {
  return {std::max(lo, -kRangeInf), std::min(hi, kRangeInf)};
}

static bool rangeWithin(IntRange r, IntRange in) {
  return r.lo >= in.lo && r.hi <= in.hi;
}

/// Width of an integer or index type (index: 64), 0 for any other type.
static unsigned intOrIndexWidth(Type t) {
  if (t.isIndex())
    return 64;
  if (auto it = t.dyn_cast<IntegerType>())
    return it.getWidth();
  return 0;
}

/// Whether a cast to, a constant of, or a div / rem in `t` is raised only
/// with a proof that it keeps the value (see above); an add / sub / mul:
/// needsNoWrapProof(Operation *).
static bool needsNoWrapProof(Type t) {
  auto it = t.dyn_cast<IntegerType>();
  return it && it.getWidth() < 64 && it.getWidth() < castNoWrapWidth();
}

/// Whether the integer op `op` is raised only with a proof that it keeps the
/// value (see above): an add / sub / mul unless it is index arithmetic or
/// marked polygeist.nsw and at least 32 bits wide (with
/// -affine-cfg-assume-nowrap-width=N: narrower than min(N, 64) bits); any
/// other op as needsNoWrapProof of its type.
static bool needsNoWrapProof(Operation *op) {
  Type t = op->getResult(0).getType();
  if (!isa<AddIOp, SubIOp, MulIOp>(op))
    return needsNoWrapProof(t);
  auto it = t.dyn_cast<IntegerType>();
  if (!it)
    return false;
  if (AssumeNoWrapWidth)
    return it.getWidth() < 64 && it.getWidth() < AssumeNoWrapWidth;
  return !(it.getWidth() >= 32 &&
           op->hasAttr(mlir::polygeist::kNoSignedWrapAttrName));
}

/// Whether the truncation `cast` (index_cast / index_castui of an index to
/// iN) is marked as keeping the value when its result is read `asUnsigned`.
static bool truncationMarkedExact(Operation *cast, bool asUnsigned) {
  return cast && cast->hasAttr(asUnsigned
                                   ? mlir::polygeist::kNoUnsignedWrapAttrName
                                   : mlir::polygeist::kNoSignedWrapAttrName);
}

/// The values of a `width`-bit integer read unsigned / signed.
static IntRange typeRange(unsigned width, bool asUnsigned) {
  if (width == 0 || width > 96)
    return unboundedRange();
  if (asUnsigned)
    return {0, ((WideInt)1 << width) - 1};
  return {-((WideInt)1 << (width - 1)), ((WideInt)1 << (width - 1)) - 1};
}

/// The values an affine expression can stand for when it replaces a
/// `width`-bit integer read `asUnsigned`: typeRange within the range of index
/// (an unsigned 64-bit value of 2^63 or more has no index value).
static IntRange raisableRange(unsigned width, bool asUnsigned) {
  IntRange r = typeRange(width, asUnsigned);
  r.hi = std::min(r.hi, ((WideInt)1 << 63) - 1);
  return r;
}

static IntRange addRange(IntRange a, IntRange b) {
  return clampRange(a.lo + b.lo, a.hi + b.hi);
}
static IntRange subRange(IntRange a, IntRange b) {
  return clampRange(a.lo - b.hi, a.hi - b.lo);
}
static IntRange mulRange(IntRange a, IntRange b) {
  const WideInt big = (WideInt)1 << 62;
  for (WideInt x : {a.lo, a.hi, b.lo, b.hi})
    if (x > big || x < -big)
      return unboundedRange();
  WideInt c[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  return clampRange(*std::min_element(c, c + 4), *std::max_element(c, c + 4));
}
static IntRange unionRange(IntRange a, IntRange b) {
  return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
}
/// floor / ceil of a / b for b > 0.
static WideInt floorDivW(WideInt a, WideInt b) {
  WideInt q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}
static WideInt ceilDivW(WideInt a, WideInt b) {
  WideInt q = a / b;
  return (a % b != 0 && a > 0) ? q + 1 : q;
}

/// The exact result range of the integer add / sub / mul `op` for operand
/// ranges `a` and `b`.
static IntRange arithRange(Operation *op, IntRange a, IntRange b) {
  if (isa<AddIOp>(op))
    return addRange(a, b);
  if (isa<SubIOp>(op))
    return subRange(a, b);
  return mulRange(a, b);
}

/// The operand of the add / sub / mul `op` that is an integer constant (the
/// right one if both are), if any.
static std::optional<unsigned> constantOperand(Operation *op) {
  for (unsigned i : {1u, 0u})
    if (op->getOperand(i).getDefiningOp<ConstantIntOp>())
      return i;
  return std::nullopt;
}

/// An iN constant stands for every integer congruent to it modulo 2^N, and an
/// iN add / sub / mul gives the same bits with any of them (MLIR folds
/// `subi x, 5` into `addi x, -5`; read unsigned, an i8 -5 is 251). For the
/// add / sub / mul `op` on iN (not index) whose operand `cIdx` is a constant
/// and whose other operand lies in `other`: the first of the constant read as
/// `asUnsigned`, that minus 2^N, that plus 2^N, for which the exact result
/// lies in the range of the iN read as `asUnsigned` (the result then is the
/// iN result read that way), if any: the uint8_t k - 5 for k in [5, 99] is
/// k + (-5), not k + 251.
static std::optional<WideInt> constantRepresentative(Operation *op,
                                                     unsigned cIdx,
                                                     IntRange other,
                                                     bool asUnsigned) {
  Type t = op->getResult(0).getType();
  unsigned w = intOrIndexWidth(t);
  auto ci = op->getOperand(cIdx).getDefiningOp<ConstantIntOp>();
  if (!ci || t.isIndex() || w == 0 || w > 64)
    return std::nullopt;
  APInt c = ci.getValue().cast<IntegerAttr>().getValue();
  if (c.getBitWidth() != w)
    return std::nullopt;
  WideInt read =
      asUnsigned ? (WideInt)c.getZExtValue() : (WideInt)c.getSExtValue();
  WideInt m = (WideInt)1 << w;
  IntRange whole = raisableRange(w, asUnsigned);
  for (WideInt rep : {read, read - m, read + m}) {
    // The raised expression holds it as an int64_t affine constant.
    if (rep < -((WideInt)1 << 63) || rep >= ((WideInt)1 << 63))
      continue;
    IntRange cr{rep, rep};
    IntRange r =
        cIdx == 0 ? arithRange(op, cr, other) : arithRange(op, other, cr);
    if (rangeWithin(r, whole))
      return rep;
  }
  return std::nullopt;
}

namespace {
/// The ranges integer values can take, as far as their definitions show
/// them; memoised for the lifetime of one oracle (one query, the IR fixed).
class IntRangeOracle {
public:
  /// The values `v` can hold, read unsigned or signed (an index: signed).
  IntRange get(Value v, bool asUnsigned) { return get(v, asUnsigned, 0); }

private:
  DenseMap<std::pair<Value, unsigned>, IntRange> cache;

  IntRange get(Value v, bool asUnsigned, unsigned depth) {
    if (v.getType().isIndex())
      asUnsigned = false;
    auto key = std::make_pair(v, (unsigned)asUnsigned);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
    IntRange r = compute(v, asUnsigned, depth);
    cache[key] = r;
    return r;
  }

  IntRange exprRange(AffineExpr e, ArrayRef<IntRange> dims,
                     ArrayRef<IntRange> syms) {
    switch (e.getKind()) {
    case AffineExprKind::Constant: {
      WideInt c = e.cast<AffineConstantExpr>().getValue();
      return {c, c};
    }
    case AffineExprKind::DimId:
      return dims[e.cast<AffineDimExpr>().getPosition()];
    case AffineExprKind::SymbolId:
      return syms[e.cast<AffineSymbolExpr>().getPosition()];
    default:
      break;
    }
    auto bin = e.cast<AffineBinaryOpExpr>();
    IntRange l = exprRange(bin.getLHS(), dims, syms);
    IntRange r = exprRange(bin.getRHS(), dims, syms);
    switch (e.getKind()) {
    case AffineExprKind::Add:
      return addRange(l, r);
    case AffineExprKind::Mul:
      return mulRange(l, r);
    default:
      break;
    }
    // floordiv / ceildiv / mod by a positive constant.
    if (r.lo != r.hi || r.lo <= 0)
      return unboundedRange();
    WideInt c = r.lo;
    if (e.getKind() == AffineExprKind::FloorDiv)
      return clampRange(floorDivW(l.lo, c), floorDivW(l.hi, c));
    if (e.getKind() == AffineExprKind::CeilDiv)
      return clampRange(ceilDivW(l.lo, c), ceilDivW(l.hi, c));
    if (l.lo >= 0 && l.hi < c)
      return l;
    return {0, c - 1};
  }

  /// The range of the bound the map gives: the largest of its results (a
  /// lower bound) or the smallest (an upper bound).
  IntRange mapRange(AffineMap map, ValueRange operands, bool largest,
                    unsigned depth) {
    SmallVector<IntRange> dims, syms;
    for (auto en : llvm::enumerate(operands)) {
      IntRange r = get(en.value(), false, depth + 1);
      (en.index() < map.getNumDims() ? dims : syms).push_back(r);
    }
    std::optional<IntRange> res;
    for (AffineExpr e : map.getResults()) {
      IntRange r = exprRange(e, dims, syms);
      if (!res)
        res = r;
      else if (largest)
        res = IntRange{std::max(res->lo, r.lo), std::max(res->hi, r.hi)};
      else
        res = IntRange{std::min(res->lo, r.lo), std::min(res->hi, r.hi)};
    }
    return res ? *res : unboundedRange();
  }

  /// The range of an induction variable from the ranges of its loop's lower
  /// and upper bound; a loop that never runs gives its lower bound.
  static IntRange ivRange(IntRange lb, IntRange ub) {
    if (ub.hi - 1 < lb.lo)
      return {lb.lo, lb.lo};
    return {lb.lo, ub.hi - 1};
  }

  //===--------------------------------------------------------------------===//
  // Index arithmetic as a linear combination of values, bounded through the
  // bounds of the loops whose induction variables it contains: X = A - (iv -
  // lb) of a countdown whose upper bound is A + 1 lies in [lb, A], which the
  // intervals of A and iv alone do not show.
  //===--------------------------------------------------------------------===//

  /// One value of a linear combination. Index casts of the same iN with the
  /// same signedness are one atom (`base`, `kind`: 0 the value itself, 1 an
  /// index_cast of it, 2 an index_castui), `rep` one of them.
  struct LinearTerm {
    Value base;
    unsigned kind;
    Value rep;
    WideInt coeff;
  };
  static constexpr WideInt kLinearMax = (WideInt)1 << 62;

  struct LinearForm {
    SmallVector<LinearTerm, 4> terms;
    WideInt cst = 0;
    // A coefficient or the constant left [-kLinearMax, kLinearMax]: no bound.
    bool overflow = false;

    static bool big(WideInt x) { return x >= kLinearMax || x <= -kLinearMax; }
    WideInt scaled(WideInt x, WideInt scale) {
      if (big(x) || big(scale)) {
        overflow = true;
        return 0;
      }
      WideInt r = x * scale;
      overflow |= big(r);
      return r;
    }
    void add(const LinearForm &o, WideInt scale) {
      overflow |= o.overflow;
      cst += scaled(o.cst, scale);
      overflow |= big(cst);
      for (const LinearTerm &t : o.terms) {
        WideInt c = scaled(t.coeff, scale);
        bool merged = false;
        for (LinearTerm &u : terms)
          if (u.base == t.base && u.kind == t.kind) {
            u.coeff += c;
            overflow |= big(u.coeff);
            merged = true;
            break;
          }
        if (!merged)
          terms.push_back({t.base, t.kind, t.rep, c});
      }
    }
  };

  static LinearForm atom(Value v) {
    LinearForm f;
    Value base = v;
    unsigned kind = 0;
    if (auto c = v.getDefiningOp<IndexCastOp>()) {
      if (!c.getIn().getType().isIndex()) {
        base = c.getIn();
        kind = 1;
      }
    } else if (auto c = v.getDefiningOp<IndexCastUIOp>()) {
      if (!c.getIn().getType().isIndex()) {
        base = c.getIn();
        kind = 2;
      }
    }
    f.terms.push_back({base, kind, v, 1});
    return f;
  }

  /// The linear form of the affine expression `e` over `operands`, if it is
  /// linear (no floordiv / ceildiv / mod, products by constants only).
  std::optional<LinearForm> linearExpr(AffineExpr e, ValueRange operands,
                                       unsigned numDims, unsigned depth) {
    switch (e.getKind()) {
    case AffineExprKind::Constant: {
      LinearForm f;
      f.cst = e.cast<AffineConstantExpr>().getValue();
      return f;
    }
    case AffineExprKind::DimId:
      return linearize(operands[e.cast<AffineDimExpr>().getPosition()],
                       depth + 1);
    case AffineExprKind::SymbolId:
      return linearize(
          operands[numDims + e.cast<AffineSymbolExpr>().getPosition()],
          depth + 1);
    case AffineExprKind::Add: {
      auto bin = e.cast<AffineBinaryOpExpr>();
      auto l = linearExpr(bin.getLHS(), operands, numDims, depth);
      auto r = linearExpr(bin.getRHS(), operands, numDims, depth);
      if (!l || !r)
        return std::nullopt;
      l->add(*r, 1);
      return l;
    }
    case AffineExprKind::Mul: {
      auto bin = e.cast<AffineBinaryOpExpr>();
      auto c = bin.getRHS().dyn_cast<AffineConstantExpr>();
      if (!c || c.getValue() <= -kLinearMax || c.getValue() >= kLinearMax)
        return std::nullopt;
      auto l = linearExpr(bin.getLHS(), operands, numDims, depth);
      if (!l)
        return std::nullopt;
      LinearForm f;
      f.add(*l, c.getValue());
      return f;
    }
    default:
      return std::nullopt;
    }
  }

  /// `v` (index) as a linear combination of atoms.
  LinearForm linearize(Value v, unsigned depth) {
    APInt c;
    if (matchPattern(v, m_ConstantInt(&c)) && c.getBitWidth() <= 64) {
      LinearForm f;
      f.cst = c.getSExtValue();
      return f;
    }
    Operation *op = v.getDefiningOp();
    if (!op || !v.getType().isIndex() || depth > 8)
      return atom(v);
    if (isa<AddIOp, SubIOp>(op)) {
      LinearForm f = linearize(op->getOperand(0), depth + 1);
      f.add(linearize(op->getOperand(1), depth + 1), isa<AddIOp>(op) ? 1 : -1);
      return f;
    }
    if (isa<MulIOp>(op)) {
      for (unsigned i = 0; i < 2; ++i) {
        APInt m;
        if (matchPattern(op->getOperand(i), m_ConstantInt(&m)) &&
            m.getBitWidth() <= 64 && m.getSExtValue() > -kLinearMax &&
            m.getSExtValue() < kLinearMax) {
          LinearForm f;
          f.add(linearize(op->getOperand(1 - i), depth + 1), m.getSExtValue());
          return f;
        }
      }
      return atom(v);
    }
    if (auto ap = dyn_cast<affine::AffineApplyOp>(op))
      if (ap.getAffineMap().getNumResults() == 1)
        if (auto f = linearExpr(ap.getAffineMap().getResult(0),
                                ap.getMapOperands(),
                                ap.getAffineMap().getNumDims(), depth))
          return *f;
    return atom(v);
  }

  /// The loop whose induction variable `v` is, if any.
  static Operation *ivLoop(Value v) {
    auto ba = v.dyn_cast<BlockArgument>();
    if (!ba)
      return nullptr;
    Operation *owner = ba.getOwner()->getParentOp();
    if (auto af = dyn_cast_or_null<affine::AffineForOp>(owner))
      return af.getInductionVar() == v ? owner : nullptr;
    if (auto sf = dyn_cast_or_null<scf::ForOp>(owner))
      return sf.getInductionVar() == v ? owner : nullptr;
    return nullptr;
  }

  /// A lower bound of the induction variable of `loop` (`upper`: an upper
  /// bound, ub - 1) as a linear form: any result of a max / min bound map
  /// bounds it.
  std::optional<LinearForm> ivBound(Operation *loop, bool upper,
                                    unsigned depth) {
    if (auto af = dyn_cast<affine::AffineForOp>(loop)) {
      AffineMap map = upper ? af.getUpperBoundMap() : af.getLowerBoundMap();
      if (map.getNumResults() == 0)
        return std::nullopt;
      ValueRange ops =
          upper ? af.getUpperBoundOperands() : af.getLowerBoundOperands();
      auto f = linearExpr(map.getResult(0), ops, map.getNumDims(), depth);
      if (f && upper)
        f->cst -= 1;
      return f;
    }
    auto sf = cast<scf::ForOp>(loop);
    LinearForm f =
        linearize(upper ? sf.getUpperBound() : sf.getLowerBound(), depth + 1);
    if (upper)
      f.cst -= 1;
    return f;
  }

  /// The smallest (`upper`: largest) value of `f`: induction variables are
  /// replaced by their bounds, innermost loop first, then every remaining
  /// atom by the end of its range.
  WideInt linearExtreme(LinearForm f, bool upper, unsigned depth) {
    for (unsigned iter = 0; iter < 16 && !f.overflow; ++iter) {
      int pick = -1;
      Operation *pickLoop = nullptr;
      for (auto en : llvm::enumerate(f.terms)) {
        if (en.value().kind != 0 || en.value().coeff == 0)
          continue;
        Operation *loop = ivLoop(en.value().base);
        if (!loop)
          continue;
        if (!pickLoop || pickLoop->isProperAncestor(loop)) {
          pick = en.index();
          pickLoop = loop;
        }
      }
      if (pick < 0)
        break;
      LinearTerm t = f.terms[pick];
      // Maximising c * iv takes the largest iv for c > 0, the smallest for
      // c < 0; minimising the reverse.
      bool useUpper = upper ? t.coeff > 0 : t.coeff < 0;
      std::optional<LinearForm> b = ivBound(pickLoop, useUpper, depth);
      if (!b)
        break;
      f.terms.erase(f.terms.begin() + pick);
      f.add(*b, t.coeff);
    }
    if (f.overflow)
      return upper ? kRangeInf : -kRangeInf;
    WideInt r = f.cst;
    for (const LinearTerm &t : f.terms) {
      if (t.coeff == 0)
        continue;
      IntRange a = get(t.rep, false, depth + 1);
      WideInt end = (upper == (t.coeff > 0)) ? a.hi : a.lo;
      if (end >= kLinearMax || end <= -kLinearMax || t.coeff >= kLinearMax ||
          t.coeff <= -kLinearMax)
        return upper ? kRangeInf : -kRangeInf;
      r += t.coeff * end;
      if (r >= kRangeInf || r <= -kRangeInf)
        return upper ? kRangeInf : -kRangeInf;
    }
    return r;
  }

  /// The range of the index value `v` from its linear form, when that
  /// contains an induction variable (unbounded otherwise).
  IntRange linearRange(Value v, unsigned depth) {
    LinearForm f = linearize(v, 0);
    bool hasIV = llvm::any_of(f.terms, [](const LinearTerm &t) {
      return t.kind == 0 && t.coeff != 0 && ivLoop(t.base);
    });
    if (!hasIV)
      return unboundedRange();
    return clampRange(linearExtreme(f, false, depth),
                      linearExtreme(f, true, depth));
  }

  static IntRange intersectRange(IntRange a, IntRange b) {
    IntRange r{std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
    return r.lo <= r.hi ? r : a;
  }

  IntRange compute(Value v, bool asUnsigned, unsigned depth) {
    unsigned w = intOrIndexWidth(v.getType());
    if (w == 0)
      return unboundedRange();
    IntRange whole = typeRange(w, asUnsigned);
    if (depth > 12)
      return whole;
    auto fit = [&](IntRange r) { return rangeWithin(r, whole) ? r : whole; };

    APInt c;
    if (matchPattern(v, m_ConstantInt(&c))) {
      if (c.getBitWidth() > 64)
        return whole;
      WideInt x = asUnsigned ? (WideInt)c.getZExtValue()
                             : (WideInt)c.getSExtValue();
      return {x, x};
    }

    if (auto ba = v.dyn_cast<BlockArgument>()) {
      Operation *owner = ba.getOwner()->getParentOp();
      if (auto af = dyn_cast_or_null<affine::AffineForOp>(owner)) {
        if (af.getInductionVar() == v)
          return fit(ivRange(mapRange(af.getLowerBoundMap(),
                                      af.getLowerBoundOperands(),
                                      /*largest=*/true, depth),
                             mapRange(af.getUpperBoundMap(),
                                      af.getUpperBoundOperands(),
                                      /*largest=*/false, depth)));
      } else if (auto sf = dyn_cast_or_null<scf::ForOp>(owner)) {
        if (sf.getInductionVar() == v)
          return fit(ivRange(get(sf.getLowerBound(), false, depth + 1),
                             get(sf.getUpperBound(), false, depth + 1)));
      } else if (auto sp = dyn_cast_or_null<scf::ParallelOp>(owner)) {
        // Steps are positive (verifier), as for the loops above.
        for (auto [i, iv] : llvm::enumerate(sp.getInductionVars()))
          if (iv == v)
            return fit(
                ivRange(get(sp.getLowerBound()[i], false, depth + 1),
                        get(sp.getUpperBound()[i], false, depth + 1)));
      } else if (auto ap = dyn_cast_or_null<affine::AffineParallelOp>(owner)) {
        for (auto [i, iv] : llvm::enumerate(ap.getIVs()))
          if (iv == v)
            return fit(ivRange(
                mapRange(ap.getLowerBoundMap(i), ap.getLowerBoundsOperands(),
                         /*largest=*/true, depth),
                mapRange(ap.getUpperBoundMap(i), ap.getUpperBoundsOperands(),
                         /*largest=*/false, depth)));
      }
      return whole;
    }

    Operation *op = v.getDefiningOp();
    // CUDA's thread / block indices and dimensions (cgeist's threadIdx,
    // blockIdx, blockDim, gridDim) are unsigned and below 2^31 (the largest
    // dimension, gridDim.x, is at most 2^31 - 1).
    if (isa<gpu::ThreadIdOp, gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(
            op))
      return fit({0, ((WideInt)1 << 31) - 1});
    if (isa<IndexCastOp, IndexCastUIOp, ExtUIOp, ExtSIOp>(op)) {
      Value in = op->getOperand(0);
      if (in.getType().isIndex()) // a truncation index -> iN
        return fit(get(in, false, depth + 1));
      if (isa<ExtUIOp, IndexCastUIOp>(op))
        return fit(get(in, true, depth + 1));
      IntRange r = get(in, false, depth + 1);
      if (!asUnsigned || r.lo >= 0)
        return fit(r);
      if (r.hi < 0 && w <= 96)
        return fit({r.lo + ((WideInt)1 << w), r.hi + ((WideInt)1 << w)});
      return whole;
    }
    if (auto tr = dyn_cast<TruncIOp>(op)) {
      for (bool u : {asUnsigned, !asUnsigned}) {
        IntRange r = get(tr.getIn(), u, depth + 1);
        if (rangeWithin(r, whole))
          return r;
      }
      return whole;
    }
    if (isa<AddIOp, SubIOp, MulIOp>(op)) {
      IntRange a = get(op->getOperand(0), asUnsigned, depth + 1);
      IntRange b = get(op->getOperand(1), asUnsigned, depth + 1);
      IntRange r = arithRange(op, a, b);
      if (v.getType().isIndex())
        return fit(intersectRange(r, linearRange(v, depth)));
      // An op raised without a proof (polygeist.nsw), read signed, does not
      // wrap (see above): its range is that of its raised expression.
      if (!asUnsigned && !needsNoWrapProof(op))
        return r;
      // A constant operand as the representative that keeps the result in
      // range (constantRepresentative).
      if (!rangeWithin(r, whole))
        if (auto ci = constantOperand(op))
          if (auto rep = constantRepresentative(op, *ci, *ci == 0 ? b : a,
                                                asUnsigned)) {
            IntRange cr{*rep, *rep};
            r = *ci == 0 ? arithRange(op, cr, b) : arithRange(op, a, cr);
          }
      return fit(r);
    }
    if (isa<AndIOp>(op)) {
      IntRange a = get(op->getOperand(0), asUnsigned, depth + 1);
      IntRange b = get(op->getOperand(1), asUnsigned, depth + 1);
      if (a.lo >= 0 && b.lo >= 0)
        return {0, std::min(a.hi, b.hi)};
      if (a.lo >= 0)
        return {0, a.hi};
      if (b.lo >= 0)
        return {0, b.hi};
      return whole;
    }
    if (isa<RemUIOp, DivUIOp, ShRUIOp>(op)) {
      IntRange a = get(op->getOperand(0), true, depth + 1);
      IntRange b = get(op->getOperand(1), true, depth + 1);
      if (isa<ShRUIOp>(op)) {
        if (b.lo != b.hi || b.lo < 0 || b.lo >= (WideInt)w)
          return fit({0, a.hi});
        return fit({a.lo >> (int)b.lo, a.hi >> (int)b.lo});
      }
      // Division by zero is undefined: the divisor is at least 1.
      WideInt bLo = std::max<WideInt>(b.lo, 1);
      if (isa<RemUIOp>(op))
        return fit({0, std::min(a.hi, b.hi - 1)});
      return fit({a.lo / b.hi, a.hi / bLo});
    }
    if (isa<RemSIOp, DivSIOp>(op)) {
      IntRange a = get(op->getOperand(0), false, depth + 1);
      IntRange b = get(op->getOperand(1), false, depth + 1);
      if (b.lo <= 0)
        return whole;
      if (isa<RemSIOp>(op))
        return fit({a.lo >= 0 ? 0 : std::max(a.lo, -(b.hi - 1)),
                    a.hi <= 0 ? 0 : std::min(a.hi, b.hi - 1)});
      WideInt q[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
      return fit({*std::min_element(q, q + 4), *std::max_element(q, q + 4)});
    }
    if (auto sel = dyn_cast<SelectOp>(op))
      return unionRange(get(sel.getTrueValue(), asUnsigned, depth + 1),
                        get(sel.getFalseValue(), asUnsigned, depth + 1));
    if (isa<MaxSIOp, MinSIOp>(op) && !asUnsigned) {
      IntRange a = get(op->getOperand(0), false, depth + 1);
      IntRange b = get(op->getOperand(1), false, depth + 1);
      if (isa<MaxSIOp>(op))
        return {std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
      return {std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
    }
    if (auto ap = dyn_cast<affine::AffineApplyOp>(op))
      return fit(intersectRange(mapRange(ap.getAffineMap(), ap.getMapOperands(),
                                         /*largest=*/true, depth),
                                linearRange(v, depth)));
    if (auto mn = dyn_cast<affine::AffineMinOp>(op))
      return fit(mapRange(mn.getAffineMap(), mn.getMapOperands(),
                          /*largest=*/false, depth));
    if (auto mx = dyn_cast<affine::AffineMaxOp>(op))
      return fit(mapRange(mx.getAffineMap(), mx.getMapOperands(),
                          /*largest=*/true, depth));
    return whole;
  }
};
} // namespace

/// Whether an integer of type `readType` raised without a proof that it
/// keeps its value (a marked add / sub / mul, a cast at least 32 bits wide;
/// see above) keeps it when it is read `asUnsigned`, `signedValue` being what
/// it is exact as (the op's result read signed; a cast's input): read signed
/// or 64 bits wide, always; read unsigned and narrower, only when that value
/// is provably non-negative (a negative iN value read unsigned is 2^N more).
/// With -affine-cfg-assume-nowrap-width=N: always, as before.
static bool unsignedReadKeepsValue(IntRangeOracle &ranges, Type readType,
                                   Value signedValue, bool asUnsigned) {
  if (!asUnsigned || AssumeNoWrapWidth || intOrIndexWidth(readType) >= 64)
    return true;
  return ranges.get(signedValue, /*asUnsigned=*/false).lo >= 0;
}

/// Strips from `v`, read as `asUnsigned` (an index: as is), the casts that
/// keep its value (see above): returns the innermost value reached and how it
/// is read there. With `indexCastsOnly`, only index_cast / index_castui.
static std::pair<Value, bool> peelExactCasts(Value v, bool asUnsigned,
                                             bool indexCastsOnly = false) {
  if (v.getType().isIndex())
    asUnsigned = false;
  IntRangeOracle ranges;
  while (Operation *op = v.getDefiningOp()) {
    bool indexCast = isa<IndexCastOp, IndexCastUIOp>(op);
    if (!indexCast && (indexCastsOnly || !isa<ExtSIOp, ExtUIOp>(op)))
      break;
    Value in = op->getOperand(0);
    Type outTy = v.getType();
    bool inUnsigned = false;
    if (in.getType().isIndex()) {
      // A truncation index -> iN: exact while the index fits the iN as read
      // (provably, or as the frontend marked it). A wide one is assumed to
      // fit; read unsigned, only a non-negative index does.
      if (!truncationMarkedExact(op, asUnsigned) &&
          (needsNoWrapProof(outTy)
               ? !rangeWithin(ranges.get(in, false),
                              typeRange(intOrIndexWidth(outTy), asUnsigned))
               : !unsignedReadKeepsValue(ranges, outTy, in, asUnsigned)))
        break;
    } else if (isa<ExtUIOp, IndexCastUIOp>(op)) {
      inUnsigned = true;
    } else if (needsNoWrapProof(outTy)
                   ? asUnsigned && ranges.get(in, false).lo < 0
                   : !unsignedReadKeepsValue(ranges, outTy, in, asUnsigned)) {
      // A sign extension read unsigned: exact for a non-negative input only.
      break;
    }
    v = in;
    asUnsigned = !v.getType().isIndex() && inUnsigned;
  }
  return {v, asUnsigned};
}

/// How the operands of `op`, read as `asUnsigned`, are read: an add / sub /
/// mul that needs the proof reads them the same way, a narrow unsigned div /
/// rem unsigned, a signed one signed; an op raised without a proof (a marked
/// add / sub / mul, a wide div / rem) signed.
static bool operandsUnsigned(Operation *op, bool asUnsigned) {
  if (!needsNoWrapProof(op))
    return false;
  if (isa<DivUIOp, RemUIOp>(op))
    return true;
  if (isa<DivSIOp, RemSIOp>(op))
    return false;
  return asUnsigned;
}

static bool isRaisedIntOp(Operation *op) {
  return isa_and_nonnull<AddIOp, SubIOp, MulIOp, DivSIOp, DivUIOp, RemSIOp,
                         RemUIOp>(op);
}

/// Whether the integer op `op` (isRaisedIntOp), read as `asUnsigned`, equals
/// the affine expression of its operands read as operandsUnsigned (see
/// above). An index add / sub / mul: no proof; a marked one: none read
/// signed, read unsigned the proof that it is non-negative
/// (unsignedReadKeepsValue); a div / rem of any width needs the floordiv /
/// mod proof.
static bool exactArith(Operation *op, bool asUnsigned) {
  Type t = op->getResult(0).getType();
  bool divRem = isa<DivSIOp, DivUIOp, RemSIOp, RemUIOp>(op);
  IntRangeOracle ranges;
  if (!divRem && !needsNoWrapProof(op))
    return unsignedReadKeepsValue(ranges, t, op->getResult(0), asUnsigned);
  bool ou = operandsUnsigned(op, asUnsigned);
  IntRange a = ranges.get(op->getOperand(0), ou);
  IntRange b = ranges.get(op->getOperand(1), ou);
  IntRange whole = raisableRange(intOrIndexWidth(t), asUnsigned);
  if (!divRem) {
    if (auto ci = constantOperand(op))
      return constantRepresentative(op, *ci, *ci == 0 ? b : a, asUnsigned)
          .has_value();
    return rangeWithin(arithRange(op, a, b), whole);
  }
  // floordiv / mod is C's truncating / and % only for a non-negative
  // dividend and a positive divisor (an unsigned divisor of 0 is undefined
  // behaviour), at every width.
  if (a.lo < 0 || (ou ? b.hi < 1 : b.lo < 1))
    return false;
  if (!needsNoWrapProof(t))
    return true;
  WideInt bLo = std::max<WideInt>(b.lo, 1);
  IntRange r = isa<DivUIOp, DivSIOp>(op)
                   ? IntRange{a.lo / b.hi, a.hi / bLo}
                   : IntRange{0, std::min(a.hi, b.hi - 1)};
  return rangeWithin(r, whole);
}

/// The value of the integer constant `ci` read as `asUnsigned` (wide
/// constants as before: signed).
static int64_t constantRead(ConstantIntOp ci, bool asUnsigned) {
  if (!asUnsigned || !needsNoWrapProof(ci.getType()))
    return ci.value();
  return (int64_t)ci.getValue().cast<IntegerAttr>().getValue().getZExtValue();
}

static bool isValidIndexAs(Value val, bool asUnsigned);

struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands,
                        PatternRewriter &rewriter, DominanceInfo &DI);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value, 8> getOperands() {
    SmallVector<Value, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr renumberOneDim(Value v);

  /// Maps of Value to position in `affineMap`.
  DenseMap<Value, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value, 8> reorderedDims;
  SmallVector<Value, 8> concatenatedSymbols;

  AffineMap affineMap;
};

static bool isAffineForArg(Value val) {
  if (!val.isa<BlockArgument>())
    return false;
  Operation *parentOp = val.cast<BlockArgument>().getOwner()->getParentOp();
  return (
      isa_and_nonnull<affine::AffineForOp, affine::AffineParallelOp>(parentOp));
}

static bool legalCondition(Value en, bool dim = false) {
  if (en.getDefiningOp<affine::AffineApplyOp>())
    return true;

  if (!dim && !isValidSymbolInt(en, /*recur*/ false)) {
    if (isValidIndex(en) || isValidSymbolInt(en, /*recur*/ true)) {
      return true;
    }
  }

  // The index casts that keep the value (the normalizer strips at least
  // these), then an op with a constant operand it raises exactly.
  auto [stripped, strippedUnsigned] =
      peelExactCasts(en, /*asUnsigned=*/false, /*indexCastsOnly=*/true);
  en = stripped;

  if ((en.getDefiningOp<AddIOp>() || en.getDefiningOp<SubIOp>() ||
       en.getDefiningOp<MulIOp>() || en.getDefiningOp<RemUIOp>() ||
       en.getDefiningOp<RemSIOp>()) &&
      (en.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIntOp>() ||
       en.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIndexOp>()) &&
      exactArith(en.getDefiningOp(), strippedUnsigned))
    return true;
  // if (auto IC = dyn_cast_or_null<IndexCastOp>(en.getDefiningOp())) {
  //	if (!outer || legalCondition(IC.getOperand(), false)) return true;
  //}
  if (!dim)
    if (auto BA = dyn_cast<BlockArgument>(en)) {
      if (isa<affine::AffineForOp, affine::AffineParallelOp>(
              BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

/// The AffineNormalizer composes AffineApplyOp recursively. Its purpose is to
/// keep a correspondence between the mathematical `map` and the `operands` of
/// a given affine::AffineApplyOp. This correspondence is maintained by
/// iterating over the operands and forming an `auxiliaryMap` that can be
/// composed mathematically with `map`. To keep this correspondence in cases
/// where symbols are produced by affine.apply operations, we perform a local
/// rewrite of symbols as dims.
///
/// Rationale for locally rewriting symbols as dims:
/// ================================================
/// The mathematical composition of AffineMap must always concatenate symbols
/// because it does not have enough information to do otherwise. For example,
/// composing `(d0)[s0] -> (d0 + s0)` with itself must produce
/// `(d0)[s0, s1] -> (d0 + s0 + s1)`.
///
/// The result is only equivalent to `(d0)[s0] -> (d0 + 2 * s0)` when
/// applied to the same mlir::Value for both s0 and s1.
/// As a consequence mathematical composition of AffineMap always concatenates
/// symbols.
///
/// When AffineMaps are used in affine::AffineApplyOp however, they may specify
/// composition via symbols, which is ambiguous mathematically. This corner case
/// is handled by locally rewriting such symbols that come from
/// affine::AffineApplyOp into dims and composing through dims.
/// TODO: Composition via symbols comes at a significant code
/// complexity. Alternatively we should investigate whether we want to
/// explicitly disallow symbols coming from affine.apply and instead force the
/// user to compose symbols beforehand. The annoyances may be small (i.e. 1 or 2
/// extra API calls for such uses, which haven't popped up until now) and the
/// benefit potentially big: simpler and more maintainable code for a
/// non-trivial, recursive, procedure.
AffineApplyNormalizer::AffineApplyNormalizer(AffineMap map,
                                             ArrayRef<Value> operands,
                                             PatternRewriter &rewriter,
                                             DominanceInfo &DI) {
  assert(map.getNumInputs() == operands.size() &&
         "number of operands does not match the number of map inputs");

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nInput map: "));

  SmallVector<Value, 8> addedValues;

  llvm::SmallSet<unsigned, 1> symbolsToPromote;

  unsigned numDims = map.getNumDims();
  unsigned numSymbols = map.getNumSymbols();

  SmallVector<AffineExpr, 8> dimReplacements;
  SmallVector<AffineExpr, 8> symReplacements;

  SmallVector<SmallVectorImpl<Value> *> opsTodos;
  auto replaceOp = [&](Operation *oldOp, Operation *newOp) {
    for (auto [oldV, newV] :
         llvm::zip(oldOp->getResults(), newOp->getResults()))
      for (auto ops : opsTodos)
        for (auto &op : *ops)
          if (op == oldV)
            op = newV;
  };

  std::function<Value(Value, bool)> fix = [&](Value v,
                                              bool index) -> Value /*legal*/ {
    if (isValidSymbolInt(v, /*recur*/ false))
      return v;
    if (index && isAffineForArg(v))
      return v;
    auto *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    if (!op)
      llvm::errs() << v << "\n";
    assert(op);
    if (isa<ConstantOp>(op) || isa<ConstantIndexOp>(op))
      return v;
    if (!isReadOnly(op)) {
      return nullptr;
    }
    Operation *front = nullptr;
    SmallVector<Value> ops;
    opsTodos.push_back(&ops);
    std::function<void(Operation *)> getAllOps = [&](Operation *todo) {
      for (auto v : todo->getOperands()) {
        if (llvm::all_of(op->getRegions(), [&](Region &r) {
              return !r.isAncestor(v.getParentRegion());
            }))
          ops.push_back(v);
      }
      for (auto &r : todo->getRegions()) {
        for (auto &b : r.getBlocks())
          for (auto &o2 : b.without_terminator())
            getAllOps(&o2);
      }
    };
    getAllOps(op);
    for (auto o : ops) {
      Operation *next;
      if (auto *op = o.getDefiningOp()) {
        if (Value nv = fix(o, index)) {
          op = nv.getDefiningOp();
        } else {
          return nullptr;
        }
        next = op->getNextNode();
      } else {
        auto BA = o.cast<BlockArgument>();
        if (index && isAffineForArg(BA)) {
        } else if (!isValidSymbolInt(o, /*recur*/ false)) {
          return nullptr;
        }
        next = &BA.getOwner()->front();
      }
      if (front == nullptr)
        front = next;
      else if (DI.dominates(front, next))
        front = next;
    }
    opsTodos.pop_back();
    if (!front)
      op->dump();
    assert(front);
    PatternRewriter::InsertionGuard B(rewriter);
    rewriter.setInsertionPoint(front);
    auto cloned = rewriter.clone(*op);
    replaceOp(op, cloned);
    rewriter.replaceOp(op, cloned->getResults());
    return cloned->getResult(0);
  };
  auto renumberOneSymbol = [&](Value v) {
    for (auto i : llvm::enumerate(addedValues)) {
      if (i.value() == v)
        return getAffineSymbolExpr(i.index(), map.getContext());
    }
    auto expr = getAffineSymbolExpr(addedValues.size(), map.getContext());
    addedValues.push_back(v);
    return expr;
  };

  // The narrow iN `v` read as `asUnsigned`, as an index map input: an
  // index_cast / index_castui of it that dominates the rewritten op (a valid
  // symbol if `v` is one), else a new one right after its definition.
  auto indexInput = [&](Value v, bool asUnsigned) -> Value {
    if (v.getType().isIndex())
      return v;
    Operation *ip = nullptr;
    if (rewriter.getInsertionBlock() &&
        rewriter.getInsertionPoint() != rewriter.getInsertionBlock()->end())
      ip = &*rewriter.getInsertionPoint();
    bool symbol = isValidSymbolInt(v, /*recur*/ false);
    for (Operation *u : v.getUsers())
      if ((asUnsigned ? isa<IndexCastUIOp>(u) : isa<IndexCastOp>(u)) &&
          u->getResult(0).getType().isIndex() && ip && DI.dominates(u, ip) &&
          (!symbol || isValidSymbolInt(u->getResult(0), /*recur*/ false)))
        return u->getResult(0);
    PatternRewriter::InsertionGuard guard(rewriter);
    if (Operation *def = v.getDefiningOp())
      rewriter.setInsertionPointAfter(def);
    else
      rewriter.setInsertionPointToStart(v.cast<BlockArgument>().getOwner());
    if (asUnsigned)
      return rewriter.create<IndexCastUIOp>(v.getLoc(), rewriter.getIndexType(),
                                            v);
    return rewriter.create<IndexCastOp>(v.getLoc(), rewriter.getIndexType(), v);
  };

  // 2. Compose affine::AffineApplyOps and dispatch dims or symbols.
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    auto t = operands[i];
    // The casts that keep the value (peelExactCasts); `decast` is read as
    // `decastUnsigned`.
    auto [decast, decastUnsigned] = peelExactCasts(t, /*asUnsigned=*/false);
    bool tUnsigned = false;

    if (!isValidSymbolInt(t, /*recur*/ false)) {
      t = decast;
      tUnsigned = decastUnsigned;
    }

    // An integer op becomes the affine expression of its operands only when
    // that keeps its value (exactArith), its operands read as
    // operandsUnsigned says.
    auto exact = [](Value v, bool asUnsigned) {
      Operation *op = v.getDefiningOp();
      return !isRaisedIntOp(op) || exactArith(op, asUnsigned);
    };
    bool tOperandsUnsigned = isRaisedIntOp(t.getDefiningOp()) &&
                             operandsUnsigned(t.getDefiningOp(), tUnsigned);
    auto validIndex = [&](Value v) {
      return isValidIndexAs(v, tOperandsUnsigned);
    };

    // Only promote one at a time, lest we end up with two dimensions
    // multiplying each other.

    if (((!isValidSymbolInt(t, /*recur*/ false) && exact(t, tUnsigned) &&
          (t.getDefiningOp<AddIOp>() || t.getDefiningOp<SubIOp>() ||
           (t.getDefiningOp<MulIOp>() &&
            ((validIndex(t.getDefiningOp()->getOperand(0)) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(1))) ||
             (validIndex(t.getDefiningOp()->getOperand(1)) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(0)))) &&
            !(fix(t.getDefiningOp()->getOperand(0), false) &&
              fix(t.getDefiningOp()->getOperand(1), false))

                ) ||
           ((t.getDefiningOp<DivUIOp>() || t.getDefiningOp<DivSIOp>()) &&
            (validIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1))) &&
            (!(fix(t.getDefiningOp()->getOperand(0), false) &&
               fix(t.getDefiningOp()->getOperand(1), false)))) ||
           (t.getDefiningOp<DivSIOp>() &&
            (validIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           (t.getDefiningOp<RemUIOp>() &&
            (validIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           (t.getDefiningOp<RemSIOp>() &&
            (validIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           t.getDefiningOp<ConstantIntOp>() ||
           t.getDefiningOp<ConstantIndexOp>())) ||
         (exact(decast, decastUnsigned) &&
          (decast.getDefiningOp<AddIOp>() || decast.getDefiningOp<SubIOp>() ||
           decast.getDefiningOp<MulIOp>() || decast.getDefiningOp<RemUIOp>() ||
           decast.getDefiningOp<RemSIOp>()) &&
          (decast.getDefiningOp()
               ->getOperand(1)
               .getDefiningOp<ConstantIntOp>() ||
           decast.getDefiningOp()
               ->getOperand(1)
               .getDefiningOp<ConstantIndexOp>())))) {
      t = decast;
      tUnsigned = decastUnsigned;
      LLVM_DEBUG(llvm::dbgs() << " Replacing: " << t << "\n");

      // The op's operands as map inputs: those of an op that needs the proof
      // (needsNoWrapProof: a narrow op, an unmarked add / sub / mul) through
      // an index cast with the signedness its proof read them with, its
      // constants read that way; any other op's as before.
      bool narrowOp = isRaisedIntOp(t.getDefiningOp()) &&
                      needsNoWrapProof(t.getDefiningOp());
      bool inputsUnsigned =
          narrowOp && operandsUnsigned(t.getDefiningOp(), tUnsigned);
      auto input = [&](Value v) -> Value {
        return narrowOp ? indexInput(v, inputsUnsigned) : v;
      };
      auto cstValue = [&](ConstantIntOp ci) -> int64_t {
        return narrowOp ? constantRead(ci, inputsUnsigned) : ci.value();
      };

      AffineMap affineApplyMap;
      SmallVector<Value, 8> affineApplyOperands;

      // llvm::dbgs() << "\nop to start: " << t << "\n";

      // A narrow add / sub / mul with a constant operand: the constant is the
      // representative its proof used (constantRepresentative).
      std::optional<unsigned> narrowCst;
      std::optional<WideInt> narrowRep;
      if (narrowOp && isa<AddIOp, SubIOp, MulIOp>(t.getDefiningOp()) &&
          (narrowCst = constantOperand(t.getDefiningOp()))) {
        IntRangeOracle ranges;
        Operation *op = t.getDefiningOp();
        narrowRep = constantRepresentative(
            op, *narrowCst,
            ranges.get(op->getOperand(1 - *narrowCst), inputsUnsigned),
            tUnsigned);
      }

      if (narrowRep) {
        Operation *op = t.getDefiningOp();
        AffineExpr c =
            getAffineConstantExpr((int64_t)*narrowRep, op->getContext());
        AffineExpr x = getAffineSymbolExpr(0, op->getContext());
        AffineExpr lhs = *narrowCst == 0 ? c : x;
        AffineExpr rhs = *narrowCst == 0 ? x : c;
        affineApplyMap = AffineMap::get(0, 1,
                                        isa<AddIOp>(op)   ? lhs + rhs
                                        : isa<SubIOp>(op) ? lhs - rhs
                                                          : lhs * rhs);
        affineApplyOperands.push_back(input(op->getOperand(1 - *narrowCst)));
      } else if (auto op = t.getDefiningOp<AddIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) +
                               getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(input(op.getLhs()));
        affineApplyOperands.push_back(input(op.getRhs()));
      } else if (auto op = t.getDefiningOp<SubIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) -
                               getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(input(op.getLhs()));
        affineApplyOperands.push_back(input(op.getRhs()));
      } else if (auto op = t.getDefiningOp<MulIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) * cstValue(ci));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) * ci.value());
          affineApplyOperands.push_back(input(op.getLhs()));
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) *
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(input(op.getLhs()));
          affineApplyOperands.push_back(input(op.getRhs()));
        }
      } else if (auto op = t.getDefiningOp<DivSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(cstValue(ci)));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else {
          affineApplyMap = AffineMap::get(
              0, 2,
              getAffineSymbolExpr(0, op.getContext())
                  .floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(input(op.getLhs()));
          affineApplyOperands.push_back(input(op.getRhs()));
        }
      } else if (auto op = t.getDefiningOp<DivUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(cstValue(ci)));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else {
          affineApplyMap = AffineMap::get(
              0, 2,
              getAffineSymbolExpr(0, op.getContext())
                  .floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(input(op.getLhs()));
          affineApplyOperands.push_back(input(op.getRhs()));
        }
      } else if (auto op = t.getDefiningOp<RemSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % cstValue(ci));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(input(op.getLhs()));
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) %
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(input(op.getLhs()));
          affineApplyOperands.push_back(input(op.getRhs()));
        }
      } else if (auto op = t.getDefiningOp<RemUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % cstValue(ci));
          affineApplyOperands.push_back(input(op.getLhs()));
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(input(op.getLhs()));
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) %
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(input(op.getLhs()));
          affineApplyOperands.push_back(input(op.getRhs()));
        }
      } else if (auto op = t.getDefiningOp<ConstantIntOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0,
            getAffineConstantExpr(constantRead(op, tUnsigned),
                                  op.getContext()));
      } else if (auto op = t.getDefiningOp<ConstantIndexOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else {
        llvm_unreachable("");
      }

      SmallVector<AffineExpr, 0> dimRemapping;
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols);
      for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
        symRemapping[idx] = renumberOneSymbol(affineApplyOperands[idx]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(affineApplyMap.print(
          llvm::dbgs() << "\nRenumber into current normalizer: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));

    } else if (isAffineForArg(t)) {
      if (i >= numDims)
        symReplacements.push_back(renumberOneDim(t));
      else
        dimReplacements.push_back(renumberOneDim(t));
    } else if (t.getDefiningOp<affine::AffineApplyOp>()) {
      auto affineApply = t.getDefiningOp<affine::AffineApplyOp>();
      // a. Compose affine.apply operations.
      LLVM_DEBUG(affineApply->print(
          llvm::dbgs() << "\nCompose affine::AffineApplyOp recursively: "));
      AffineMap affineApplyMap = affineApply.getAffineMap();
      SmallVector<Value, 8> affineApplyOperands(
          affineApply.getOperands().begin(), affineApply.getOperands().end());

      SmallVector<AffineExpr, 0> dimRemapping(affineApplyMap.getNumDims());

      for (size_t i = 0; i < affineApplyMap.getNumDims(); ++i) {
        assert(i < affineApplyOperands.size());
        dimRemapping[i] = renumberOneDim(affineApplyOperands[i]);
      }
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols -
                                              affineApplyMap.getNumDims());
      for (unsigned idx = 0; idx < symRemapping.size(); ++idx) {
        symRemapping[idx] = renumberOneSymbol(
            affineApplyOperands[idx + affineApplyMap.getNumDims()]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(
          affineApplyMap.print(llvm::dbgs() << "\nAffine apply fixup map: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));
    } else {
      if (!isValidSymbolInt(t, /*recur*/ false)) {
        if (t.getDefiningOp()) {
          if ((t = fix(t, false))) {
            assert(isValidSymbolInt(t, /*recur*/ false));
          } else
            llvm_unreachable("cannot move");
        } else
          llvm_unreachable("cannot move2");
      }
      // A narrow iN symbol, read as the casts stripped above it read it: a
      // constant as that value, anything else through an index cast with
      // that signedness.
      if (needsNoWrapProof(t.getType())) {
        if (auto ci = t.getDefiningOp<ConstantIntOp>()) {
          AffineExpr c = getAffineConstantExpr(constantRead(ci, tUnsigned),
                                               ci.getContext());
          if (i < numDims)
            dimReplacements.push_back(c);
          else
            symReplacements.push_back(c);
          continue;
        }
        t = indexInput(t, tUnsigned);
      }
      if (i < numDims) {
        // b. The mathematical composition of AffineMap composes dims.
        dimReplacements.push_back(renumberOneDim(t));
      } else {
        // c. The mathematical composition of AffineMap concatenates symbols.
        //    Note that the map composition will put symbols already present
        //    in the map before any symbols coming from the auxiliary map, so
        //    we insert them before any symbols that are due to renumbering,
        //    and after the proper symbols we have seen already.
        symReplacements.push_back(renumberOneSymbol(t));
      }
    }
  }
  for (auto v : addedValues)
    concatenatedSymbols.push_back(v);

  // Create the new map by replacing each symbol at pos by the next new dim.
  unsigned numNewDims = reorderedDims.size();
  unsigned numNewSymbols = addedValues.size();
  assert(dimReplacements.size() == map.getNumDims());
  assert(symReplacements.size() == map.getNumSymbols());
  auto auxillaryMap = map.replaceDimsAndSymbols(
      dimReplacements, symReplacements, numNewDims, numNewSymbols);
  LLVM_DEBUG(auxillaryMap.print(llvm::dbgs() << "\nRewritten map: "));

  affineMap = auxillaryMap; // simplifyAffineMap(auxillaryMap);

  LLVM_DEBUG(affineMap.print(llvm::dbgs() << "\nSimplified result: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

AffineDimExpr AffineApplyNormalizer::renumberOneDim(Value v) {
  DenseMap<Value, unsigned>::iterator iterPos;
  bool inserted = false;
  std::tie(iterPos, inserted) =
      dimValueToPosition.insert(std::make_pair(v, dimValueToPosition.size()));
  if (inserted) {
    reorderedDims.push_back(v);
  }
  return getAffineDimExpr(iterPos->second, v.getContext())
      .cast<AffineDimExpr>();
}

static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value> *operands,
                                        PatternRewriter &rewriter,
                                        DominanceInfo &DI) {
  AffineApplyNormalizer normalizer(*map, *operands, rewriter, DI);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  affine::canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

bool need(AffineMap *map, SmallVectorImpl<Value> *operands) {
  assert(map->getNumInputs() == operands->size());
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}
bool need(IntegerSet *map, SmallVectorImpl<Value> *operands) {
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}

/// If `op` is an index cast (signed or unsigned), remember a cast of the same
/// kind of its input that is a valid affine symbol, so that the input is
/// re-cast through it (keeping the cast's signedness) rather than a new cast.
static void recordIndexCastOperand(Value op, IRMapping &indexMap,
                                   PatternRewriter &builder,
                                   DominanceInfo &DI) {
  Operation *idx0 = op.getDefiningOp();
  if (!idx0 || !isa<IndexCastOp, IndexCastUIOp>(idx0))
    return;
  Value in = idx0->getOperand(0);
  SmallVector<Value> attempt{op};
  for (auto &u : in.getUses()) {
    Operation *idx = u.getOwner();
    if (idx != idx0 && idx->getName() == idx0->getName())
      if (DI.dominates(idx, &*builder.getInsertionPoint()))
        attempt.push_back(idx->getResult(0));
  }
  for (Value idx : attempt) {
    if (affine::isValidSymbol(idx)) {
      indexMap.map(in, idx);
      break;
    }
  }
}

/// Cast a non-index map operand to index. The signedness is taken from how the
/// value is otherwise widened in the IR: a value that only reaches index or a
/// wider integer through index_castui/extui (an unsigned C index) gets
/// index_castui; anything else keeps the signed index_cast.
static Value castOperandToIndex(OpBuilder &builder, Value op) {
  bool hasUnsigned = false, hasSigned = false;
  for (Operation *u : op.getUsers()) {
    if (isa<IndexCastUIOp, ExtUIOp>(u))
      hasUnsigned = true;
    else if (isa<IndexCastOp, ExtSIOp>(u))
      hasSigned = true;
  }
  if (hasUnsigned && !hasSigned)
    return builder.create<IndexCastUIOp>(op.getLoc(), builder.getIndexType(),
                                         op);
  return builder.create<IndexCastOp>(op.getLoc(), builder.getIndexType(), op);
}

void fully2ComposeAffineMapAndOperands(PatternRewriter &builder, AffineMap *map,
                                       SmallVectorImpl<Value> *operands,
                                       DominanceInfo &DI) {
  IRMapping indexMap;
  for (auto op : *operands)
    recordIndexCastOperand(op, indexMap, builder, DI);
  assert(map->getNumInputs() == operands->size());
  while (need(map, operands)) {
    AffineMap prevMap = *map;
    SmallVector<Value, 8> prevOperands(operands->begin(), operands->end());
    composeAffineMapAndOperands(map, operands, builder, DI);
    assert(map->getNumInputs() == operands->size());
    // Nothing left that can be raised exactly.
    if (*map == prevMap && llvm::equal(*operands, prevOperands))
      break;
  }
  *map = simplifyAffineMap(*map);
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto *o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto BA = op.cast<BlockArgument>();
        toInsert = &BA.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        PatternRewriter::InsertionGuard B(builder);
        builder.setInsertionPoint(toInsert);
        op = castOperandToIndex(builder, op);
      }
    }
  }
}

void fully2ComposeIntegerSetAndOperands(PatternRewriter &builder,
                                        IntegerSet *set,
                                        SmallVectorImpl<Value> *operands,
                                        DominanceInfo &DI) {
  IRMapping indexMap;
  for (auto op : *operands)
    recordIndexCastOperand(op, indexMap, builder, DI);
  auto map = AffineMap::get(set->getNumDims(), set->getNumSymbols(),
                            set->getConstraints(), set->getContext());
  while (need(&map, operands)) {
    AffineMap prevMap = map;
    SmallVector<Value, 8> prevOperands(operands->begin(), operands->end());
    composeAffineMapAndOperands(&map, operands, builder, DI);
    // Nothing left that can be raised exactly.
    if (map == prevMap && llvm::equal(*operands, prevOperands))
      break;
  }
  map = simplifyAffineMap(map);
  *set = IntegerSet::get(map.getNumDims(), map.getNumSymbols(),
                         map.getResults(), set->getEqFlags());
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto *o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto BA = op.cast<BlockArgument>();
        toInsert = &BA.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        PatternRewriter::InsertionGuard B(builder);
        builder.setInsertionPoint(toInsert);
        op = castOperandToIndex(builder, op);
      }
    }
  }
}

namespace {
struct AffineCFGPass : public AffineCFGBase<AffineCFGPass> {
  void runOnOperation() override;
};
} // namespace

static void setLocationAfter(PatternRewriter &b, mlir::Value val) {
  if (val.getDefiningOp()) {
    auto it = val.getDefiningOp()->getIterator();
    it++;
    b.setInsertionPoint(val.getDefiningOp()->getBlock(), it);
  }
  if (auto bop = dyn_cast<mlir::BlockArgument>(val))
    b.setInsertionPoint(bop.getOwner(), bop.getOwner()->begin());
}

struct IndexCastMovement : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    mlir::Value val = op.getOperand();
    if (auto bop = dyn_cast<mlir::BlockArgument>(val)) {
      if (op.getOperation()->getBlock() != bop.getOwner()) {
        op.getOperation()->moveBefore(bop.getOwner(), bop.getOwner()->begin());
        return success();
      }
      return failure();
    }

    if (val.getDefiningOp()) {
      if (op.getOperation()->getBlock() != val.getDefiningOp()->getBlock()) {
        auto it = val.getDefiningOp()->getIterator();
        op.getOperation()->moveAfter(val.getDefiningOp()->getBlock(), it);
      }
      return failure();
    }
    return failure();
  }
};

/*
struct SimplfyIntegerCastMath : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<AddIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<AddIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), op.getType(), iadd.getOperand(0)),
          b2.create<IndexCastOp>(op.getLoc(), op.getType(),
                                 iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SubIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<SubIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<MulIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<MulIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), op.getType(), iadd.getOperand(0)),
          b2.create<IndexCastOp>(op.getLoc(), op.getType(),
                                 iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<DivUIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<DivUIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<DivSIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<DivSIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<RemUIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<RemUIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<RemSIOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<RemSIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SelectOp>()) {
      PatternRewriter b(rewriter);
      setLocationAfter(b, iadd.getTrueValue());
      PatternRewriter b2(rewriter);
      setLocationAfter(b2, iadd.getFalseValue());
      auto cond = iadd.getCondition();
      PatternRewriter b3(rewriter);
      setLocationAfter(b3, cond);
      if (auto cmp = iadd.getCondition().getDefiningOp<CmpIOp>()) {
        if (cmp.getLhs() == iadd.getTrueValue() &&
            cmp.getRhs() == iadd.getFalseValue()) {

          auto truev = b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                                    iadd.getTrueValue());
          auto falsev = b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                                      iadd.getFalseValue());
          cond = b3.create<CmpIOp>(cmp.getLoc(), cmp.getPredicate(), truev,
                                   falsev);
          rewriter.replaceOpWithNewOp<SelectOp>(op, cond, truev, falsev);
          return success();
        }
      }
    }
    return failure();
  }
};
*/

struct CanonicalizeAffineApply
    : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 4> mapOperands(affineOp.getMapOperands());
    auto map = affineOp.getMap();
    auto prevMap = map;

    auto *scope = affine::getAffineScope(affineOp)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &map, &mapOperands, DI);
    affine::canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(affineOp, map,
                                                       mapOperands);
    return success();
  }
};

/// Whether the cast `inner` (index_cast / index_castui of `src` to `mid`)
/// followed by a cast back that reads `mid` as `asUnsigned` gives `src` again:
/// always for an iN -> index -> iN round trip (an extension, then its
/// truncation); for index -> iN -> index only when the index value fits iN
/// as it is read (needsNoWrapProof; provably, or as the frontend marked the
/// truncation: the counter casts CanonicalizeFor creates hold such values).
/// A wide iN is assumed to hold the index value read signed; read unsigned,
/// it needs a non-negative one (unsignedReadKeepsValue): the bound
/// (unsigned)i of an int counter i = -1 is 4294967295, not -1.
static bool roundTripKeepsValue(Operation *inner, bool asUnsigned) {
  Value src = inner->getOperand(0);
  Type mid = inner->getResult(0).getType();
  if (!src.getType().isIndex() || truncationMarkedExact(inner, asUnsigned))
    return true;
  IntRangeOracle ranges;
  if (!needsNoWrapProof(mid))
    return unsignedReadKeepsValue(ranges, mid, src, asUnsigned);
  return rangeWithin(ranges.get(src, false),
                     typeRange(intOrIndexWidth(mid), asUnsigned));
}

struct CanonicalizeIndexCast : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp indexcastOp,
                                PatternRewriter &rewriter) const override {

    // Fold IndexCast(IndexCast(x)) -> x and IndexCast(IndexCastUI(x)) -> x
    // when the round trip keeps the value (roundTripKeepsValue).
    Operation *inner = indexcastOp.getOperand().getDefiningOp();
    if (isa_and_nonnull<IndexCastOp, IndexCastUIOp>(inner) &&
        inner->getOperand(0).getType() == indexcastOp.getType() &&
        roundTripKeepsValue(inner, /*asUnsigned=*/false)) {
      Value src = inner->getOperand(0);
      rewriter.replaceOp(indexcastOp, src);
      return success();
    }

    // Fold IndexCast(constant) -> constant
    // A little hack because we go through int.  Otherwise, the size
    // of the constant might need to change.
    if (auto cst = indexcastOp.getOperand().getDefiningOp<ConstantIntOp>()) {
      APInt v = cst.getValue().cast<IntegerAttr>().getValue();
      if (v.getSignificantBits() <= 64) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(indexcastOp,
                                                     v.getSExtValue());
        return success();
      }
    }
    return failure();
  }
};

/// The unsigned counterpart of CanonicalizeIndexCast:
/// IndexCastUI(IndexCastUI(x)) -> x and IndexCastUI(IndexCast(x)) -> x when
/// the types match and the round trip keeps the value (roundTripKeepsValue;
/// the index -> iN -> index round trip of an unsigned loop counter), and
/// IndexCastUI(constant) -> the zero-extended index constant.
struct CanonicalizeIndexCastUI : public OpRewritePattern<IndexCastUIOp> {
  using OpRewritePattern<IndexCastUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastUIOp indexcastOp,
                                PatternRewriter &rewriter) const override {
    Operation *inner = indexcastOp.getOperand().getDefiningOp();
    if (isa_and_nonnull<IndexCastOp, IndexCastUIOp>(inner) &&
        inner->getOperand(0).getType() == indexcastOp.getType() &&
        roundTripKeepsValue(inner, /*asUnsigned=*/true)) {
      Value src = inner->getOperand(0);
      rewriter.replaceOp(indexcastOp, src);
      return success();
    }

    if (auto cst = indexcastOp.getOperand().getDefiningOp<ConstantIntOp>()) {
      APInt v = cst.getValue().cast<IntegerAttr>().getValue();
      if (indexcastOp.getType().isIndex() && v.getActiveBits() < 64) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(indexcastOp,
                                                     v.getZExtValue());
        return success();
      }
    }
    return failure();
  }
};

/*
struct CanonicalizeAffineIf : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineIfOp affineOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;
    fully2ComposeAffineMapAndOperands(&map, &mapOperands);
    affine::canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);
    if (map == prevMap)
      return failure();
    rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(affineOp, map,
mapOperands); return success();
  }
};
*/

bool isValidIndex(Value val) { return isValidIndexAs(val, false); }

/// Whether `val`, read as `asUnsigned`, can be raised to an affine expression
/// of induction variables and symbols that keeps its value (see "Exact
/// raising of integer casts and arithmetic"); the normalizer raises exactly
/// what this accepts.
static bool isValidIndexAs(Value val, bool asUnsigned) {
  if (isValidSymbolInt(val))
    return true;

  auto [base, baseUnsigned] = peelExactCasts(val, asUnsigned);
  if (base != val)
    return isValidIndexAs(base, baseUnsigned);
  if (isa_and_nonnull<IndexCastOp, IndexCastUIOp, ExtSIOp, ExtUIOp>(
          val.getDefiningOp()))
    return false; // a cast that does not keep the value

  if (Operation *bop = val.getDefiningOp(); isRaisedIntOp(bop)) {
    if (!exactArith(bop, asUnsigned))
      return false;
    bool ou = operandsUnsigned(bop, asUnsigned);
    Value lhs = bop->getOperand(0), rhs = bop->getOperand(1);
    if (isa<AddIOp, SubIOp>(bop))
      return isValidIndexAs(lhs, ou) && isValidIndexAs(rhs, ou);
    if (isa<MulIOp>(bop))
      return (isValidIndexAs(lhs, ou) && isValidSymbolInt(rhs)) ||
             (isValidIndexAs(rhs, ou) && isValidSymbolInt(lhs));
    if (isa<DivSIOp, DivUIOp>(bop))
      return isValidIndexAs(lhs, ou) && isValidSymbolInt(rhs);
    return isValidIndexAs(lhs, ou) && rhs.getDefiningOp<arith::ConstantOp>();
  }

  if (val.getDefiningOp<ConstantIndexOp>())
    return true;

  if (val.getDefiningOp<ConstantIntOp>())
    return true;

  if (auto ba = dyn_cast<BlockArgument>(val)) {
    auto *owner = ba.getOwner();
    assert(owner);

    auto *parentOp = owner->getParentOp();
    if (!parentOp) {
      owner->dump();
      llvm::errs() << " ba: " << ba << "\n";
    }
    assert(parentOp);
    if (isa<FunctionOpInterface>(parentOp))
      return true;
    if (auto af = dyn_cast<affine::AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<affine::AffineParallelOp>(parentOp))
      return true;

    if (isa<FunctionOpInterface>(parentOp))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

// returns legality
bool handleMinMax(Value start, SmallVectorImpl<Value> &out, bool &min,
                  bool &max) {

  SmallVector<Value> todo = {start};
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (isValidIndex(cur)) {
      out.push_back(cur);
      continue;
    } else if (auto selOp = cur.getDefiningOp<SelectOp>()) {
      // UB only has min of operands
      if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
        if (cmp.getLhs() == selOp.getTrueValue() &&
            cmp.getRhs() == selOp.getFalseValue()) {
          todo.push_back(cmp.getLhs());
          todo.push_back(cmp.getRhs());
          if (cmp.getPredicate() == CmpIPredicate::sle ||
              cmp.getPredicate() == CmpIPredicate::slt) {
            min = true;
            continue;
          }
          if (cmp.getPredicate() == CmpIPredicate::sge ||
              cmp.getPredicate() == CmpIPredicate::sgt) {
            max = true;
            continue;
          }
        }
      }
    }
    return false;
  }
  return !(min && max);
}

bool handle(PatternRewriter &b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs,
            SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
  SmallVector<Value> lhs;
  bool lhs_min = false;
  bool lhs_max = false;
  if (!handleMinMax(cmpi.getLhs(), lhs, lhs_min, lhs_max)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal lhs: " << cmpi.getLhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(lhs.size());
  SmallVector<Value> rhs;
  bool rhs_min = false;
  bool rhs_max = false;
  if (!handleMinMax(cmpi.getRhs(), rhs, rhs_min, rhs_max)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal rhs: " << cmpi.getRhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(rhs.size());
  for (auto &lhspack : lhs)
    if (!lhspack.getType().isa<IndexType>()) {
      lhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), lhspack);
    }

  for (auto &rhspack : rhs)
    if (!rhspack.getType().isa<IndexType>()) {
      rhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), rhspack);
    }

  switch (cmpi.getPredicate()) {
  case CmpIPredicate::eq: {
    if (lhs_min || lhs_max || rhs_min || rhs_max)
      return false;
    eqflags.push_back(true);

    applies.push_back(lhs[0]);
    applies.push_back(rhs[0]);
    AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                          b.getAffineSymbolExpr(2 * exprs.size() + 1)};
    exprs.push_back(dims[0] - dims[1]);
  } break;

  case CmpIPredicate::ugt:
  case CmpIPredicate::uge:
    for (auto lhspack : lhs)
      if (!valueCmp(Cmp::GE, lhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal greater lhs icmp: " << cmpi << " - "
                                << lhspack << "\n");
        return false;
      }
    for (auto rhspack : rhs)
      if (!valueCmp(Cmp::GE, rhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal greater rhs icmp: " << cmpi << " - "
                                << rhspack << "\n");
        return false;
      }

  case CmpIPredicate::sge:
  case CmpIPredicate::sgt: {
    // if lhs >=? rhs
    // if lhs is a min(a, b) both must be true and this is fine
    // if lhs is a max(a, b) either may be true, and sets require and
    // similarly if rhs is a max(), both must be true;
    if (lhs_max || rhs_min)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[0] - dims[1];
        if (cmpi.getPredicate() == CmpIPredicate::sgt ||
            cmpi.getPredicate() == CmpIPredicate::ugt)
          expr = expr - 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::ult:
  case CmpIPredicate::ule:
    for (auto lhspack : lhs)
      if (!valueCmp(Cmp::GE, lhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal less lhs icmp: " << cmpi << " - "
                                << lhspack << "\n");
        return false;
      }
    for (auto rhspack : rhs)
      if (!valueCmp(Cmp::GE, rhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal less rhs icmp: " << cmpi << " - "
                                << rhspack << "\n");
        return false;
      }

  case CmpIPredicate::slt:
  case CmpIPredicate::sle: {
    if (lhs_min || rhs_max)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[1] - dims[0];
        if (cmpi.getPredicate() == CmpIPredicate::slt ||
            cmpi.getPredicate() == CmpIPredicate::ult)
          expr = expr - 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::ne:
    LLVM_DEBUG(llvm::dbgs() << "illegal icmp: " << cmpi << "\n");
    return false;
  }
  return true;
}
/*
static void replaceStore(memref::StoreOp store,
                         const SmallVector<Value, 2> &newIndexes) {
  auto memrefType = store.getMemRef().getType().cast<MemRefType>();
  size_t rank = memrefType.getRank();
  if (rank != newIndexes.size()) {
    llvm::errs() << store << "\n";
  }
  assert(rank == newIndexes.size() && "Expect rank to match new indexes");

  PatternRewriter builder(store);
  Location loc = store.getLoc();
  builder.create<affine::AffineStoreOp>(loc, store.getValueToStore(),
store.getMemRef(), newIndexes); store.erase();
}

static void replaceLoad(memref::LoadOp load,
                        const SmallVector<Value, 2> &newIndexes) {
  PatternRewriter builder(load);
  Location loc = load.getLoc();

  auto memrefType = load.getMemRef().getType().cast<MemRefType>();
  size_t rank = memrefType.getRank();
  if (rank != newIndexes.size()) {
    llvm::errs() << load << "\n";
  }
  assert(rank == newIndexes.size() && "rank must equal new indexes size");

  affine::AffineLoadOp affineLoad =
      builder.create<affine::AffineLoadOp>(loc, load.getMemRef(), newIndexes);
  load.getResult().replaceAllUsesWith(affineLoad.getResult());
  load.erase();
}
*/
struct MoveLoadToAffine : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(load.getIndices(), isValidIndex))
      return failure();

    auto memrefType = load.getMemRef().getType().cast<MemRefType>();
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());

    SmallVector<Value, 4> operands = load.getIndices();

    if (map.getNumInputs() != operands.size()) {
      // load->getParentOfType<FuncOp>().dump();
      llvm::errs() << " load: " << load << "\n";
    }
    auto *scope = affine::getAffineScope(load)->getParentOp();
    DominanceInfo DI(scope);
    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, DI);
    assert(map.getNumInputs() == operands.size());
    affine::canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    affine::AffineLoadOp affineLoad = rewriter.create<affine::AffineLoadOp>(
        load.getLoc(), load.getMemRef(), map, operands);
    load.getResult().replaceAllUsesWith(affineLoad.getResult());
    rewriter.eraseOp(load);
    return success();
  }
};

struct MoveStoreToAffine : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(store.getIndices(), isValidIndex))
      return failure();

    auto memrefType = store.getMemRef().getType().cast<MemRefType>();
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());
    SmallVector<Value, 4> operands = store.getIndices();

    auto *scope = affine::getAffineScope(store)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, DI);
    affine::canonicalizeMapAndOperands(&map, &operands);

    auto newStore = rewriter.create<affine::AffineStoreOp>(
        store.getLoc(), store.getValueToStore(), store.getMemRef(), map,
        operands);
    copyHLSMarks(store, newStore);
    rewriter.eraseOp(store);
    return success();
  }
};

static bool areChanged(SmallVectorImpl<Value> &afterOperands,
                       SmallVectorImpl<Value> &beforeOperands) {
  if (afterOperands.size() != beforeOperands.size())
    return true;
  if (!std::equal(afterOperands.begin(), afterOperands.end(),
                  beforeOperands.begin()))
    return true;
  return false;
}

template <typename T> struct AffineFixup : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  /// Replace the affine op with another instance of it with the supplied
  /// map and mapOperands.
  void replaceAffineOp(PatternRewriter &rewriter, T affineOp, AffineMap map,
                       ArrayRef<Value> mapOperands) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();
    SmallVector<Value, 4> operands = op.getMapOperands();

    auto prevMap = map;
    auto prevOperands = operands;

    auto *scope = affine::getAffineScope(op)->getParentOp();
    DominanceInfo DI(scope);

    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, DI);
    assert(map.getNumInputs() == operands.size());
    affine::canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    if (map == prevMap && !areChanged(operands, prevOperands))
      return failure();

    replaceAffineOp(rewriter, op, map, operands);
    return success();
  }
};

// Specialize the template to account for the different build signatures for
// affine load, store, and apply ops.
template <>
void AffineFixup<affine::AffineLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineLoadOp load, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(load, load.getMemRef(), map,
                                                    mapOperands);
}
template <>
void AffineFixup<affine::AffinePrefetchOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffinePrefetchOp prefetch, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffinePrefetchOp>(
      prefetch, prefetch.getMemref(), map, mapOperands,
      prefetch.getLocalityHint(), prefetch.getIsWrite(),
      prefetch.getIsDataCache());
}
template <>
void AffineFixup<affine::AffineStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineStoreOp store, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  auto newStore = rewriter.create<affine::AffineStoreOp>(
      store.getLoc(), store.getValueToStore(), store.getMemRef(), map,
      mapOperands);
  copyHLSMarks(store, newStore);
  rewriter.replaceOp(store, newStore->getResults());
}
template <>
void AffineFixup<affine::AffineVectorLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineVectorLoadOp vectorload,
    AffineMap map, ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineVectorLoadOp>(
      vectorload, vectorload.getVectorType(), vectorload.getMemRef(), map,
      mapOperands);
}
template <>
void AffineFixup<affine::AffineVectorStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineVectorStoreOp vectorstore,
    AffineMap map, ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineVectorStoreOp>(
      vectorstore, vectorstore.getValueToStore(), vectorstore.getMemRef(), map,
      mapOperands);
}

// Generic version for ops that don't have extra operands.
template <typename AffineOpTy>
void AffineFixup<AffineOpTy>::replaceAffineOp(
    PatternRewriter &rewriter, AffineOpTy op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineOpTy>(op, map, mapOperands);
}

struct CanonicalieForBounds : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());
    SmallVector<Value, 4> origLbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> origUbOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    auto *scope = affine::getAffineScope(forOp)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands, DI);
    affine::canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands, DI);
    affine::canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    // ubMap.dump();
    // forOp.dump();

    // Any canonicalization change in map or operands always leads to updated
    // map(s).
    if ((lbMap == prevLbMap && ubMap == prevUbMap) &&
        (!areChanged(lbOperands, origLbOperands)) &&
        (!areChanged(ubOperands, origUbOperands)))
      return failure();

    // llvm::errs() << "oldParent:" << *forOp.getParentOp() << "\n";
    // llvm::errs() << "oldfor:" << forOp << "\n";

    if ((lbMap != prevLbMap) || areChanged(lbOperands, origLbOperands))
      forOp.setLowerBound(lbOperands, lbMap);
    if ((ubMap != prevUbMap) || areChanged(ubOperands, origUbOperands))
      forOp.setUpperBound(ubOperands, ubMap);

    // llvm::errs() << "newfor:" << forOp << "\n";
    return success();
  }
};

struct CanonicalizIfBounds : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands(op.getOperands());
    SmallVector<Value, 4> origOperands(operands);

    auto map = op.getIntegerSet();
    auto prevMap = map;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    auto *scope = affine::getAffineScope(op)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeIntegerSetAndOperands(rewriter, &map, &operands, DI);
    affine::canonicalizeSetAndOperands(&map, &operands);

    // map(s).
    if (map == prevMap && !areChanged(operands, origOperands))
      return failure();

    op.setConditional(map, operands);

    return success();
  }
};

struct MoveIfToAffine : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp->getParentOfType<affine::AffineForOp>() &&
        !ifOp->getParentOfType<affine::AffineParallelOp>())
      return failure();

    std::vector<mlir::Type> types;
    for (auto v : ifOp.getResults()) {
      types.push_back(v.getType());
    }

    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;
    SmallVector<Value, 4> applies;

    std::deque<Value> todo = {ifOp.getCondition()};
    while (todo.size()) {
      auto cur = todo.front();
      todo.pop_front();
      if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
        if (!handle(rewriter, cmpi, exprs, eqflags, applies)) {
          return failure();
        }
        continue;
      }
      if (auto andi = cur.getDefiningOp<AndIOp>()) {
        todo.push_back(andi.getOperand(0));
        todo.push_back(andi.getOperand(1));
        continue;
      }
      return failure();
    }

    auto *scope = affine::getAffineScope(ifOp)->getParentOp();
    DominanceInfo DI(scope);

    auto iset =
        IntegerSet::get(/*dim*/ 0, /*symbol*/ 2 * exprs.size(), exprs, eqflags);
    fully2ComposeIntegerSetAndOperands(rewriter, &iset, &applies, DI);
    affine::canonicalizeSetAndOperands(&iset, &applies);
    affine::AffineIfOp affineIfOp =
        rewriter.create<affine::AffineIfOp>(ifOp.getLoc(), types, iset, applies,
                                            /*elseBlock=*/true);

    rewriter.setInsertionPoint(ifOp.thenYield());
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
        ifOp.thenYield(), ifOp.thenYield().getOperands());

    rewriter.eraseBlock(affineIfOp.getThenBlock());
    rewriter.eraseBlock(affineIfOp.getElseBlock());
    if (ifOp.getElseRegion().getBlocks().size()) {
      rewriter.setInsertionPoint(ifOp.elseYield());
      rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
          ifOp.elseYield(), ifOp.elseYield().getOperands());
    }

    rewriter.inlineRegionBefore(ifOp.getThenRegion(),
                                affineIfOp.getThenRegion(),
                                affineIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(),
                                affineIfOp.getElseRegion(),
                                affineIfOp.getElseRegion().begin());

    rewriter.replaceOp(ifOp, affineIfOp.getResults());
    return success();
  }
};

void AffineCFGPass::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add</*SimplfyIntegerCastMath, */ CanonicalizeAffineApply,
          CanonicalizeIndexCast, CanonicalizeIndexCastUI,
          /* IndexCastMovement,*/ AffineFixup<affine::AffineLoadOp>,
          AffineFixup<affine::AffineStoreOp>, CanonicalizIfBounds,
          MoveStoreToAffine, MoveIfToAffine, MoveLoadToAffine,
          CanonicalieForBounds>(getOperation()->getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}
