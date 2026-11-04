//===- XlxStringBits.cc - Fold ap_int / ap_fixed string constructors ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Under __SYNTHESIS__ the Vitis HLS headers construct ap_int / ap_fixed values
// from C strings through
//
//   typeof(Base::V) t;
//   _ssdm_string2bits((void*)(&t), (const char*)(s), radix, I, S, Q, O, N, C99);
//   Base::V = t;
//
// (etc/ap_fixed_base.h:458-471: I = _AP_I and the object's Q / O / N modes;
// etc/ap_int_base.h:398-410: I = _AP_W, AP_TRN, AP_WRAP, 0). The header notes
// that "_ssdm_string2bits only takes const string and const radix" and that
// "XFORM will do compile time processing of the string": the closed Vitis
// front end folds it. The fixed-point math headers rely on it for their
// lookup tables (etc/hls_exp_apfixed.h:225 `const static ap_ufixed<25,11>
// exp_x_msb_1_table[32] = {"0x1.0000p0", ...}` and others). Left as a call,
// every element is constructed at run time behind a static-init guard and
// cgeist spends minutes on a 256-entry table (W11 experiment: 493 s).
//
// This file evaluates such a construction at compile time. The value follows
// the Vitis C simulation of the same constructor (Vitis/include/etc/
// ap_fixed_base.h:175-262 fromString, etc/ap_private.h:372-500 parseString;
// read, not copied): the string's number (sign; 0x / 0b / 0o prefix, else the
// given radix; a decimal `e` exponent, or a binary `p` exponent for the
// power-of-two radixes) converted to the target format with its quantisation
// and overflow modes (etc/ap_fixed_base.h:302-381 overflow_adjust /
// quantization_adjust). The C simulation rounds from a (W - I + 4)-bit
// fraction with a sticky bit; this evaluates the exact value, which can only
// differ for a decimal string within 2^-(W-I+4) of a rounding boundary
// (unconfirmed against Vitis: the XFORM fold is closed).
//
//===----------------------------------------------------------------------===//

#include "XlxStringBits.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using llvm::APInt;

namespace cgeist {
namespace xlx {

namespace {

/// A parsed number: (-1)^neg * mant * radix^-fracDigits * base^exp, with
/// base 10 for radix 10 and 2 otherwise (ap_private.h:372-500: a radix-10
/// string shifts its decimal point by the `e` exponent, the others are
/// rewritten to binary and shift the binary point by the `p` exponent).
struct ParsedNumber {
  bool neg = false;
  unsigned radix = 10;
  std::string digits; ///< mantissa digits without the point
  int64_t fracDigits = 0;
  int64_t exp = 0;
};

int digitValue(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'z')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'Z')
    return c - 'A' + 10;
  return -1;
}

std::optional<int64_t> parseExponent(llvm::StringRef s) {
  if (s.empty())
    return std::nullopt;
  bool neg = false;
  if (s[0] == '+' || s[0] == '-') {
    neg = s[0] == '-';
    s = s.drop_front();
  }
  if (s.empty() || s.size() > 9)
    return std::nullopt;
  int64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9')
      return std::nullopt;
    v = v * 10 + (c - '0');
  }
  return neg ? -v : v;
}

std::optional<ParsedNumber> parseNumber(llvm::StringRef s, unsigned radix) {
  // parseString trims spaces only.
  while (!s.empty() && s.front() == ' ')
    s = s.drop_front();
  while (!s.empty() && s.back() == ' ')
    s = s.drop_back();
  if (s.empty())
    return std::nullopt;
  ParsedNumber n;
  if (s[0] == '-' || s[0] == '+') {
    n.neg = s[0] == '-';
    s = s.drop_front();
  }
  if (radix == 0)
    radix = 10;
  if (s.size() >= 2 && s[0] == '0') {
    char p = s[1];
    if (p == 'x' || p == 'X')
      radix = 16;
    else if (p == 'b' || p == 'B')
      radix = 2;
    else if (p == 'o' || p == 'O')
      radix = 8;
    if (p == 'x' || p == 'X' || p == 'b' || p == 'B' || p == 'o' || p == 'O')
      s = s.drop_front(2);
  }
  if (radix != 2 && radix != 8 && radix != 10 && radix != 16)
    return std::nullopt;
  n.radix = radix;
  size_t expPos = radix == 10 ? s.find_first_of("eE") : s.find_first_of("pP");
  llvm::StringRef mant = s.substr(0, expPos);
  if (expPos != llvm::StringRef::npos) {
    auto e = parseExponent(s.substr(expPos + 1));
    if (!e)
      return std::nullopt;
    n.exp = *e;
  }
  bool point = false;
  for (char c : mant) {
    if (c == '.') {
      if (point)
        return std::nullopt;
      point = true;
      continue;
    }
    int d = digitValue(c);
    if (d < 0 || d >= (int)radix)
      return std::nullopt;
    n.digits.push_back(c);
    if (point)
      n.fracDigits++;
  }
  if (n.digits.empty())
    return std::nullopt;
  return n;
}

/// floor(|value| * 2^F) and the two rounding bits of the discarded fraction
/// f: qb = (f >= 1/2), r = the bits below qb are not all zero (the
/// quantization_adjust inputs, ap_fixed_base.h:366-381).
struct Scaled {
  APInt q; ///< unsigned magnitude
  bool qb = false;
  bool r = false;
};

constexpr unsigned kMaxBits = 1u << 16;

std::optional<Scaled> scaleMagnitude(const ParsedNumber &n, int64_t F) {
  unsigned bitsPerDigit = n.radix == 16 ? 4 : n.radix == 8 ? 3 : n.radix == 2
                                                                  ? 1
                                                                  : 4;
  uint64_t mantBits = n.digits.size() * (uint64_t)bitsPerDigit + 2;
  if (mantBits > kMaxBits)
    return std::nullopt;
  APInt mant(mantBits, 0);
  for (char c : n.digits) {
    mant *= APInt(mantBits, n.radix);
    mant += APInt(mantBits, digitValue(c));
  }
  Scaled out;
  if (n.radix != 10) {
    // value = mant * 2^(exp - bitsPerDigit * fracDigits); scaled by 2^F.
    int64_t t = n.exp - (int64_t)bitsPerDigit * n.fracDigits + F;
    if (t >= 0) {
      if ((uint64_t)t + mantBits > kMaxBits)
        return std::nullopt;
      out.q = mant.zext(mantBits + t).shl(t);
      return out;
    }
    uint64_t s = -t;
    if (s > mantBits) {
      // Every mantissa bit lies below the qb position.
      out.q = APInt(mantBits, 0);
      out.qb = false;
      out.r = !mant.isZero();
      return out;
    }
    out.q = mant.lshr(s);
    out.qb = mant[s - 1];
    out.r = s >= 2 && !mant.trunc(s - 1).isZero();
    return out;
  }
  // value = mant * 10^d, d = exp - fracDigits; scaled by 2^F.
  int64_t d = n.exp - n.fracDigits;
  uint64_t width = mantBits + (uint64_t)std::abs(F) + 4 * (uint64_t)std::abs(d) + 4;
  if (width > kMaxBits)
    return std::nullopt;
  APInt num = mant.zext(width), den(width, 1);
  if (F > 0)
    num = num.shl(F);
  else if (F < 0)
    den = den.shl(-F);
  APInt ten(width, 10);
  for (int64_t i = 0; i < std::abs(d); ++i) {
    if (d > 0)
      num *= ten;
    else
      den *= ten;
  }
  APInt rem(width, 0);
  APInt::udivrem(num, den, out.q, rem);
  APInt twice = rem.shl(1);
  out.qb = twice.uge(den);
  out.r = out.qb ? twice.ugt(den) : !rem.isZero();
  return out;
}

bool knownQ(llvm::StringRef q) {
  return llvm::StringSwitch<bool>(q)
      .Cases("AP_RND", "AP_RND_ZERO", "AP_RND_MIN_INF", "AP_RND_INF", true)
      .Cases("AP_RND_CONV", "AP_TRN", "AP_TRN_ZERO", true)
      .Default(false);
}

bool knownO(llvm::StringRef o) {
  return llvm::StringSwitch<bool>(o)
      .Cases("AP_SAT", "AP_SAT_ZERO", "AP_SAT_SYM", "AP_WRAP", "AP_WRAP_SM",
             true)
      .Default(false);
}

/// Set bits [lo, hi] of v to `ones`.
void setRange(APInt &v, unsigned lo, unsigned hi, bool ones) {
  for (unsigned b = lo; b <= hi && b < v.getBitWidth(); ++b) {
    if (ones)
      v.setBit(b);
    else
      v.clearBit(b);
  }
}

} // namespace

std::optional<APInt> stringToFixedBits(llvm::StringRef str, unsigned radix,
                                       const FixedFormat &fmt) {
  if (fmt.W == 0 || fmt.W > kMaxBits || !knownQ(fmt.Q) || !knownO(fmt.O) ||
      fmt.N < 0 || fmt.N > (int64_t)fmt.W)
    return std::nullopt;
  auto n = parseNumber(str, radix);
  if (!n)
    return std::nullopt;
  int64_t F = (int64_t)fmt.W - fmt.I;
  if (std::abs(F) > (int64_t)kMaxBits)
    return std::nullopt;
  auto sc = scaleMagnitude(*n, F);
  if (!sc)
    return std::nullopt;

  // Signed floor of the value, and the rounding bits of its fraction.
  unsigned width = std::max(sc->q.getBitWidth(), fmt.W) + 3;
  APInt q = sc->q.zext(width);
  bool qb = sc->qb, r = sc->r;
  bool nonzero = !q.isZero() || qb || r;
  bool isneg = n->neg && nonzero;
  if (isneg) {
    bool frac = qb || r;
    q.negate();
    if (frac) {
      q -= 1;
      // fraction 1 - f: >= 1/2 iff f <= 1/2; the bits below are nonzero
      // unless 1 - f is exactly 1/2.
      bool nqb = !(qb && r);
      bool nr = nqb ? !qb : true;
      qb = nqb;
      r = nr;
    }
  }

  // Quantisation (quantization_adjust): add qb under the mode's condition.
  bool inc = false;
  if (fmt.Q == "AP_RND")
    inc = qb;
  else if (fmt.Q == "AP_RND_ZERO")
    inc = qb && (isneg || r);
  else if (fmt.Q == "AP_RND_MIN_INF")
    inc = qb && r;
  else if (fmt.Q == "AP_RND_INF")
    inc = qb && (!isneg || r);
  else if (fmt.Q == "AP_RND_CONV")
    inc = qb && (q[0] || r);
  else if (fmt.Q == "AP_TRN_ZERO")
    inc = isneg && (qb || r);
  if (inc)
    q += 1;

  // Overflow (overflow_adjust): the target's range for the W-bit pattern.
  unsigned W = fmt.W;
  APInt maxV = fmt.S ? APInt::getSignedMaxValue(W).sext(width)
                     : APInt::getMaxValue(W).zext(width);
  APInt minV = fmt.S ? APInt::getSignedMinValue(W).sext(width)
                     : APInt(width, 0);
  bool overflow = q.sgt(maxV);
  bool underflow = q.slt(minV);
  APInt v = q.trunc(W);
  if (!overflow && !underflow)
    return v;
  bool sign = isneg; // neg_src
  if (fmt.O == "AP_WRAP") {
    if (fmt.N == 0)
      return v;
    if (fmt.S) {
      setRange(v, W - 1, W - 1, sign);
      if (fmt.N > 1)
        setRange(v, W - fmt.N, W - 2, !sign);
    } else {
      setRange(v, W - fmt.N, W - 1, true);
    }
    return v;
  }
  if (fmt.O == "AP_SAT_ZERO")
    return APInt(W, 0);
  if (fmt.O == "AP_WRAP_SM" && fmt.S) {
    bool Ro = v[W - 1];
    bool lD = q[W]; // the last deleted bit
    if (fmt.N == 0) {
      if (lD != Ro) {
        v.flipAllBits();
        setRange(v, W - 1, W - 1, lD);
      }
    } else {
      if (fmt.N == 1 && sign != Ro) {
        v.flipAllBits();
      } else if (fmt.N > 1) {
        bool lNo = v[W - fmt.N];
        if (lNo == sign)
          v.flipAllBits();
        setRange(v, W - fmt.N, W - 2, !sign);
      }
      setRange(v, W - 1, W - 1, sign);
    }
    return v;
  }
  // AP_SAT, AP_SAT_SYM (and AP_WRAP_SM of an unsigned type).
  if (fmt.S) {
    if (overflow)
      return APInt::getSignedMaxValue(W);
    APInt m = APInt::getSignedMinValue(W);
    if (fmt.O == "AP_SAT_SYM")
      m.setBit(0);
    return m;
  }
  return overflow ? APInt::getMaxValue(W) : APInt(W, 0);
}

namespace {

/// Looks through temporaries, elidable copies and no-op / constructor
/// conversions to the constructor call that makes the value.
const CXXConstructExpr *stripToConstruct(const Expr *E) {
  for (unsigned guard = 0; E && guard < 32; ++guard) {
    E = E->IgnoreParens();
    if (auto *X = dyn_cast<ExprWithCleanups>(E)) {
      E = X->getSubExpr();
      continue;
    }
    if (auto *X = dyn_cast<MaterializeTemporaryExpr>(E)) {
      E = X->getSubExpr();
      continue;
    }
    if (auto *X = dyn_cast<CXXBindTemporaryExpr>(E)) {
      E = X->getSubExpr();
      continue;
    }
    if (auto *X = dyn_cast<CastExpr>(E)) {
      if (X->getCastKind() == CK_NoOp ||
          X->getCastKind() == CK_ConstructorConversion) {
        E = X->getSubExpr();
        continue;
      }
      return nullptr;
    }
    if (auto *CE = dyn_cast<CXXConstructExpr>(E)) {
      const CXXConstructorDecl *C = CE->getConstructor();
      if (CE->getNumArgs() == 1 &&
          (CE->isElidable() ||
           ((C->isCopyConstructor() || C->isMoveConstructor()) &&
            C->isTrivial()))) {
        E = CE->getArg(0);
        continue;
      }
      return CE;
    }
    return nullptr;
  }
  return nullptr;
}

/// An argument of a call inside the constructor `fn` (called with `args`):
/// a reference to one of fn's parameters becomes the caller's argument.
const Expr *mapArg(const Expr *e, const FunctionDecl *fn,
                   llvm::ArrayRef<const Expr *> args) {
  if (auto *DRE = dyn_cast<DeclRefExpr>(e->IgnoreParenCasts()))
    if (auto *P = dyn_cast<ParmVarDecl>(DRE->getDecl()))
      if (auto *Owner = dyn_cast<FunctionDecl>(P->getDeclContext()))
        if (Owner->getCanonicalDecl() == fn->getCanonicalDecl()) {
          unsigned i = P->getFunctionScopeIndex();
          return i < args.size() ? args[i] : nullptr;
        }
  return e;
}

std::optional<int64_t> evalInt(const Expr *e, ASTContext &Ctx) {
  if (!e || e->isValueDependent())
    return std::nullopt;
  Expr::EvalResult R;
  if (!e->EvaluateAsInt(R, Ctx))
    return std::nullopt;
  return R.Val.getInt().getExtValue();
}

/// The enumerator name of an ap_q_mode / ap_o_mode argument.
std::optional<llvm::StringRef> enumName(const Expr *e, ASTContext &Ctx) {
  if (!e)
    return std::nullopt;
  auto v = evalInt(e, Ctx);
  if (!v)
    return std::nullopt;
  const Expr *x = e->IgnoreParenImpCasts();
  const EnumType *ET = x->getType()->getAs<EnumType>();
  if (!ET)
    if (auto *S = dyn_cast<SubstNonTypeTemplateParmExpr>(x))
      ET = S->getReplacement()->IgnoreParenImpCasts()->getType()
               ->getAs<EnumType>();
  if (!ET)
    return std::nullopt;
  for (const EnumConstantDecl *EC : ET->getDecl()->enumerators())
    if (EC->getInitVal().getExtValue() == *v)
      return EC->getName();
  return std::nullopt;
}

const CallExpr *findCall(const Stmt *S, llvm::StringRef name) {
  if (!S)
    return nullptr;
  if (auto *CE = dyn_cast<CallExpr>(S))
    if (const FunctionDecl *FD = CE->getDirectCallee())
      if (FD->getIdentifier() && FD->getName() == name)
        return CE;
  for (const Stmt *C : S->children())
    if (const CallExpr *R = findCall(C, name))
      return R;
  return nullptr;
}

std::optional<APInt> foldCtor(const CXXConstructorDecl *ctor,
                              llvm::ArrayRef<const Expr *> args,
                              ASTContext &Ctx, unsigned depth) {
  if (!ctor || depth > 4)
    return std::nullopt;
  const FunctionDecl *defFn = nullptr;
  if (!ctor->hasBody(defFn) || !defFn)
    return std::nullopt;
  auto *def = dyn_cast<CXXConstructorDecl>(defFn);
  if (!def)
    return std::nullopt;

  // ap_fixed(const char* s) : Base(s) {} - follow the base / delegating
  // initialiser that receives the string.
  for (const CXXCtorInitializer *init : def->inits()) {
    if (!init->isBaseInitializer() && !init->isDelegatingInitializer())
      continue;
    const CXXConstructExpr *inner = stripToConstruct(init->getInit());
    if (!inner)
      continue;
    llvm::SmallVector<const Expr *, 4> innerArgs;
    bool passesString = false;
    for (const Expr *a : inner->arguments()) {
      const Expr *m = mapArg(a, def, args);
      if (!m)
        return std::nullopt;
      if (m != a)
        passesString |= isa<StringLiteral>(m->IgnoreParenImpCasts());
      innerArgs.push_back(m);
    }
    if (passesString)
      return foldCtor(inner->getConstructor(), innerArgs, Ctx, depth + 1);
  }

  const CallExpr *call = findCall(def->getBody(), "_ssdm_string2bits");
  if (!call || call->getNumArgs() < 8)
    return std::nullopt;
  // Destination: (void*)(&t) with t an integer of the object's width.
  auto *addr = dyn_cast<UnaryOperator>(call->getArg(0)->IgnoreParenCasts());
  if (!addr || addr->getOpcode() != UO_AddrOf)
    return std::nullopt;
  QualType dstTy = addr->getSubExpr()->getType();
  if (!dstTy->isIntegerType() || dstTy->isBooleanType())
    return std::nullopt;
  unsigned W = Ctx.getIntWidth(dstTy);

  const Expr *strArg = mapArg(call->getArg(1), def, args);
  if (!strArg)
    return std::nullopt;
  auto *lit = dyn_cast<StringLiteral>(strArg->IgnoreParenCasts());
  if (!lit || !lit->isOrdinary())
    return std::nullopt;
  auto radix = evalInt(mapArg(call->getArg(2), def, args), Ctx);
  auto I = evalInt(mapArg(call->getArg(3), def, args), Ctx);
  auto S = evalInt(mapArg(call->getArg(4), def, args), Ctx);
  auto Q = enumName(mapArg(call->getArg(5), def, args), Ctx);
  auto O = enumName(mapArg(call->getArg(6), def, args), Ctx);
  auto N = evalInt(mapArg(call->getArg(7), def, args), Ctx);
  if (!radix || !I || !S || !Q || !O || !N || *radix < 0)
    return std::nullopt;
  FixedFormat fmt;
  fmt.W = W;
  fmt.I = *I;
  fmt.S = *S != 0;
  fmt.Q = *Q;
  fmt.O = *O;
  fmt.N = *N;
  return stringToFixedBits(lit->getString(), (unsigned)*radix, fmt);
}

/// The enumerator name of an enum-typed integral template argument.
std::optional<llvm::StringRef> enumArgName(const TemplateArgument &a) {
  if (a.getKind() != TemplateArgument::Integral)
    return std::nullopt;
  const EnumType *ET = a.getIntegralType()->getAs<EnumType>();
  if (!ET)
    return std::nullopt;
  for (const EnumConstantDecl *EC : ET->getDecl()->enumerators())
    if (llvm::APSInt::isSameValue(EC->getInitVal(), a.getAsIntegral()))
      return EC->getName();
  return std::nullopt;
}

std::optional<int64_t> intArg(const TemplateArgument &a) {
  if (a.getKind() != TemplateArgument::Integral)
    return std::nullopt;
  return a.getAsIntegral().getExtValue();
}

/// The format of an ap_int_base / ap_fixed_base class or of a class derived
/// from one through single bases (ap_uint<W>, ap_fixed<W, I, Q, O, N>).
/// `fixed` says which of the two it is.
std::optional<FixedFormat> apFormat(const CXXRecordDecl *RD, bool &fixed) {
  for (unsigned guard = 0; RD && guard < 8; ++guard) {
    if (auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      const TemplateArgumentList &args = spec->getTemplateArgs();
      if (spec->getName() == "ap_int_base" && args.size() == 2) {
        auto W = intArg(args[0]), S = intArg(args[1]);
        if (!W || !S || *W <= 0)
          return std::nullopt;
        FixedFormat f;
        f.W = (unsigned)*W;
        f.I = *W;
        f.S = *S != 0;
        fixed = false;
        return f;
      }
      if (spec->getName() == "ap_fixed_base" && args.size() == 6) {
        auto W = intArg(args[0]), I = intArg(args[1]), S = intArg(args[2]),
             N = intArg(args[5]);
        auto Q = enumArgName(args[3]), O = enumArgName(args[4]);
        if (!W || !I || !S || !N || !Q || !O || *W <= 0)
          return std::nullopt;
        FixedFormat f;
        f.W = (unsigned)*W;
        f.I = *I;
        f.S = *S != 0;
        f.Q = *Q;
        f.O = *O;
        f.N = *N;
        fixed = true;
        return f;
      }
    }
    if (RD->getNumBases() != 1)
      return std::nullopt;
    RD = RD->bases_begin()->getType()->getAsCXXRecordDecl();
  }
  return std::nullopt;
}

} // namespace

std::optional<APInt> foldDefaultConstruct(const Expr *E, ASTContext &Ctx) {
  const CXXConstructExpr *CE = stripToConstruct(E);
  if (!CE || CE->getNumArgs() != 0 ||
      !CE->getConstructor()->isDefaultConstructor())
    return std::nullopt;
  bool fixed = false;
  auto fmt = apFormat(CE->getConstructor()->getParent(), fixed);
  if (!fmt)
    return std::nullopt;
  return APInt(fmt->W, 0);
}

std::optional<APInt> foldNumericConstruct(const Expr *E, ASTContext &Ctx) {
  const CXXConstructExpr *CE = stripToConstruct(E);
  if (!CE || CE->getNumArgs() != 1)
    return std::nullopt;
  const Expr *arg = CE->getArg(0);
  QualType argTy = arg->IgnoreParenImpCasts()->getType();
  if (arg->isValueDependent() ||
      !(argTy->isIntegerType() || argTy->isRealFloatingType()))
    return std::nullopt;
  bool fixed = false;
  auto fmt = apFormat(CE->getConstructor()->getParent(), fixed);
  // N saturation bits and AP_WRAP_SM are rare in tables and not checked
  // against the numeric constructors: leave them to the caller's fallback.
  if (!fmt || fmt->N != 0 || fmt->O == "AP_WRAP_SM")
    return std::nullopt;
  // The constant at the argument's own type (before the implicit conversion
  // to the constructor's parameter type, which the headers' overloads avoid
  // by having one constructor per C type).
  const Expr *value = arg->IgnoreParenImpCasts();
  Expr::EvalResult R;
  if (!value->EvaluateAsRValue(R, Ctx) || R.HasSideEffects)
    return std::nullopt;
  // The exact value as a string in the form stringToFixedBits reads: decimal
  // for an integer, the exact hexadecimal significand and binary exponent
  // for a floating-point value.
  llvm::SmallString<64> text;
  if (R.Val.isInt()) {
    R.Val.getInt().toString(text, 10);
  } else if (R.Val.isFloat() && fixed) {
    const llvm::APFloat &f = R.Val.getFloat();
    if (!f.isFinite())
      return std::nullopt;
    if (f.isZero())
      return APInt(fmt->W, 0);
    char buf[128];
    unsigned n = f.convertToHexString(buf, 0, /*UpperCase*/ false,
                                      llvm::APFloat::rmNearestTiesToEven);
    text.assign(buf, buf + n);
  } else {
    return std::nullopt;
  }
  return stringToFixedBits(text, 10, *fmt);
}

std::optional<APInt> foldStringConstruct(const Expr *E, ASTContext &Ctx) {
  const CXXConstructExpr *CE = stripToConstruct(E);
  if (!CE || CE->getNumArgs() == 0)
    return std::nullopt;
  if (!isa<StringLiteral>(CE->getArg(0)->IgnoreParenImpCasts()))
    return std::nullopt;
  llvm::SmallVector<const Expr *, 4> args(CE->arguments());
  return foldCtor(CE->getConstructor(), args, Ctx, 0);
}

} // namespace xlx
} // namespace cgeist
