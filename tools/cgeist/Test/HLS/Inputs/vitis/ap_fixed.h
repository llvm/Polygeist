// hls-est "exactclass" stand-in for the Vitis HLS ap_fixed.h.
//
// ap_fixed<W,I,Q,O,N> / ap_ufixed<...> derive from ap_fixed_base<W,I,S,Q,O,N>
// whose ONLY data member is the raw `_BitInt(W)` / `unsigned _BitInt(W)` V
// holding value * 2^(W-I). cgeist (--scalar-wrapper-abi) lowers the class to
// that scalar, so arrays of ap_fixed<16,5> are memref<...xi16> in the MLIR and
// every operation is the integer arithmetic below.
//
// Semantics follow etc/ap_fixed_base.h of Vitis 2025.1 (validated bit for bit
// against the real headers by tools/hls-est-validate/check_shim.py):
//   * result types of + - * / & | ^ per ap_fixed_base::RType, computed
//     exactly; `a OP= b` and `T x = expr` convert back with the target's
//     quantization (AP_TRN floor, AP_RND half-up, AP_RND_ZERO, AP_RND_MIN_INF,
//     AP_RND_INF, AP_RND_CONV, AP_TRN_ZERO) and overflow mode (AP_WRAP modular,
//     AP_SAT clamp, AP_SAT_ZERO, AP_SAT_SYM). N (saturation bits) and
//     AP_WRAP_SM are not implemented (treated as AP_WRAP / N = 0);
//   * int -> fixed: value << F then wrap/saturate; double -> fixed: the IEEE
//     mantissa is shifted exactly (floor for AP_TRN), then wrap/saturate;
//   * a / b: truncation toward zero of (a.V << max(F2,0)) / b.V, exact (the
//     synthesis path; the C simulation path divides in fewer bits, so
//     min / -1 LSB wraps there - VITIS_NOTES N11 - and traps on a zero
//     divisor; Vitis synthesis treats the quotient as undef - hls-est's
//     --vitis-fold models that);
//   * comparisons are exact (operands aligned to the larger F);
//   * Vitis has no ap_fixed OP double: such expressions are computed in
//     double through `operator double()`, as here; `(int)x` and to_int()
//     both truncate toward zero (to_int() through to_ap_int_base(), the
//     integer part, as in Vitis).
#ifndef HLSEST_EXACTCLASS_AP_FIXED_H
#define HLSEST_EXACTCLASS_AP_FIXED_H
#ifdef __cplusplus
#include "ap_int.h"
// Vitis marks every ap_int / ap_fixed member and operator AP_INLINE, i.e.
// `inline __attribute__((always_inline))` under __SYNTHESIS__
// (etc/ap_common.h): clang inlines them before Vitis's own inliner, which
// never lists them ([HLS 214-178]). cgeist's automatic-inlining rule
// (--hls-auto-inline, VITIS_NOTES N18) reads the same attribute.
#ifdef __clang__
#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#endif

template <int W, int I, bool S> struct _ap_fixed_factory { typedef ap_fixed<W, I, AP_TRN, AP_WRAP, 0> type; };
template <int W, int I> struct _ap_fixed_factory<W, I, false> { typedef ap_ufixed<W, I, AP_TRN, AP_WRAP, 0> type; };

template <int W, int I, bool S, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP, int N = 0>
struct ap_fixed_base {
  static_assert(W > 0, "ap_fixed_base bitwidth must be positive");
  typedef typename ap_detail::bits<W, S>::type raw_t;
  static const int width = W;
  static const int iwidth = I;
  static const ap_q_mode qmode = Q;
  static const ap_o_mode omode = O;
  static const int F = W - I; // fractional bits (may be negative or > W)

  raw_t V; // the only data member (uninitialised by default, like Vitis)

  /// The value of V (sign-extends the bit of a signed 1-bit type).
  typedef typename ap_detail::bits<W, S>::vtype val_t;
  constexpr val_t rval() const { return ap_detail::bits<W, S>::val(V); }

  /// Result types of the binary operators (ap_fixed_base.h RType).
  template <int W2, int I2, bool S2> struct RType {
    enum {
      F2 = W2 - I2,
      mult_w = W + W2,
      mult_i = I + I2,
      mult_s = S || S2,
      plus_w = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)) + 1 + ap_detail::max(F, F2),
      plus_i = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)) + 1,
      plus_s = S || S2,
      minus_w = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)) + 1 + ap_detail::max(F, F2),
      minus_i = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)) + 1,
      minus_s = true,
      div_w = S2 + W + ap_detail::max(F2, 0),
      div_i = S2 + I + F2,
      div_s = S || S2,
      logic_w = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)) + ap_detail::max(F, F2),
      logic_i = ap_detail::max(I + (S2 && !S), I2 + (S && !S2)),
      logic_s = S || S2
    };
    typedef ap_fixed_base<W, I, S> lhs;
    typedef ap_fixed_base<W2, I2, S2> rhs;
    typedef ap_fixed_base<mult_w, mult_i, mult_s> mult_base;
    typedef ap_fixed_base<plus_w, plus_i, plus_s> plus_base;
    typedef ap_fixed_base<minus_w, minus_i, minus_s> minus_base;
    typedef ap_fixed_base<logic_w, logic_i, logic_s> logic_base;
    typedef ap_fixed_base<div_w, div_i, div_s> div_base;
    typedef ap_fixed_base<W, I, S> arg1_base;
    typedef typename _ap_fixed_factory<mult_w, mult_i, mult_s>::type mult;
    typedef typename _ap_fixed_factory<plus_w, plus_i, plus_s>::type plus;
    typedef typename _ap_fixed_factory<minus_w, minus_i, minus_s>::type minus;
    typedef typename _ap_fixed_factory<logic_w, logic_i, logic_s>::type logic;
    typedef typename _ap_fixed_factory<div_w, div_i, div_s>::type div;
    typedef typename _ap_fixed_factory<W, I, S>::type arg1;
  };

  // ---- the conversion core -------------------------------------------------
  // from_raw: a value given as the W2-bit raw integer v2 with F2 fractional
  // bits becomes this type's raw: align to F (quantizing with Q when bits are
  // dropped), then apply the overflow mode O. The exact value is kept in a
  // signed type wide enough for the shift, so AP_SAT is an exact clamp.
  template <int W2, bool S2>
  static constexpr raw_t from_raw(typename ap_detail::bits<W2, S2>::vtype v2, int F2) {
    const int CW = W2 + W + 3;
    typedef ap_detail::sbits<CW> wide;
    wide v = (wide)v2;
    wide q = 0;
    bool big = false; // |value| >= 2^W after the shift: out of range unless 0
    if (F2 > F) {
      int sh = F2 - F;
      bool qb = false, r = false, s = v < 0;
      if (sh >= CW) {
        q = s ? (wide)-1 : (wide)0;
        qb = s;
        r = v != 0;
      } else {
        q = v >> sh;
        qb = ((v >> (sh - 1)) & 1) != 0;
        r = sh > 1 ? ((v & (((wide)1 << (sh - 1)) - 1)) != 0) : false;
      }
      bool add = false;
      if (Q == AP_TRN) add = false;
      else if (Q == AP_RND) add = qb;
      else if (Q == AP_RND_ZERO) add = qb && (s || r);
      else if (Q == AP_RND_MIN_INF) add = qb && r;
      else if (Q == AP_RND_INF) add = qb && (!s || r);
      else if (Q == AP_RND_CONV) add = qb && (((q & 1) != 0) || r);
      else /* AP_TRN_ZERO */ add = s && (qb || r);
      q += add;
    } else if (F2 < F) {
      int sh = F - F2;
      if (v == 0) q = 0;
      else if (sh >= W + 1) { big = true; q = v < 0 ? (wide)-1 : (wide)1; }
      else q = v << sh;
    } else {
      q = v;
    }
    if (O == AP_WRAP) return big ? (raw_t)0 : (raw_t)q;
    const wide mn = (wide)ap_detail::smin<W, S>(), mx = (wide)ap_detail::smax<W, S>();
    bool over = (big && q > 0) || q > mx;
    bool under = (big && q < 0) || q < mn;
    if (!over && !under) return (raw_t)q;
    if (O == AP_SAT) return over ? (raw_t)mx : (raw_t)mn;
    if (O == AP_SAT_ZERO) return (raw_t)0;
    if (O == AP_SAT_SYM) return over ? (raw_t)mx : (S ? (raw_t)(-mx) : (raw_t)0);
    return big ? (raw_t)0 : (raw_t)q; // AP_WRAP_SM: not modelled, plain wrap
  }
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  static constexpr raw_t from(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) { return from_raw<W2, S2>(o.rval(), W2 - I2); }
  // double -> fixed: the IEEE mantissa (negated if negative) has 52 - exp
  // fractional bits; NaN/inf are not special-cased (Vitis warns and goes on).
  static constexpr raw_t from_double(double d) {
    unsigned long long b = __builtin_bit_cast(unsigned long long, d);
    bool neg = (b >> 63) != 0;
    int e = (int)((b >> 52) & 0x7ff);
    unsigned long long m = b & ((1ull << 52) - 1);
    if (e == 0 && m == 0) return (raw_t)0;
    m |= 1ull << 52;
    ap_detail::sbits<55> man = (ap_detail::sbits<55>)m;
    if (neg) man = -man;
    return from_raw<55, true>(man, 52 - (e - 1023));
  }

  // ---- constructors --------------------------------------------------------
  ap_fixed_base() = default;
  ap_fixed_base(const ap_fixed_base &) = default;
  ap_fixed_base &operator=(const ap_fixed_base &) = default;
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr ap_fixed_base(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) : V(from(o)) {}
  template <int W2, bool S2>
  constexpr ap_fixed_base(const ap_int_base<W2, S2> &o) : V(from_raw<W2, S2>(o.rval(), 0)) {}
  template <class T, class = typename ap_detail::ctype<T>::type>
  constexpr ap_fixed_base(T v)
      : V(from_raw<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(
            (typename ap_detail::bits<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>::vtype)v, 0)) {}
  constexpr ap_fixed_base(double d) : V(from_double(d)) {}
  constexpr ap_fixed_base(float d) : V(from_double((double)d)) {}
  constexpr ap_fixed_base(_Float16 d) : V(from_double((double)d)) {}

  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr ap_fixed_base &operator=(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) { V = from(o); return *this; }
  template <int W2, bool S2>
  constexpr ap_fixed_base &operator=(const ap_int_base<W2, S2> &o) { V = from_raw<W2, S2>(o.rval(), 0); return *this; }
  template <class T, class = typename ap_detail::ctype<T>::type>
  constexpr ap_fixed_base &operator=(T v) { V = ap_fixed_base(v).V; return *this; }
  constexpr ap_fixed_base &operator=(double d) { V = from_double(d); return *this; }
  constexpr ap_fixed_base &operator=(float d) { V = from_double((double)d); return *this; }

  // ---- conversions ---------------------------------------------------------
  constexpr double to_double() const { return (double)rval() * ap_detail::pow2<-F>::v; }
  constexpr operator double() const { return to_double(); }
  // The other implicit conversions Vitis provides (ap_fixed_base.h:1184-1217):
  // the integer ones truncate toward zero through to_int() and friends. With
  // them, `x * 0.5` is ambiguous - as with the real headers.
  constexpr operator float() const { return to_float(); }
  constexpr operator _Float16() const { return to_half(); }
  constexpr operator bool() const { return V != 0; }
  constexpr operator char() const { return (char)to_int(); }
  constexpr operator signed char() const { return (signed char)to_int(); }
  constexpr operator unsigned char() const { return (unsigned char)to_uint(); }
  constexpr operator short() const { return (short)to_int(); }
  constexpr operator unsigned short() const { return (unsigned short)to_uint(); }
  constexpr operator int() const { return to_int(); }
  constexpr operator unsigned() const { return to_uint(); }
  constexpr operator long() const { return (long)to_int64(); }
  constexpr operator unsigned long() const { return (unsigned long)to_uint64(); }
  constexpr operator long long() const { return to_int64(); }
  constexpr operator unsigned long long() const { return to_uint64(); }
  constexpr float to_float() const { return (float)to_double(); }
  constexpr _Float16 to_half() const { return (_Float16)to_float(); }
  // integer part, as Vitis to_ap_int_base(): truncated toward zero (see ipart()).
  constexpr ap_int_base<ap_detail::max(I, 1), S> to_ap_int_base() const {
    ap_int_base<ap_detail::max(I, 1), S> r;
    r.V = (typename ap_int_base<ap_detail::max(I, 1), S>::raw_t)ipart();
    return r;
  }
  constexpr ap_int<ap_detail::max(I, 1)> to_ap_int() const { return ap_int<ap_detail::max(I, 1)>(to_ap_int_base()); }
  constexpr char to_char() const { return (char)ipart(); }
  constexpr short to_short() const { return (short)ipart(); }
  constexpr int to_int() const { return (int)ipart(); }
  constexpr unsigned to_uint() const { return (unsigned)ipart(); }
  constexpr long to_long() const { return (long)ipart(); }
  constexpr unsigned long to_ulong() const { return (unsigned long)ipart(); }
  constexpr ap_slong to_int64() const { return (ap_slong)ipart(); }
  constexpr ap_ulong to_uint64() const { return (ap_ulong)ipart(); }

  constexpr int length() const { return W; }
  constexpr int wl() const { return W; }
  constexpr int iwl() const { return I; }
  constexpr bool is_zero() const { return V == 0; }
  constexpr bool is_neg() const { return S && rval() < 0; }
  constexpr bool sign() const { return S && rval() < 0; }

  // ---- bit and slice selects on the raw bits (value-returning) -------------
  constexpr bool operator[](int i) const { return (((ap_detail::ubits<W>)V >> i) & 1) != 0; }
  constexpr bool bit(int i) const { return (*this)[i]; }
  constexpr bool get_bit(int i) const { return (*this)[i]; }
  constexpr void set_bit(int i, bool b) {
    ap_detail::ubits<W> m = (ap_detail::ubits<W>)1 << i;
    V = (raw_t)(b ? ((ap_detail::ubits<W>)V | m) : ((ap_detail::ubits<W>)V & (ap_detail::ubits<W>)~m));
  }
  constexpr void set(int i) { set_bit(i, true); }
  constexpr void set(int i, bool b) { set_bit(i, b); }
  constexpr void clear(int i) { set_bit(i, false); }
  constexpr const ap_int_base<W, false> range(int hi, int lo) const {
    ap_int_base<W, false> r;
    r.V = (ap_detail::ubits<W>)V;
    return r.range(hi, lo);
  }
  constexpr const ap_int_base<W, false> range() const { ap_int_base<W, false> r; r.V = (ap_detail::ubits<W>)V; return r; }
  constexpr const ap_int_base<W, false> operator()(int hi, int lo) const { return range(hi, lo); }
  constexpr const ap_int_base<W, false> range(int hi, int lo) { return ((const ap_fixed_base *)this)->range(hi, lo); }
  constexpr const ap_int_base<W, false> operator()(int hi, int lo) { return range(hi, lo); }

  // ---- unary ---------------------------------------------------------------
  constexpr typename _ap_fixed_factory<W + 1, I + 1, true>::type operator-() const {
    ap_fixed_base<W + 1, I + 1, true> r;
    r.V = (typename ap_fixed_base<W + 1, I + 1, true>::raw_t)(-(ap_detail::sbits<W + 2>)rval());
    return typename _ap_fixed_factory<W + 1, I + 1, true>::type(r);
  }
  constexpr typename _ap_fixed_factory<W, I, S>::type operator+() const { return typename _ap_fixed_factory<W, I, S>::type(*this); }
  constexpr bool operator!() const { return V == 0; }
  constexpr ap_fixed_base<W, I, S> operator~() const {
    ap_fixed_base<W, I, S> r;
    r.V = (raw_t)~V;
    return r;
  }
  constexpr ap_fixed_base &operator++() { return *this += 1; }
  constexpr ap_fixed_base &operator--() { return *this -= 1; }
  constexpr const ap_fixed_base operator++(int) { ap_fixed_base t(*this); ++*this; return t; }
  constexpr const ap_fixed_base operator--(int) { ap_fixed_base t(*this); --*this; return t; }

  // ---- shifts on the raw bits, width W kept --------------------------------
  constexpr ap_fixed_base operator<<(unsigned sh) const {
    ap_fixed_base r;
    r.V = sh < (unsigned)W ? (raw_t)((ap_detail::ubits<W>)V << sh) : (raw_t)0;
    return r;
  }
  constexpr ap_fixed_base operator>>(unsigned sh) const {
    ap_fixed_base r;
    if (sh < (unsigned)W) r.V = (raw_t)(rval() >> sh);
    else r.V = (S && rval() < 0) ? (raw_t)-1 : (raw_t)0;
    return r;
  }
  constexpr ap_fixed_base operator<<(int sh) const { return sh < 0 ? operator>>((unsigned)-sh) : operator<<((unsigned)sh); }
  constexpr ap_fixed_base operator>>(int sh) const { return sh < 0 ? operator<<((unsigned)-sh) : operator>>((unsigned)sh); }
  template <int W2, bool S2>
  constexpr ap_fixed_base operator<<(const ap_int_base<W2, S2> &o) const { return operator<<((int)o.to_int64()); }
  template <int W2, bool S2>
  constexpr ap_fixed_base operator>>(const ap_int_base<W2, S2> &o) const { return operator>>((int)o.to_int64()); }
  template <class T> constexpr ap_fixed_base &operator<<=(const T &sh) { *this = operator<<(sh); return *this; }
  template <class T> constexpr ap_fixed_base &operator>>=(const T &sh) { *this = operator>>(sh); return *this; }

  // ---- binary arithmetic with another fixed-point value -------------------
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr typename RType<W2, I2, S2>::mult operator*(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    typename RType<W2, I2, S2>::mult_base r;
    typedef typename RType<W2, I2, S2>::mult_base::raw_t mr;
    r.V = (mr)((mr)rval() * (mr)o.rval()); // W + W2 bits hold the product exactly
    return typename RType<W2, I2, S2>::mult(r);
  }
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr typename RType<W2, I2, S2>::div operator/(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    typedef typename RType<W2, I2, S2>::div_base R;
    const int F2 = W2 - I2;
    typedef ap_detail::sbits<ap_detail::max(W + ap_detail::max(F2, 0), W2) + 2> wide;
    R r;
    r.V = (typename R::raw_t)(((wide)rval() << ap_detail::max(F2, 0)) / (wide)o.rval()); // truncation toward zero
    return typename RType<W2, I2, S2>::div(r);
  }
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr typename RType<W2, I2, S2>::plus operator+(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    typedef typename RType<W2, I2, S2>::plus_base R;
    const int F2 = W2 - I2, F3 = ap_detail::max(F, F2);
    typedef ap_detail::sbits<ap_detail::max(W + (F3 - F), W2 + (F3 - F2)) + 2> wide;
    R r;
    r.V = (typename R::raw_t)(((wide)rval() << (F3 - F)) + ((wide)o.rval() << (F3 - F2)));
    return typename RType<W2, I2, S2>::plus(r);
  }
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr typename RType<W2, I2, S2>::minus operator-(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    typedef typename RType<W2, I2, S2>::minus_base R;
    const int F2 = W2 - I2, F3 = ap_detail::max(F, F2);
    typedef ap_detail::sbits<ap_detail::max(W + (F3 - F), W2 + (F3 - F2)) + 2> wide;
    R r;
    r.V = (typename R::raw_t)(((wide)rval() << (F3 - F)) - ((wide)o.rval() << (F3 - F2)));
    return typename RType<W2, I2, S2>::minus(r);
  }
  // Bitwise: both operands converted to the logic type first (Vitis OP_BIN_AF).
#define HLSEST_AF_LOGIC(SYM)                                                                 \
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>                     \
  constexpr typename RType<W2, I2, S2>::logic operator SYM(                                  \
      const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {                                \
    typedef typename RType<W2, I2, S2>::logic_base R;                                        \
    R l(*this), rr(o), r;                                                                    \
    r.V = (typename R::raw_t)(l.V SYM rr.V);                                                 \
    return typename RType<W2, I2, S2>::logic(r);                                             \
  }
  HLSEST_AF_LOGIC(&)
  HLSEST_AF_LOGIC(|)
  HLSEST_AF_LOGIC(^)
#undef HLSEST_AF_LOGIC

  // ---- comparisons: operands aligned to the larger F (the narrower-F side
  // widened, as Vitis OP_CMP_AF), then compared as bitwidth integers with
  // the C conversion rule of ap_detail::cmp_lt --------------------------------
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr bool lt_(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    const int F2 = W2 - I2;
    const int WL = F >= F2 ? W : ap_detail::max(W + F2 - F + 1, 1);
    const int WR = F > F2 ? ap_detail::max(W2 + F - F2, 1) : W2;
    const int F3 = ap_detail::max(F, F2);
    const int MW = ap_detail::max(WL, WR) + 1;
    typedef ap_detail::sbits<MW> sw;
    return ap_detail::cmp_lt<WL, S, WR, S2, MW>((sw)((sw)rval() << (F3 - F)), (sw)((sw)o.rval() << (F3 - F2)));
  }
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr bool eq_(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {
    const int F2 = W2 - I2;
    const int WL = F >= F2 ? W : ap_detail::max(W + F2 - F + 1, 1);
    const int WR = F > F2 ? ap_detail::max(W2 + F - F2, 1) : W2;
    const int F3 = ap_detail::max(F, F2);
    const int MW = ap_detail::max(WL, WR) + 1;
    typedef ap_detail::sbits<MW> sw;
    return ap_detail::cmp_eq<WL, S, WR, S2, MW>((sw)((sw)rval() << (F3 - F)), (sw)((sw)o.rval() << (F3 - F2)));
  }
#define HLSEST_AF_CMP(SYM, EXPR)                                                             \
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>                     \
  constexpr bool operator SYM(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o) const {        \
    return EXPR;                                                                             \
  }                                                                                          \
  constexpr bool operator SYM(double d) const { return to_double() SYM d; }
  HLSEST_AF_CMP(<, lt_(o))
  HLSEST_AF_CMP(>, o.lt_(*this))
  HLSEST_AF_CMP(<=, !o.lt_(*this))
  HLSEST_AF_CMP(>=, !lt_(o))
  HLSEST_AF_CMP(==, eq_(o))
  HLSEST_AF_CMP(!=, !eq_(o))
#undef HLSEST_AF_CMP

  // ---- compound assignment: the binary operator, then this type's
  // quantization / overflow (Vitis OP_ASSIGN_AF) -----------------------------
  template <class T> constexpr ap_fixed_base &operator*=(const T &o) { *this = (*this) * o; return *this; }
  template <class T> constexpr ap_fixed_base &operator/=(const T &o) { *this = (*this) / o; return *this; }
  template <class T> constexpr ap_fixed_base &operator+=(const T &o) { *this = (*this) + o; return *this; }
  template <class T> constexpr ap_fixed_base &operator-=(const T &o) { *this = (*this) - o; return *this; }
  template <class T> constexpr ap_fixed_base &operator&=(const T &o) { *this = (*this) & o; return *this; }
  template <class T> constexpr ap_fixed_base &operator|=(const T &o) { *this = (*this) | o; return *this; }
  template <class T> constexpr ap_fixed_base &operator^=(const T &o) { *this = (*this) ^ o; return *this; }

private:
  // The integer part as Vitis to_ap_int_base(): the I bits above F (floor),
  // then +1 for a negative value with non-zero fraction bits, i.e. truncation
  // toward zero like a C double -> int conversion.
  constexpr ap_detail::sbits<ap_detail::max(W, I) + 2> ipart() const {
    typedef ap_detail::sbits<ap_detail::max(W, I) + 2> wide;
    const int shr = F >= 0 ? (F < W + 2 ? F : 0) : 0, shl = F < 0 ? -F : 0;
    if (I <= 0) return (S && rval() < 0) ? (wide)-1 : (wide)0; // 1-bit field: 0, or 0 + 1 wrapped
    wide r = (F >= 0 && F >= W + 2) ? (wide)((S && rval() < 0) ? -1 : 0) : (wide)(((wide)rval() >> shr) << shl);
    if (S && rval() < 0 && F > 0) {
      bool frac = F >= W ? (V != 0) : (((ap_detail::ubits<W>)V & (ap_detail::ubits<W>)((((ap_detail::ubits<W + 1>)1) << F) - 1)) != 0);
      if (frac) r += 1;
    }
    return r;
  }
};

// ---- ap_int_base from ap_fixed_base (declared in ap_int.h) -----------------
template <int W, bool S>
template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
constexpr ap_int_base<W, S>::ap_int_base(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o)
    : V((raw_t)o.to_ap_int_base().rval()) {}

// ---- mixed operators: C integer -> ap_fixed_base<ctype::W, ctype::W, ctype::S>,
// ap_int_base<W2,S2> -> ap_fixed_base<W2,W2,S2> (Vitis AF_BIN_OP_WITH_INT /
// AF_BIN_OP_WITH_AP_INT). There is deliberately no operator with double:
// those go through operator double(), as with the Vitis headers. -------------
#define HLSEST_AF_MIX(SYM, RTY)                                                                      \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, class T>                          \
  constexpr typename ap_fixed_base<W, I, S>::template RType<ap_detail::ctype<T>::W,                  \
                                                             ap_detail::ctype<T>::W,                 \
                                                             ap_detail::ctype<T>::S>::RTY            \
  operator SYM(const ap_fixed_base<W, I, S, Q, O, N> &a, T b) {                                      \
    return a SYM ap_fixed_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(b); \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, class T>                          \
  constexpr typename ap_fixed_base<W, I, S>::template RType<ap_detail::ctype<T>::W,                  \
                                                             ap_detail::ctype<T>::W,                 \
                                                             ap_detail::ctype<T>::S>::RTY            \
  operator SYM(T a, const ap_fixed_base<W, I, S, Q, O, N> &b) {                                      \
    return ap_fixed_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(a) SYM b; \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>                  \
  constexpr typename ap_fixed_base<W, I, S>::template RType<W2, W2, S2>::RTY operator SYM(           \
      const ap_fixed_base<W, I, S, Q, O, N> &a, const ap_int_base<W2, S2> &b) {                      \
    return a SYM ap_fixed_base<W2, W2, S2>(b);                                                       \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>                  \
  constexpr typename ap_fixed_base<W2, W2, S2>::template RType<W, I, S>::RTY operator SYM(           \
      const ap_int_base<W2, S2> &a, const ap_fixed_base<W, I, S, Q, O, N> &b) {                      \
    return ap_fixed_base<W2, W2, S2>(a) SYM b;                                                       \
  }
HLSEST_AF_MIX(+, plus)
HLSEST_AF_MIX(-, minus)
HLSEST_AF_MIX(*, mult)
HLSEST_AF_MIX(/, div)
HLSEST_AF_MIX(&, logic)
HLSEST_AF_MIX(|, logic)
HLSEST_AF_MIX(^, logic)
#undef HLSEST_AF_MIX
#define HLSEST_AF_MIXREL(SYM)                                                                        \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, class T,                          \
            class = typename ap_detail::ctype<T>::type>                                              \
  constexpr bool operator SYM(const ap_fixed_base<W, I, S, Q, O, N> &a, T b) {                       \
    return a SYM ap_fixed_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(b); \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, class T,                          \
            class = typename ap_detail::ctype<T>::type>                                              \
  constexpr bool operator SYM(T a, const ap_fixed_base<W, I, S, Q, O, N> &b) {                       \
    return ap_fixed_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(a) SYM b; \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>                  \
  constexpr bool operator SYM(const ap_fixed_base<W, I, S, Q, O, N> &a, const ap_int_base<W2, S2> &b) { \
    return a SYM ap_fixed_base<W2, W2, S2>(b);                                                       \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>                  \
  constexpr bool operator SYM(const ap_int_base<W2, S2> &a, const ap_fixed_base<W, I, S, Q, O, N> &b) { \
    return ap_fixed_base<W2, W2, S2>(a) SYM b;                                                       \
  }                                                                                                  \
  template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N>                                   \
  constexpr bool operator SYM(double a, const ap_fixed_base<W, I, S, Q, O, N> &b) {                  \
    return a SYM b.to_double();                                                                      \
  }
HLSEST_AF_MIXREL(<)
HLSEST_AF_MIXREL(>)
HLSEST_AF_MIXREL(<=)
HLSEST_AF_MIXREL(>=)
HLSEST_AF_MIXREL(==)
HLSEST_AF_MIXREL(!=)
#undef HLSEST_AF_MIXREL

// Concatenation of the raw bits with an ap_int or another ap_fixed.
template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>
constexpr const ap_uint<W + W2> operator,(const ap_fixed_base<W, I, S, Q, O, N> &a, const ap_int_base<W2, S2> &b) {
  return (a.range(), b);
}
template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, bool S2>
constexpr const ap_uint<W + W2> operator,(const ap_int_base<W2, S2> &a, const ap_fixed_base<W, I, S, Q, O, N> &b) {
  return (a, b.range());
}
template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N, int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
constexpr const ap_uint<W + W2> operator,(const ap_fixed_base<W, I, S, Q, O, N> &a, const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &b) {
  return (a.range(), b.range());
}

template <int W, int I, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP, int N = 0>
struct ap_fixed : ap_fixed_base<W, I, true, Q, O, N> {
  typedef ap_fixed_base<W, I, true, Q, O, N> Base;
  ap_fixed() = default;
  ap_fixed(const ap_fixed &) = default;
  ap_fixed &operator=(const ap_fixed &) = default;
  template <class T> constexpr ap_fixed(const T &v) : Base(v) {}
  template <class T> constexpr ap_fixed &operator=(const T &v) { this->V = ap_fixed(v).V; return *this; }
};
template <int W, int I, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP, int N = 0>
struct ap_ufixed : ap_fixed_base<W, I, false, Q, O, N> {
  typedef ap_fixed_base<W, I, false, Q, O, N> Base;
  ap_ufixed() = default;
  ap_ufixed(const ap_ufixed &) = default;
  ap_ufixed &operator=(const ap_ufixed &) = default;
  template <class T> constexpr ap_ufixed(const T &v) : Base(v) {}
  template <class T> constexpr ap_ufixed &operator=(const T &v) { this->V = ap_ufixed(v).V; return *this; }
};

#ifdef __clang__
#pragma clang attribute pop
#endif

#endif // __cplusplus
#endif
