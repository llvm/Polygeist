// hls-est "exactclass" stand-in for the Vitis HLS ap_int.h.
//
// ap_int<W> / ap_uint<W> are classes whose ONLY data member is a
// `_BitInt(W)` / `unsigned _BitInt(W)` (ap_int_base<W,S>::V), exactly like
// the Vitis headers in __SYNTHESIS__ mode (`int __attribute__((bitwidth(W)))`).
// cgeist (Polygeist branch hls-est-vitis-match, --scalar-wrapper-abi) lowers a
// single-scalar-member class to that scalar, so every ap_int<W> value, array
// element, reference and `this` is a plain `iW` in the MLIR.
//
// Semantics follow etc/ap_int_base.h of Vitis 2025.1:
//   * result widths of + - * / % & | ^ per ap_int_base::RType (the wider
//     Vitis result type), computed exactly; assigning back truncates (wrap);
//   * shifts keep the width of the left operand (bits shifted out are lost);
//   * conversion from float truncates toward zero, then wraps to W bits;
//     conversion from double does the same except that, as in Vitis'
//     ap_int_base(double), the shifted mantissa of a value with
//     1 <= |d| < 2^52 passes through a 25-bit signed integer
//     (ap_int_base<FLOAT_MAN + 2, true>): ap_int<32>(1073741824.0) is 0,
//     ap_uint<40>(4294967295.0) is 0xFFFFFFFFFF (g++ and csynth agree);
//   * the implicit conversion to a built-in integer is the 64-bit one Vitis
//     provides (`operator RetType()`): long long / unsigned long long;
//   * range() / operator() / operator[] / bit() / (a, b) are VALUE-returning
//     (const): `x.range(7, 0) = v` does not compile (Vitis uses reference
//     proxies there; see tools/hls-est-validate/OPEN_ISSUES.md).
// clang rejects a signed _BitInt(1), so the signed 1-bit types (ap_int<1>,
// ap_fixed<1,I>) keep their bit in an `unsigned _BitInt(1)` (an i1 in the
// MLIR, as wide as in Vitis) and every value read goes through rval(), which
// sign-extends it (1 reads as -1). Known deviation: string constructors are
// not provided.
#ifndef HLSEST_EXACTCLASS_AP_INT_H
#define HLSEST_EXACTCLASS_AP_INT_H
#ifdef __cplusplus
// Vitis marks every ap_int / ap_fixed member and operator AP_INLINE, i.e.
// `inline __attribute__((always_inline))` under __SYNTHESIS__
// (etc/ap_common.h): clang inlines them before Vitis's own inliner, which
// never lists them ([HLS 214-178]). cgeist's automatic-inlining rule
// (--hls-auto-inline, VITIS_NOTES N18) reads the same attribute.
#ifdef __clang__
#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#endif

enum ap_q_mode { AP_RND, AP_RND_ZERO, AP_RND_MIN_INF, AP_RND_INF, AP_RND_CONV, AP_TRN, AP_TRN_ZERO };
enum ap_o_mode { AP_SAT, AP_SAT_ZERO, AP_SAT_SYM, AP_WRAP, AP_WRAP_SM };

typedef long long ap_slong;
typedef unsigned long long ap_ulong;

namespace ap_detail {
constexpr int max(int a, int b) { return a > b ? a : b; }
constexpr int min(int a, int b) { return a < b ? a : b; }

// bits<N, S>::type is the raw STORAGE type of an N-bit value (the one data
// member V); bits<N, S>::vtype the type of its value and val() the read.
// A signed _BitInt needs >= 2 bits, so a signed 1-bit value is stored in an
// unsigned _BitInt(1) and read as -bit (0 or -1).
template <int N, bool S> struct bits {
  typedef _BitInt((N < 2 ? 2 : N)) type;
  typedef type vtype;
  static constexpr vtype val(type v) { return v; }
};
template <> struct bits<1, true> {
  typedef unsigned _BitInt(1) type;
  typedef _BitInt(2) vtype;
  static constexpr vtype val(type v) { return -(vtype)v; }
};
template <int N> struct bits<N, false> {
  typedef unsigned _BitInt((N < 1 ? 1 : N)) type;
  typedef type vtype;
  static constexpr vtype val(type v) { return v; }
};
// Computation types (any width; >= 2 bits signed, >= 1 bit unsigned).
template <int N> using sbits = _BitInt((N < 2 ? 2 : N));
template <int N> using ubits = unsigned _BitInt((N < 1 ? 1 : N));

// Width and signedness Vitis assigns to each C integer type in mixed
// expressions (ap_int_base.h ALL_OP_BIN_WITH_INT, ap_fixed_base.h
// ALL_AF_OP_WITH_INT). The primary template is empty so that
// `typename ctype<T>::...` is a substitution failure for any other T.
template <class T> struct ctype {};
template <> struct ctype<bool> { static constexpr int W = 1; static constexpr bool S = false; typedef bool type; };
template <> struct ctype<char> { static constexpr int W = 8; static constexpr bool S = ((char)-1 < 0); typedef char type; };
template <> struct ctype<signed char> { static constexpr int W = 8; static constexpr bool S = true; typedef signed char type; };
template <> struct ctype<unsigned char> { static constexpr int W = 8; static constexpr bool S = false; typedef unsigned char type; };
template <> struct ctype<short> { static constexpr int W = 16; static constexpr bool S = true; typedef short type; };
template <> struct ctype<unsigned short> { static constexpr int W = 16; static constexpr bool S = false; typedef unsigned short type; };
template <> struct ctype<int> { static constexpr int W = 32; static constexpr bool S = true; typedef int type; };
template <> struct ctype<unsigned> { static constexpr int W = 32; static constexpr bool S = false; typedef unsigned type; };
template <> struct ctype<long> { static constexpr int W = 64; static constexpr bool S = true; typedef long type; };
template <> struct ctype<unsigned long> { static constexpr int W = 64; static constexpr bool S = false; typedef unsigned long type; };
template <> struct ctype<long long> { static constexpr int W = 64; static constexpr bool S = true; typedef long long type; };
template <> struct ctype<unsigned long long> { static constexpr int W = 64; static constexpr bool S = false; typedef unsigned long long type; };

// 2^E as a double, computed at compile time (a static constexpr member, so
// the frontend folds it to a constant; __builtin_ldexp would be a libc call).
constexpr double pow2_eval(int e) {
  double r = 1.0;
  for (int i = 0; i < (e < 0 ? -e : e); i++) r = e < 0 ? r * 0.5 : r * 2.0;
  return r;
}
template <int E> struct pow2 { static constexpr double v = pow2_eval(E); };
template <bool S> struct rettype { typedef long long type; };
template <> struct rettype<false> { typedef unsigned long long type; };

// Conversion of a double holding an exactly converted FLOAT to a W-bit
// integer, as ap_int_base(float): decode the IEEE mantissa, shift it by the
// exponent (bits shifted out of W are lost), then negate in W bits, i.e.
// truncation toward zero, then wrap. NaN/inf: Vitis warns and proceeds with
// the raw bits; we do the same.
template <int W, bool S>
constexpr typename bits<W, S>::type float_to_int(double d) {
  typedef typename bits<W, S>::type raw_t;
  typedef ubits<(W > 64 ? W : 64)> uw;
  unsigned long long b = __builtin_bit_cast(unsigned long long, d);
  bool neg = (b >> 63) != 0;
  int e = (int)((b >> 52) & 0x7ff);
  unsigned long long m = b & ((1ull << 52) - 1);
  if (e == 0 && m == 0) return (raw_t)0;
  m |= 1ull << 52;
  int exp = e - 1023;
  uw mag = 0;
  if (exp < 52) {
    int sh = 52 - exp;
    mag = sh < 54 ? (uw)(m >> sh) : (uw)0;
  } else {
    int sh = exp - 52;
    mag = sh < W ? (uw)((uw)m << sh) : (uw)0;
  }
  ubits<W> r = (ubits<W>)mag;
  return (raw_t)(neg ? (ubits<W>)-r : r);
}
// ap_int_base(double) (etc/ap_int_base.h, Vitis 2025.1): as float_to_int,
// except that for 0 < 52 - exp < 54 (|d| < 2^52) the shifted mantissa is
// stored in an ap_int_base<FLOAT_MAN + 2, true>, a 25-bit SIGNED integer,
// before it is assigned (sign-extended) to the W-bit result. A value with
// 2^24 <= |d| < 2^52 therefore keeps only its low 25 bits, read as signed:
// ap_int<32>(1073741824.0) is 0, ap_int<32>(16777219.0) is -16777213,
// ap_uint<40>(4294967295.0) is 0xFFFFFFFFFF. Confirmed with g++ against the
// real headers and with csynth (a.o.3.ll stores the same constants).
template <int W, bool S>
constexpr typename bits<W, S>::type double_to_int(double d) {
  typedef typename bits<W, S>::type raw_t;
  unsigned long long b = __builtin_bit_cast(unsigned long long, d);
  int e = (int)((b >> 52) & 0x7ff);
  int sh = 52 - (e - 1023);
  if (((b << 1) >> 1) == 0 || sh <= 0 || sh >= 54) return float_to_int<W, S>(d);
  bool neg = (b >> 63) != 0;
  unsigned long long m = (b & ((1ull << 52) - 1)) | (1ull << 52);
  unsigned long long t = (m >> sh) & 0x1ffffffull; // ap_int_base<25, true>
  long long v = (t & 0x1000000ull) ? (long long)t - 0x2000000ll : (long long)t;
  ubits<W> r = (ubits<W>)(sbits<(W > 64 ? W : 64)>)v; // sign-extend, then wrap
  return (raw_t)(neg ? (ubits<W>)-r : r);
}

template <int W, bool S> constexpr sbits<W + 1> smin() { return S ? (sbits<W + 1>)(-((sbits<W + 1>)1 << (W - 1))) : (sbits<W + 1>)0; }
template <int W, bool S> constexpr sbits<W + 1> smax() { return S ? (sbits<W + 1>)(((sbits<W + 1>)1 << (W - 1)) - 1) : (sbits<W + 1>)(((sbits<W + 1>)1 << W) - 1); }
} // namespace ap_detail

namespace ap_detail {
// Vitis compares two bitwidth integers with the C usual arithmetic
// conversions (etc/ap_private.h operator<): both narrower than int -> signed;
// same signedness -> that; mixed -> unsigned when the unsigned operand is at
// least as wide as the signed one, else signed. Operands arrive extended to
// MW bits by their own signedness.
template <int W1, bool S1, int W2, bool S2, int MW>
constexpr bool cmp_lt(sbits<MW> l, sbits<MW> r) {
  const bool sgn = (S1 == S2) ? S1 : ((W1 < 32 && W2 < 32) ? true : (S1 ? !(W2 >= W1) : !(W1 >= W2)));
  return sgn ? (l < r) : ((ubits<MW>)l < (ubits<MW>)r);
}
template <int W1, bool S1, int W2, bool S2, int MW>
constexpr bool cmp_eq(sbits<MW> l, sbits<MW> r) { return l == r; }
} // namespace ap_detail

template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N> struct ap_fixed_base;
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N> struct ap_fixed;
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N> struct ap_ufixed;
template <int W> struct ap_int;
template <int W> struct ap_uint;

template <int W, bool S> struct _ap_int_factory { typedef ap_int<W> type; };
template <int W> struct _ap_int_factory<W, false> { typedef ap_uint<W> type; };

template <int W, bool S> struct ap_int_base {
  static_assert(W > 0, "ap_int_base bitwidth must be positive");
  typedef typename ap_detail::bits<W, S>::type raw_t;
  static const int width = W;
  static const bool sign = S;

  raw_t V; // the only data member (uninitialised by default, like Vitis)

  /// The value of V (sign-extends the bit of a signed 1-bit type).
  typedef typename ap_detail::bits<W, S>::vtype val_t;
  constexpr val_t rval() const { return ap_detail::bits<W, S>::val(V); }

  /// Result types of the binary operators (ap_int_base.h RType).
  template <int W2, bool S2> struct RType {
    enum {
      mult_w = W + W2,
      mult_s = S || S2,
      plus_w = ap_detail::max(W + (S2 && !S), W2 + (S && !S2)) + 1,
      plus_s = S || S2,
      minus_w = ap_detail::max(W + (S2 && !S), W2 + (S && !S2)) + 1,
      minus_s = true,
      div_w = W + S2,
      div_s = S || S2,
      mod_w = ap_detail::min(W, W2 + (!S2 && S)),
      mod_s = S,
      logic_w = ap_detail::max(W + (S2 && !S), W2 + (S && !S2)),
      logic_s = S || S2
    };
    typedef ap_int_base<mult_w, mult_s> mult_base;
    typedef ap_int_base<plus_w, plus_s> plus_base;
    typedef ap_int_base<minus_w, minus_s> minus_base;
    typedef ap_int_base<logic_w, logic_s> logic_base;
    typedef ap_int_base<div_w, div_s> div_base;
    typedef ap_int_base<mod_w, mod_s> mod_base;
    typedef ap_int_base<W, S> arg1_base;
    typedef typename _ap_int_factory<mult_w, mult_s>::type mult;
    typedef typename _ap_int_factory<plus_w, plus_s>::type plus;
    typedef typename _ap_int_factory<minus_w, minus_s>::type minus;
    typedef typename _ap_int_factory<logic_w, logic_s>::type logic;
    typedef typename _ap_int_factory<div_w, div_s>::type div;
    typedef typename _ap_int_factory<mod_w, mod_s>::type mod;
    typedef typename _ap_int_factory<W, S>::type arg1;
    typedef bool reduce;
  };

  // ---- constructors (all trivial or constexpr) ----------------------------
  ap_int_base() = default;
  ap_int_base(const ap_int_base &) = default;
  ap_int_base &operator=(const ap_int_base &) = default;
  // from another width: sign/zero-extend by the SOURCE signedness, then wrap.
  template <int W2, bool S2>
  constexpr ap_int_base(const ap_int_base<W2, S2> &o) : V((raw_t)o.rval()) {}
  template <class T, class = typename ap_detail::ctype<T>::type>
  constexpr ap_int_base(T v) : V((raw_t)v) {}
  constexpr ap_int_base(double d) : V(ap_detail::double_to_int<W, S>(d)) {}
  constexpr ap_int_base(float d) : V(ap_detail::float_to_int<W, S>((double)d)) {}
  // from a fixed-point value: its integer part (truncated toward zero, as
  // Vitis to_ap_int_base()); defined in ap_fixed.h.
  template <int W2, int I2, bool S2, ap_q_mode Q2, ap_o_mode O2, int N2>
  constexpr ap_int_base(const ap_fixed_base<W2, I2, S2, Q2, O2, N2> &o);

  template <int W2, bool S2>
  constexpr ap_int_base &operator=(const ap_int_base<W2, S2> &o) { V = (raw_t)o.rval(); return *this; }
  template <class T, class = typename ap_detail::ctype<T>::type>
  constexpr ap_int_base &operator=(T v) { V = (raw_t)v; return *this; }

  // ---- conversions ---------------------------------------------------------
  // Vitis: `operator RetType()` with RetType the 64-bit C integer of the
  // same signedness (relied on for array subscripts and mixed arithmetic).
  constexpr operator typename ap_detail::rettype<S>::type() const {
    return (typename ap_detail::rettype<S>::type)rval();
  }
  constexpr bool to_bool() const { return V != 0; }
  constexpr char to_char() const { return (char)rval(); }
  constexpr signed char to_schar() const { return (signed char)rval(); }
  constexpr unsigned char to_uchar() const { return (unsigned char)rval(); }
  constexpr short to_short() const { return (short)rval(); }
  constexpr unsigned short to_ushort() const { return (unsigned short)rval(); }
  constexpr int to_int() const { return (int)rval(); }
  constexpr unsigned to_uint() const { return (unsigned)rval(); }
  constexpr long to_long() const { return (long)rval(); }
  constexpr unsigned long to_ulong() const { return (unsigned long)rval(); }
  constexpr ap_slong to_int64() const { return (ap_slong)rval(); }
  constexpr ap_ulong to_uint64() const { return (ap_ulong)rval(); }
  constexpr float to_float() const { return (float)rval(); }
  constexpr double to_double() const { return (double)rval(); }
  constexpr _Float16 to_half() const { return (_Float16)(float)rval(); }

  constexpr int length() const { return W; }
  constexpr bool iszero() const { return V == 0; }
  constexpr bool is_zero() const { return V == 0; }
  constexpr bool sign_bit() const { return ((ap_detail::ubits<W>)V >> (W - 1)) & 1; }

  // ---- bit and slice selects (value-returning) ----------------------------
  constexpr bool operator[](int i) const { return (((ap_detail::ubits<W>)V >> i) & 1) != 0; }
  constexpr bool bit(int i) const { return (*this)[i]; }
  constexpr bool get_bit(int i) const { return (*this)[i]; }
  constexpr bool test(int i) const { return (*this)[i]; }
  constexpr void set_bit(int i, bool b) {
    ap_detail::ubits<W> m = (ap_detail::ubits<W>)1 << i;
    V = (raw_t)(b ? ((ap_detail::ubits<W>)V | m) : ((ap_detail::ubits<W>)V & (ap_detail::ubits<W>)~m));
  }
  constexpr void set(int i) { set_bit(i, true); }
  constexpr void set(int i, bool b) { set_bit(i, b); }
  constexpr void clear(int i) { set_bit(i, false); }
  constexpr void invert(int i) { set_bit(i, !(*this)[i]); }
  // bits [hi:lo], zero-extended to W bits (what an ap_range_ref converts to).
  constexpr const ap_int_base<W, false> range(int hi, int lo) const {
    ap_int_base<W, false> r;
    int h = hi, l = lo;
    if (h < l) { h = lo; l = hi; }
    int n = h - l + 1;
    ap_detail::ubits<W> v = (ap_detail::ubits<W>)V >> l;
    if (n < W) v &= (ap_detail::ubits<W>)((((ap_detail::ubits<W + 1>)1) << n) - 1);
    if (hi < lo) { // reversed select
      ap_detail::ubits<W> o = 0;
      for (int k = 0; k < n; k++) o |= (ap_detail::ubits<W>)(((v >> k) & 1)) << (n - 1 - k);
      v = o;
    }
    r.V = v;
    return r;
  }
  constexpr const ap_int_base<W, false> range() const { ap_int_base<W, false> r; r.V = (ap_detail::ubits<W>)V; return r; }
  constexpr const ap_int_base<W, false> operator()(int hi, int lo) const { return range(hi, lo); }
  constexpr const ap_int_base<W, false> range(int hi, int lo) { return ((const ap_int_base *)this)->range(hi, lo); }
  constexpr const ap_int_base<W, false> operator()(int hi, int lo) { return range(hi, lo); }

  constexpr bool and_reduce() const { return (ap_detail::ubits<W>)V == (ap_detail::ubits<W>)~(ap_detail::ubits<W>)0; }
  constexpr bool nand_reduce() const { return !and_reduce(); }
  constexpr bool or_reduce() const { return V != 0; }
  constexpr bool nor_reduce() const { return V == 0; }
  constexpr bool xor_reduce() const {
    bool p = false;
    for (int i = 0; i < W; i++) p ^= (*this)[i];
    return p;
  }
  constexpr bool xnor_reduce() const { return !xor_reduce(); }
  constexpr void reverse() {
    ap_detail::ubits<W> o = 0, v = (ap_detail::ubits<W>)V;
    for (int k = 0; k < W; k++) o |= (ap_detail::ubits<W>)((v >> k) & 1) << (W - 1 - k);
    V = (raw_t)o;
  }
  constexpr int countLeadingZeros() const {
    int n = 0;
    for (int i = W - 1; i >= 0 && !(*this)[i]; i--) n++;
    return n;
  }

  // ---- unary ---------------------------------------------------------------
  constexpr typename RType<1, false>::minus operator-() const {
    typename RType<1, false>::minus_base r;
    r.V = (typename RType<1, false>::minus_base::raw_t)(-(ap_detail::sbits<W + 2>)rval());
    return typename RType<1, false>::minus(r);
  }
  constexpr typename RType<W, S>::arg1 operator+() const { return typename RType<W, S>::arg1(*this); }
  constexpr typename RType<W, S>::arg1 operator~() const {
    ap_int_base r;
    r.V = (raw_t)~V;
    return typename RType<W, S>::arg1(r);
  }
  constexpr bool operator!() const { return V == 0; }

  constexpr ap_int_base &operator++() { V = (raw_t)((ap_detail::sbits<W + 2>)rval() + 1); return *this; }
  constexpr ap_int_base &operator--() { V = (raw_t)((ap_detail::sbits<W + 2>)rval() - 1); return *this; }
  constexpr const ap_int_base operator++(int) { ap_int_base t(*this); ++*this; return t; }
  constexpr const ap_int_base operator--(int) { ap_int_base t(*this); --*this; return t; }

  // ---- shifts: the result keeps W bits ------------------------------------
  constexpr typename RType<W, S>::arg1 operator<<(unsigned sh) const {
    ap_int_base r;
    r.V = sh < (unsigned)W ? (raw_t)((ap_detail::ubits<W>)V << sh) : (raw_t)0;
    return typename RType<W, S>::arg1(r);
  }
  constexpr typename RType<W, S>::arg1 operator>>(unsigned sh) const {
    ap_int_base r;
    if (sh < (unsigned)W) r.V = (raw_t)(rval() >> sh);
    else r.V = (S && rval() < 0) ? (raw_t)-1 : (raw_t)0;
    return typename RType<W, S>::arg1(r);
  }
  constexpr typename RType<W, S>::arg1 operator<<(int sh) const { return sh < 0 ? operator>>((unsigned)-sh) : operator<<((unsigned)sh); }
  constexpr typename RType<W, S>::arg1 operator>>(int sh) const { return sh < 0 ? operator<<((unsigned)-sh) : operator>>((unsigned)sh); }
  template <int W2, bool S2>
  constexpr typename RType<W, S>::arg1 operator<<(const ap_int_base<W2, S2> &o) const { return operator<<((int)o.to_int64()); }
  template <int W2, bool S2>
  constexpr typename RType<W, S>::arg1 operator>>(const ap_int_base<W2, S2> &o) const { return operator>>((int)o.to_int64()); }
  template <class T> constexpr ap_int_base &operator<<=(const T &sh) { *this = operator<<(sh); return *this; }
  template <class T> constexpr ap_int_base &operator>>=(const T &sh) { *this = operator>>(sh); return *this; }

  // ---- compound assignment (the binary operator, then wrap) ---------------
  template <class T> constexpr ap_int_base &operator+=(const T &o) { *this = (*this + o); return *this; }
  template <class T> constexpr ap_int_base &operator-=(const T &o) { *this = (*this - o); return *this; }
  template <class T> constexpr ap_int_base &operator*=(const T &o) { *this = (*this * o); return *this; }
  template <class T> constexpr ap_int_base &operator/=(const T &o) { *this = (*this / o); return *this; }
  template <class T> constexpr ap_int_base &operator%=(const T &o) { *this = (*this % o); return *this; }
  template <class T> constexpr ap_int_base &operator&=(const T &o) { *this = (*this & o); return *this; }
  template <class T> constexpr ap_int_base &operator|=(const T &o) { *this = (*this | o); return *this; }
  template <class T> constexpr ap_int_base &operator^=(const T &o) { *this = (*this ^ o); return *this; }
};

// ---- binary operators on two ap_int_base: exact in a wide type, then the
// Vitis result width (ap_int_base.h OP_BIN_AP / OP_BIN_AP2) --------------------
namespace ap_detail {
template <int W1, bool S1, int W2, bool S2> struct wide2 { typedef sbits<W1 + W2 + 2> type; };
}
#define HLSEST_AP_INT_BIN(SYM, RTY)                                                         \
  template <int W1, bool S1, int W2, bool S2>                                               \
  constexpr typename ap_int_base<W1, S1>::template RType<W2, S2>::RTY operator SYM(           \
      const ap_int_base<W1, S1> &a, const ap_int_base<W2, S2> &b) {                         \
    typedef typename ap_detail::wide2<W1, S1, W2, S2>::type wide;                           \
    typename ap_int_base<W1, S1>::template RType<W2, S2>::RTY##_base r;                     \
    r.V = (typename ap_int_base<W1, S1>::template RType<W2, S2>::RTY##_base::raw_t)(        \
        (wide)a.rval() SYM(wide) b.rval());                                                 \
    return typename ap_int_base<W1, S1>::template RType<W2, S2>::RTY(r);                    \
  }
HLSEST_AP_INT_BIN(+, plus)
HLSEST_AP_INT_BIN(-, minus)
HLSEST_AP_INT_BIN(*, mult)
HLSEST_AP_INT_BIN(&, logic)
HLSEST_AP_INT_BIN(|, logic)
HLSEST_AP_INT_BIN(^, logic)
#undef HLSEST_AP_INT_BIN
// Division and modulo (etc/ap_private.h operator/ and operator%): both
// operands are extended by their own signedness to the common type of
// MAX(W1,W2) + (S1||S2) bits whose signedness is that of the wider operand
// (signed if either is, at equal widths); quotient and remainder are signed
// when that common type is.
// A negative value in an unsigned common type is therefore its bit pattern
// (ap_uint<16>(17272) / ap_int<8>(-1) is 0). A zero divisor is undefined
// (Vitis: `undef`; the IR carries `divsi x, 0`).
template <int W1, bool S1, int W2, bool S2>
constexpr typename ap_int_base<W1, S1>::template RType<W2, S2>::div operator/(
    const ap_int_base<W1, S1> &a, const ap_int_base<W2, S2> &b) {
  typedef typename ap_int_base<W1, S1>::template RType<W2, S2>::div_base R;
  const int MW = ap_detail::max(W1, W2) + (S1 || S2);
  const bool CS = W1 > W2 ? S1 : (W2 > W1 ? S2 : (S1 || S2));
  typedef ap_detail::sbits<MW + 1> sw;
  typedef ap_detail::ubits<MW> uw;
  sw l = (sw)a.rval(), r = (sw)b.rval();
  R res;
  if (CS) res.V = (typename R::raw_t)(l / r);
  else res.V = (typename R::raw_t)((uw)l / (uw)r);
  return typename ap_int_base<W1, S1>::template RType<W2, S2>::div(res);
}
template <int W1, bool S1, int W2, bool S2>
constexpr typename ap_int_base<W1, S1>::template RType<W2, S2>::mod operator%(
    const ap_int_base<W1, S1> &a, const ap_int_base<W2, S2> &b) {
  typedef typename ap_int_base<W1, S1>::template RType<W2, S2>::mod_base R;
  const int MW = ap_detail::max(W1, W2) + (S1 || S2);
  const bool CS = W1 > W2 ? S1 : (W2 > W1 ? S2 : (S1 || S2));
  typedef ap_detail::sbits<MW + 1> sw;
  typedef ap_detail::ubits<MW> uw;
  sw l = (sw)a.rval(), r = (sw)b.rval();
  R res;
  if (CS) res.V = (typename R::raw_t)(l % r);
  else res.V = (typename R::raw_t)((uw)l % (uw)r);
  return typename ap_int_base<W1, S1>::template RType<W2, S2>::mod(res);
}
#define HLSEST_AP_INT_REL(SYM, EXPR)                                                        \
  template <int W1, bool S1, int W2, bool S2>                                               \
  constexpr bool operator SYM(const ap_int_base<W1, S1> &a, const ap_int_base<W2, S2> &b) { \
    const int MW = ap_detail::max(W1, W2) + 1;                                              \
    ap_detail::sbits<MW> l = (ap_detail::sbits<MW>)a.rval(), r = (ap_detail::sbits<MW>)b.rval(); \
    return EXPR;                                                                            \
  }
HLSEST_AP_INT_REL(<, (ap_detail::cmp_lt<W1, S1, W2, S2, MW>(l, r)))
HLSEST_AP_INT_REL(>, (ap_detail::cmp_lt<W2, S2, W1, S1, MW>(r, l)))
HLSEST_AP_INT_REL(<=, (!ap_detail::cmp_lt<W2, S2, W1, S1, MW>(r, l)))
HLSEST_AP_INT_REL(>=, (!ap_detail::cmp_lt<W1, S1, W2, S2, MW>(l, r)))
HLSEST_AP_INT_REL(==, (ap_detail::cmp_eq<W1, S1, W2, S2, MW>(l, r)))
HLSEST_AP_INT_REL(!=, (!ap_detail::cmp_eq<W1, S1, W2, S2, MW>(l, r)))
#undef HLSEST_AP_INT_REL

// ---- mixed with C integers: the C value becomes ap_int_base<ctype::W, ctype::S>
// (ap_int_base.h ALL_OP_BIN_WITH_INT / ALL_OP_REL_WITH_INT). ------------------
#define HLSEST_AP_INT_MIX(SYM, RTY)                                                         \
  template <int W, bool S, class T>                                                         \
  constexpr typename ap_int_base<W, S>::template RType<ap_detail::ctype<T>::W,              \
                                                        ap_detail::ctype<T>::S>::RTY        \
  operator SYM(const ap_int_base<W, S> &a, T b) {                                           \
    return a SYM ap_int_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(b);            \
  }                                                                                         \
  template <int W, bool S, class T>                                                         \
  constexpr typename ap_int_base<W, S>::template RType<ap_detail::ctype<T>::W,              \
                                                        ap_detail::ctype<T>::S>::RTY        \
      operator SYM(T a, const ap_int_base<W, S> &b) {                                       \
    return ap_int_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(a) SYM b;            \
  }
HLSEST_AP_INT_MIX(+, plus)
HLSEST_AP_INT_MIX(-, minus)
HLSEST_AP_INT_MIX(*, mult)
HLSEST_AP_INT_MIX(/, div)
HLSEST_AP_INT_MIX(%, mod)
HLSEST_AP_INT_MIX(&, logic)
HLSEST_AP_INT_MIX(|, logic)
HLSEST_AP_INT_MIX(^, logic)
#undef HLSEST_AP_INT_MIX
#define HLSEST_AP_INT_MIXREL(SYM)                                                           \
  template <int W, bool S, class T, class = typename ap_detail::ctype<T>::type>             \
  constexpr bool operator SYM(const ap_int_base<W, S> &a, T b) {                            \
    return a SYM ap_int_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(b);            \
  }                                                                                         \
  template <int W, bool S, class T, class = typename ap_detail::ctype<T>::type>             \
  constexpr bool operator SYM(T a, const ap_int_base<W, S> &b) {                            \
    return ap_int_base<ap_detail::ctype<T>::W, ap_detail::ctype<T>::S>(a) SYM b;            \
  }
HLSEST_AP_INT_MIXREL(<)
HLSEST_AP_INT_MIXREL(>)
HLSEST_AP_INT_MIXREL(<=)
HLSEST_AP_INT_MIXREL(>=)
HLSEST_AP_INT_MIXREL(==)
HLSEST_AP_INT_MIXREL(!=)
#undef HLSEST_AP_INT_MIXREL

// Concatenation (a, b): the W1+W2 raw bits, unsigned.
template <int W1, bool S1, int W2, bool S2>
constexpr const ap_uint<W1 + W2> operator,(const ap_int_base<W1, S1> &a, const ap_int_base<W2, S2> &b) {
  ap_int_base<W1 + W2, false> r;
  r.V = ((ap_detail::ubits<W1 + W2>)(ap_detail::ubits<W1>)a.V << W2) | (ap_detail::ubits<W1 + W2>)(ap_detail::ubits<W2>)b.V;
  return ap_uint<W1 + W2>(r);
}

template <int W> struct ap_int : ap_int_base<W, true> {
  typedef ap_int_base<W, true> Base;
  ap_int() = default;
  ap_int(const ap_int &) = default;
  ap_int &operator=(const ap_int &) = default;
  template <class T> constexpr ap_int(const T &v) : Base(v) {}
  template <class T> constexpr ap_int &operator=(const T &v) { this->V = ap_int(v).V; return *this; }
};
template <int W> struct ap_uint : ap_int_base<W, false> {
  typedef ap_int_base<W, false> Base;
  ap_uint() = default;
  ap_uint(const ap_uint &) = default;
  ap_uint &operator=(const ap_uint &) = default;
  template <class T> constexpr ap_uint(const T &v) : Base(v) {}
  template <class T> constexpr ap_uint &operator=(const T &v) { this->V = ap_uint(v).V; return *this; }
};

#ifdef __clang__
#pragma clang attribute pop
#endif

#include "ap_fixed.h"

#endif // __cplusplus
#endif
