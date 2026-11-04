// hls-est "exactclass" stand-in for the Vitis HLS hls_math.h, over the
// _BitInt-backed ap_fixed / ap_int classes of ap_fixed.h and over the native
// float / double / half types.
//
// Bit-exactness vs Vitis 2025.1 (tools/hls-est-validate/VITIS_NOTES.md):
//   sqrt  exact: etc/hls_sqrt_apfixed.h returns 0 for a negative input and the
//         root rounded half-up to F fractional bits (checked exhaustively for
//         ap_fixed<16,5> by check_shim.py).
//   exp   etc/hls_exp_apfixed.h is a 3-table approximation that SATURATES to
//         the type's maximum (it never wraps). It is "All Rights Reserved" in
//         the Vitis install and not part of the Apache-2.0
//         HLS_arbitrary_Precision_Types repository, so it is not ported: this
//         shim returns floor(exp(x)*2^F) with the same saturation, which
//         differs from the tables by one LSB, in either direction, on ~1% of
//         the inputs (909 of 38 447 ap_fixed<16,5> inputs; check_shim.py).
//   log, tanh, sin, cos, atan, pow, rsqrt and the rest are evaluated in
//         double and truncated (floor) to the Vitis result type; Vitis uses
//         CORDIC / table algorithms there, so a last-bit difference is
//         possible ("unconfirmed" in VITIS_NOTES.md). recip is 1/x with the
//         exact fixed-point division. pow of two ap_fixed values of one type
//         is evaluated in double too (Vitis 2025.1 synthesizes a fixed-point
//         pow core, pow_16_5_s for ap_fixed<16,5>; its last bits are not
//         checked here). Before this overload the call was ambiguous between
//         the float, double and half overloads and did not compile.
// Math-core markers (for hls-est --bram-analysis math-costs=, the cost table
// tools/hls-est-validate/math_costs.json): with
// -DHLSEST_MATH_MARKERS (this directory's cgeist_flags) the builtin call at the
// heart of every fixed-point function below (__builtin_exp, __builtin_sqrt,
// ...) becomes a call of the extern "C" declaration
// __hlsest_math_<hls name>(args..., W, I, S) of the ARGUMENT's ap_fixed_base
// parameters, so hls-est can price the Vitis core of hls::<name> on
// ap_fixed<W,I> (S = 0: ap_ufixed). It replaces a call that is already opaque
// (cgeist emits these builtins as calls of the libm functions; sqrt is a
// math.sqrt next to the opaque floor of raw_sat), returns the builtin's value
// and is never constant-evaluated (none of these builtins is constexpr in
// clang 18), so the IR is the same up to the callee's name and its three
// constant operands. Functions computed in integer arithmetic (recip, abs,
// fabs, floor, ceil, trunc, fmin, fmax, fdim, copysign) have no such call and
// no marker; round / rint keep __builtin_round / __builtin_rint (hls-est does
// not price rounding). Without the macro (host compilers: check_shim.py, and
// check_semantics.py's native builds, which drop it) the builtins are called
// directly, as before; check_semantics.py links the MLIR's markers to stubs
// that call the builtin.
#ifndef HLSEST_EXACTCLASS_HLS_MATH_H
#define HLSEST_EXACTCLASS_HLS_MATH_H
#include <math.h>
#ifdef __cplusplus
#include "ap_fixed.h"
#include "hls_half.h"

#ifdef HLSEST_MATH_MARKERS
extern "C" {
#define HLSEST_MDECL1(N) double __hlsest_math_##N(double, int, int, int);
#define HLSEST_MDECL2(N) double __hlsest_math_##N(double, double, int, int, int);
HLSEST_MDECL1(sqrt) HLSEST_MDECL1(exp) HLSEST_MDECL1(exp2) HLSEST_MDECL1(expm1)
HLSEST_MDECL1(log) HLSEST_MDECL1(log2) HLSEST_MDECL1(log10) HLSEST_MDECL1(log1p)
HLSEST_MDECL1(logb) HLSEST_MDECL1(tanh) HLSEST_MDECL1(sinh) HLSEST_MDECL1(cosh)
HLSEST_MDECL1(atanh) HLSEST_MDECL1(asinh) HLSEST_MDECL1(acosh) HLSEST_MDECL1(erf)
HLSEST_MDECL1(erfc) HLSEST_MDECL1(cbrt) HLSEST_MDECL1(rsqrt) HLSEST_MDECL1(tan)
HLSEST_MDECL1(sinpi) HLSEST_MDECL1(cospi) HLSEST_MDECL1(sin) HLSEST_MDECL1(cos)
HLSEST_MDECL1(atan) HLSEST_MDECL1(asin) HLSEST_MDECL1(acos)
HLSEST_MDECL2(atan2) HLSEST_MDECL2(pow)
#undef HLSEST_MDECL1
#undef HLSEST_MDECL2
}
// N: the hls:: function, B: the builtin it stands for (the value returned);
// W, I, S: the template parameters of the enclosing fixed-point overload.
#define HLSEST_M1(N, B, X) __hlsest_math_##N((X), W, I, (int)S)
#define HLSEST_M2(N, B, X, Y) __hlsest_math_##N((X), (Y), W, I, (int)S)
#else
#define HLSEST_M1(N, B, X) B(X)
#define HLSEST_M2(N, B, X, Y) B((X), (Y))
#endif

namespace hls {
namespace detail {
// floor(rawv) clamped to the raw range of ap_fixed_base<W,I,S> (the
// fixed-point library functions saturate rather than wrap).
template <int W, bool S>
constexpr typename ap_detail::bits<W, S>::type raw_sat(double rawv) {
  typedef typename ap_detail::bits<W, S>::type raw_t;
  double s = __builtin_floor(rawv);
  double mx = (double)(ap_detail::sbits<W + 2>)ap_detail::smax<W, S>();
  double mn = (double)(ap_detail::sbits<W + 2>)ap_detail::smin<W, S>();
  if (!(s == s)) return (raw_t)0; // NaN
  if (s >= mx) return (raw_t)ap_detail::smax<W, S>();
  if (s <= mn) return (raw_t)ap_detail::smin<W, S>();
  return (raw_t)(ap_detail::sbits<W + 2>)s;
}
// floor(d * 2^F) as that raw.
template <int W, int I, bool S>
constexpr typename ap_detail::bits<W, S>::type raw_from_double_sat(double d) {
  return raw_sat<W, S>(d * ap_detail::pow2<W - I>::v);
}
template <int W, int I, bool S>
constexpr ap_fixed_base<W, I, S> fixed_from_double_sat(double d) {
  ap_fixed_base<W, I, S> r;
  r.V = raw_from_double_sat<W, I, S>(d);
  return r;
}
} // namespace detail

// ---- native types -----------------------------------------------------------
#define HLSEST_NATIVE1(NAME, F, D)                                                     \
  inline float NAME(float x) { return F(x); }                                          \
  inline double NAME(double x) { return D(x); }                                        \
  inline half NAME(half x) { return (half)F((float)x); }
HLSEST_NATIVE1(sqrt, __builtin_sqrtf, __builtin_sqrt)
HLSEST_NATIVE1(exp, __builtin_expf, __builtin_exp)
HLSEST_NATIVE1(exp2, __builtin_exp2f, __builtin_exp2)
HLSEST_NATIVE1(expm1, __builtin_expm1f, __builtin_expm1)
HLSEST_NATIVE1(log, __builtin_logf, __builtin_log)
HLSEST_NATIVE1(log2, __builtin_log2f, __builtin_log2)
HLSEST_NATIVE1(log10, __builtin_log10f, __builtin_log10)
HLSEST_NATIVE1(log1p, __builtin_log1pf, __builtin_log1p)
HLSEST_NATIVE1(sin, __builtin_sinf, __builtin_sin)
HLSEST_NATIVE1(cos, __builtin_cosf, __builtin_cos)
HLSEST_NATIVE1(tan, __builtin_tanf, __builtin_tan)
HLSEST_NATIVE1(asin, __builtin_asinf, __builtin_asin)
HLSEST_NATIVE1(acos, __builtin_acosf, __builtin_acos)
HLSEST_NATIVE1(atan, __builtin_atanf, __builtin_atan)
HLSEST_NATIVE1(sinh, __builtin_sinhf, __builtin_sinh)
HLSEST_NATIVE1(cosh, __builtin_coshf, __builtin_cosh)
HLSEST_NATIVE1(tanh, __builtin_tanhf, __builtin_tanh)
HLSEST_NATIVE1(asinh, __builtin_asinhf, __builtin_asinh)
HLSEST_NATIVE1(acosh, __builtin_acoshf, __builtin_acosh)
HLSEST_NATIVE1(atanh, __builtin_atanhf, __builtin_atanh)
HLSEST_NATIVE1(erf, __builtin_erff, __builtin_erf)
HLSEST_NATIVE1(erfc, __builtin_erfcf, __builtin_erfc)
HLSEST_NATIVE1(fabs, __builtin_fabsf, __builtin_fabs)
HLSEST_NATIVE1(abs, __builtin_fabsf, __builtin_fabs)
HLSEST_NATIVE1(floor, __builtin_floorf, __builtin_floor)
HLSEST_NATIVE1(ceil, __builtin_ceilf, __builtin_ceil)
HLSEST_NATIVE1(trunc, __builtin_truncf, __builtin_trunc)
HLSEST_NATIVE1(round, __builtin_roundf, __builtin_round)
HLSEST_NATIVE1(rint, __builtin_rintf, __builtin_rint)
HLSEST_NATIVE1(nearbyint, __builtin_nearbyintf, __builtin_nearbyint)
HLSEST_NATIVE1(cbrt, __builtin_cbrtf, __builtin_cbrt)
HLSEST_NATIVE1(logb, __builtin_logbf, __builtin_logb)
#undef HLSEST_NATIVE1
inline float rsqrt(float x) { return 1.0f / __builtin_sqrtf(x); }
inline double rsqrt(double x) { return 1.0 / __builtin_sqrt(x); }
inline half rsqrt(half x) { return (half)(1.0f / __builtin_sqrtf((float)x)); }
inline float recip(float x) { return 1.0f / x; }
inline double recip(double x) { return 1.0 / x; }
inline half recip(half x) { return (half)(1.0f / (float)x); }
inline float pow(float x, float y) { return __builtin_powf(x, y); }
inline double pow(double x, double y) { return __builtin_pow(x, y); }
inline half pow(half x, half y) { return (half)__builtin_powf((float)x, (float)y); }
inline float atan2(float y, float x) { return __builtin_atan2f(y, x); }
inline double atan2(double y, double x) { return __builtin_atan2(y, x); }
inline float hypot(float x, float y) { return __builtin_hypotf(x, y); }
inline double hypot(double x, double y) { return __builtin_hypot(x, y); }
inline float fmod(float x, float y) { return __builtin_fmodf(x, y); }
inline double fmod(double x, double y) { return __builtin_fmod(x, y); }
inline float fmax(float a, float b) { return a > b ? a : b; }
inline double fmax(double a, double b) { return a > b ? a : b; }
inline float fmin(float a, float b) { return a < b ? a : b; }
inline double fmin(double a, double b) { return a < b ? a : b; }
inline float copysign(float x, float y) { return __builtin_copysignf(x, y); }
inline double copysign(double x, double y) { return __builtin_copysign(x, y); }
inline float nextafter(float x, float y) { return __builtin_nextafterf(x, y); }
inline double nextafter(double x, double y) { return __builtin_nextafter(x, y); }
inline bool signbit(float x) { return __builtin_signbit(x); }
inline bool signbit(double x) { return __builtin_signbit(x); }
inline bool isnan(float x) { return x != x; }
inline bool isnan(double x) { return x != x; }
inline bool isinf(float x) { return __builtin_isinf(x); }
inline bool isinf(double x) { return __builtin_isinf(x); }
inline bool isfinite(float x) { return __builtin_isfinite(x); }
inline bool isfinite(double x) { return __builtin_isfinite(x); }
inline float expf(float x) { return __builtin_expf(x); }
inline float logf(float x) { return __builtin_logf(x); }
inline float sqrtf(float x) { return __builtin_sqrtf(x); }
inline float powf(float x, float y) { return __builtin_powf(x, y); }
inline float fabsf(float x) { return __builtin_fabsf(x); }
inline float sinf(float x) { return __builtin_sinf(x); }
inline float cosf(float x) { return __builtin_cosf(x); }
inline float tanhf(float x) { return __builtin_tanhf(x); }
inline float floorf(float x) { return __builtin_floorf(x); }
inline float ceilf(float x) { return __builtin_ceilf(x); }
inline float roundf(float x) { return __builtin_roundf(x); }
inline float fmaxf(float a, float b) { return a > b ? a : b; }
inline float fminf(float a, float b) { return a < b ? a : b; }
template <class T> T max(T a, T b) { return a > b ? a : b; }
template <class T> T min(T a, T b) { return a < b ? a : b; }

// ---- fixed point: core on ap_fixed_base, then the derived overloads ----------
#define HLSEST_FB template <int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N>
#define HLSEST_FBT ap_fixed_base<W, I, S, Q, O, N>

// sqrt (etc/hls_sqrt_apfixed.h): round-half-up of the exact root. The generic
// branch returns 0 for a negative input (and for I <= 0 when the bits above
// position W+I-2 are set); the small lookup-table branches (F == 0 && I <= 13,
// F <= 4 && I <= 5) drop the sign bit first and take the root of the
// remaining W-1 bits as an unsigned value.
HLSEST_FB constexpr ap_fixed_base<W, I, S> sqrt(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r;
  const int F = W - I;
  const bool table = (F == 0 && I <= 13) || (F <= 4 && I <= 5);
  double xv = x.to_double();
  if (S && x.rval() < 0) {
    if (!table) { r.V = 0; return r; }
    xv = (double)((ap_detail::ubits<W>)x.V & (ap_detail::ubits<W>)((((ap_detail::ubits<W + 1>)1) << (W - 1)) - 1)) * ap_detail::pow2<-(W - I)>::v;
  } else if (!table && I <= 0) {
    const int lsbx = (-I + 2 <= W) ? (W + I - 2) : 0;
    if (((ap_detail::ubits<W>)x.V >> lsbx) != 0) { r.V = 0; return r; }
  }
  double s = HLSEST_M1(sqrt, __builtin_sqrt, xv);
  r.V = detail::raw_sat<W, S>(s * ap_detail::pow2<W - I>::v + 0.5);
  return r;
}
// exp: floor(exp(x)*2^F), saturating to the maximum (never wraps).
HLSEST_FB constexpr ap_fixed_base<W, I, S> exp(const HLSEST_FBT &x) {
  if (I > 33 || W - I > 32) { ap_fixed_base<W, I, S> z; z.V = 0; return z; }
  return detail::fixed_from_double_sat<W, I, S>(HLSEST_M1(exp, __builtin_exp, x.to_double()));
}
#define HLSEST_FIXED_VIA_DOUBLE(NAME, EXPR)                                                 \
  HLSEST_FB constexpr ap_fixed_base<W, I, S> NAME(const HLSEST_FBT &x) {                    \
    double d = x.to_double();                                                               \
    (void)d;                                                                                \
    return detail::fixed_from_double_sat<W, I, S>(EXPR);                                    \
  }
HLSEST_FIXED_VIA_DOUBLE(log, d > 0 ? HLSEST_M1(log, __builtin_log, d) : 0.0)
HLSEST_FIXED_VIA_DOUBLE(log2, d > 0 ? HLSEST_M1(log2, __builtin_log2, d) : 0.0)
HLSEST_FIXED_VIA_DOUBLE(log10, d > 0 ? HLSEST_M1(log10, __builtin_log10, d) : 0.0)
HLSEST_FIXED_VIA_DOUBLE(logb, d > 0 ? HLSEST_M1(logb, __builtin_logb, d) : 0.0)
HLSEST_FIXED_VIA_DOUBLE(exp2, HLSEST_M1(exp2, __builtin_exp2, d))
HLSEST_FIXED_VIA_DOUBLE(expm1, HLSEST_M1(expm1, __builtin_expm1, d))
HLSEST_FIXED_VIA_DOUBLE(log1p, HLSEST_M1(log1p, __builtin_log1p, d))
HLSEST_FIXED_VIA_DOUBLE(tanh, HLSEST_M1(tanh, __builtin_tanh, d))
HLSEST_FIXED_VIA_DOUBLE(sinh, HLSEST_M1(sinh, __builtin_sinh, d))
HLSEST_FIXED_VIA_DOUBLE(cosh, HLSEST_M1(cosh, __builtin_cosh, d))
HLSEST_FIXED_VIA_DOUBLE(atanh, HLSEST_M1(atanh, __builtin_atanh, d))
HLSEST_FIXED_VIA_DOUBLE(asinh, HLSEST_M1(asinh, __builtin_asinh, d))
HLSEST_FIXED_VIA_DOUBLE(acosh, HLSEST_M1(acosh, __builtin_acosh, d))
HLSEST_FIXED_VIA_DOUBLE(erf, HLSEST_M1(erf, __builtin_erf, d))
HLSEST_FIXED_VIA_DOUBLE(erfc, HLSEST_M1(erfc, __builtin_erfc, d))
HLSEST_FIXED_VIA_DOUBLE(cbrt, HLSEST_M1(cbrt, __builtin_cbrt, d))
HLSEST_FIXED_VIA_DOUBLE(rsqrt, d > 0 ? 1.0 / HLSEST_M1(rsqrt, __builtin_sqrt, d) : 0.0)
HLSEST_FIXED_VIA_DOUBLE(tan, HLSEST_M1(tan, __builtin_tan, d))
HLSEST_FIXED_VIA_DOUBLE(sinpi, HLSEST_M1(sinpi, __builtin_sin, d * 3.14159265358979323846))
HLSEST_FIXED_VIA_DOUBLE(cospi, HLSEST_M1(cospi, __builtin_cos, d * 3.14159265358979323846))
#undef HLSEST_FIXED_VIA_DOUBLE
// sin / cos return ap_fixed<W-I+2, 2>, atan ap_fixed<W, 2>, asin / acos
// ap_fixed<W-I+3, 3> (hls_math.h signatures).
HLSEST_FB constexpr ap_fixed_base<W - I + 2, 2, true> sin(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W - I + 2, 2, true>(HLSEST_M1(sin, __builtin_sin, x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W - I + 2, 2, true> cos(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W - I + 2, 2, true>(HLSEST_M1(cos, __builtin_cos, x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W, 2, true> atan(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W, 2, true>(HLSEST_M1(atan, __builtin_atan, x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W - I + 3, 3, true> asin(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W - I + 3, 3, true>(HLSEST_M1(asin, __builtin_asin, x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W - I + 3, 3, true> acos(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W - I + 3, 3, true>(HLSEST_M1(acos, __builtin_acos, x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W, 3, true> atan2(const HLSEST_FBT &y, const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W, 3, true>(HLSEST_M2(atan2, __builtin_atan2, y.to_double(), x.to_double()));
}
// pow: evaluated in double like log (Vitis has a fixed-point pow core,
// pow_W_I_s; last-bit agreement not checked).
HLSEST_FB constexpr ap_fixed_base<W, I, S> pow(const HLSEST_FBT &x, const HLSEST_FBT &y) {
  return detail::fixed_from_double_sat<W, I, S>(HLSEST_M2(pow, __builtin_pow, x.to_double(), y.to_double()));
}
// recip: 1 / x with the exact fixed-point division (truncation).
HLSEST_FB constexpr ap_fixed_base<W, I, S> recip(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r;
  if (x.V == 0) { r.V = 0; return r; }
  r = ap_fixed_base<2, 2, true>(1) / x;
  return r;
}
// Rounding to integers, exact on the raw bits.
HLSEST_FB constexpr ap_fixed_base<W, I, S> floor(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r;
  const int F = W - I;
  if (F <= 0) { r.V = x.V; return r; }          // no fractional bits
  if (F >= W) { r.V = 0; return r; }            // |x| < 1: floor is 0 or an unrepresentable -1 (wraps to 0)
  r.V = (typename HLSEST_FBT::raw_t)((ap_detail::ubits<W>)x.V &
                                     (ap_detail::ubits<W>)~(ap_detail::ubits<W>)((((ap_detail::ubits<W + 1>)1) << F) - 1));
  return r;
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> ceil(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r = floor(x);
  if (r.V != x.V) r += 1;
  return r;
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> trunc(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r = floor(x);
  if (S && x.rval() < 0 && r.V != x.V) r += 1;
  return r;
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> round(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W, I, S>(__builtin_round(x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> rint(const HLSEST_FBT &x) {
  return detail::fixed_from_double_sat<W, I, S>(__builtin_rint(x.to_double()));
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> fabs(const HLSEST_FBT &x) {
  ap_fixed_base<W, I, S> r;
  r.V = (S && x.rval() < 0) ? (typename HLSEST_FBT::raw_t)(-x.rval()) : x.V;
  return r;
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> abs(const HLSEST_FBT &x) { return fabs(x); }
HLSEST_FB constexpr ap_fixed_base<W, I, S> fmax(const HLSEST_FBT &a, const HLSEST_FBT &b) { return a > b ? a : b; }
HLSEST_FB constexpr ap_fixed_base<W, I, S> fmin(const HLSEST_FBT &a, const HLSEST_FBT &b) { return a < b ? a : b; }
HLSEST_FB constexpr ap_fixed_base<W, I, S> fdim(const HLSEST_FBT &a, const HLSEST_FBT &b) {
  ap_fixed_base<W, I, S> r;
  if (a > b) r = a - b; else r.V = 0;
  return r;
}
HLSEST_FB constexpr ap_fixed_base<W, I, S> copysign(const HLSEST_FBT &a, const HLSEST_FBT &b) {
  ap_fixed_base<W, I, S> r = fabs(a);
  if (S && b.rval() < 0) r.V = (typename HLSEST_FBT::raw_t)(-r.rval());
  return r;
}
HLSEST_FB constexpr bool signbit(const HLSEST_FBT &x) { return S && x.rval() < 0; }
HLSEST_FB constexpr bool isnan(const HLSEST_FBT &) { return false; }
HLSEST_FB constexpr bool isinf(const HLSEST_FBT &) { return false; }
HLSEST_FB constexpr bool isfinite(const HLSEST_FBT &) { return true; }

// Derived-type overloads: same result type as the argument (Vitis returns
// ap_fixed<W,I> / ap_ufixed<W,I>); the base overload above is picked for the
// intermediate ap_fixed_base results of arithmetic.
#define HLSEST_FIXED_DERIVED(NAME)                                                          \
  template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>                                  \
  constexpr ap_fixed<W, I, Q, O, N> NAME(const ap_fixed<W, I, Q, O, N> &x) {                \
    return ap_fixed<W, I, Q, O, N>(NAME((const ap_fixed_base<W, I, true, Q, O, N> &)x));    \
  }                                                                                         \
  template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>                                  \
  constexpr ap_ufixed<W, I, Q, O, N> NAME(const ap_ufixed<W, I, Q, O, N> &x) {              \
    return ap_ufixed<W, I, Q, O, N>(NAME((const ap_fixed_base<W, I, false, Q, O, N> &)x));  \
  }                                                                                         \
  template <int W> constexpr ap_int<W> NAME(const ap_int<W> &x) {                           \
    return ap_int<W>(NAME(ap_fixed_base<W, W, true>(x)));                                   \
  }                                                                                         \
  template <int W> constexpr ap_uint<W> NAME(const ap_uint<W> &x) {                         \
    return ap_uint<W>(NAME(ap_fixed_base<W, W, false>(x)));                                 \
  }
HLSEST_FIXED_DERIVED(sqrt)
HLSEST_FIXED_DERIVED(exp)
HLSEST_FIXED_DERIVED(exp2)
HLSEST_FIXED_DERIVED(expm1)
HLSEST_FIXED_DERIVED(log)
HLSEST_FIXED_DERIVED(log2)
HLSEST_FIXED_DERIVED(log10)
HLSEST_FIXED_DERIVED(log1p)
HLSEST_FIXED_DERIVED(logb)
HLSEST_FIXED_DERIVED(tanh)
HLSEST_FIXED_DERIVED(sinh)
HLSEST_FIXED_DERIVED(cosh)
HLSEST_FIXED_DERIVED(atanh)
HLSEST_FIXED_DERIVED(asinh)
HLSEST_FIXED_DERIVED(acosh)
HLSEST_FIXED_DERIVED(erf)
HLSEST_FIXED_DERIVED(erfc)
HLSEST_FIXED_DERIVED(cbrt)
HLSEST_FIXED_DERIVED(rsqrt)
HLSEST_FIXED_DERIVED(recip)
HLSEST_FIXED_DERIVED(tan)
HLSEST_FIXED_DERIVED(floor)
HLSEST_FIXED_DERIVED(ceil)
HLSEST_FIXED_DERIVED(trunc)
HLSEST_FIXED_DERIVED(round)
HLSEST_FIXED_DERIVED(rint)
HLSEST_FIXED_DERIVED(fabs)
HLSEST_FIXED_DERIVED(abs)
HLSEST_FIXED_DERIVED(sinpi)
HLSEST_FIXED_DERIVED(cospi)
#undef HLSEST_FIXED_DERIVED
#define HLSEST_FIXED_DERIVED2(NAME)                                                         \
  template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>                                  \
  constexpr ap_fixed<W, I, Q, O, N> NAME(const ap_fixed<W, I, Q, O, N> &a,                  \
                                         const ap_fixed<W, I, Q, O, N> &b) {                \
    return ap_fixed<W, I, Q, O, N>(NAME((const ap_fixed_base<W, I, true, Q, O, N> &)a,      \
                                        (const ap_fixed_base<W, I, true, Q, O, N> &)b));    \
  }                                                                                         \
  template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>                                  \
  constexpr ap_ufixed<W, I, Q, O, N> NAME(const ap_ufixed<W, I, Q, O, N> &a,                \
                                          const ap_ufixed<W, I, Q, O, N> &b) {              \
    return ap_ufixed<W, I, Q, O, N>(NAME((const ap_fixed_base<W, I, false, Q, O, N> &)a,    \
                                         (const ap_fixed_base<W, I, false, Q, O, N> &)b));  \
  }
HLSEST_FIXED_DERIVED2(fmax)
HLSEST_FIXED_DERIVED2(fmin)
HLSEST_FIXED_DERIVED2(fdim)
HLSEST_FIXED_DERIVED2(copysign)
HLSEST_FIXED_DERIVED2(pow)
#undef HLSEST_FIXED_DERIVED2
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W - I + 2, 2> sin(const ap_fixed<W, I, Q, O, N> &x) { return ap_fixed<W - I + 2, 2>(sin((const ap_fixed_base<W, I, true, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W - I + 2, 2> cos(const ap_fixed<W, I, Q, O, N> &x) { return ap_fixed<W - I + 2, 2>(cos((const ap_fixed_base<W, I, true, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W, 2> atan(const ap_fixed<W, I, Q, O, N> &x) { return ap_fixed<W, 2>(atan((const ap_fixed_base<W, I, true, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W - I + 3, 3> asin(const ap_fixed<W, I, Q, O, N> &x) { return ap_fixed<W - I + 3, 3>(asin((const ap_fixed_base<W, I, true, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W - I + 3, 3> acos(const ap_fixed<W, I, Q, O, N> &x) { return ap_fixed<W - I + 3, 3>(acos((const ap_fixed_base<W, I, true, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_fixed<W, 3> atan2(const ap_fixed<W, I, Q, O, N> &y, const ap_fixed<W, I, Q, O, N> &x) {
  return ap_fixed<W, 3>(atan2((const ap_fixed_base<W, I, true, Q, O, N> &)y, (const ap_fixed_base<W, I, true, Q, O, N> &)x));
}
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_ufixed<W - I + 2, 2> sin(const ap_ufixed<W, I, Q, O, N> &x) { return ap_ufixed<W - I + 2, 2>(sin((const ap_fixed_base<W, I, false, Q, O, N> &)x)); }
template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
constexpr ap_ufixed<W - I + 2, 2> cos(const ap_ufixed<W, I, Q, O, N> &x) { return ap_ufixed<W - I + 2, 2>(cos((const ap_fixed_base<W, I, false, Q, O, N> &)x)); }
#undef HLSEST_FB
#undef HLSEST_FBT
} // namespace hls
#undef HLSEST_M1
#undef HLSEST_M2
#endif // __cplusplus
#endif
