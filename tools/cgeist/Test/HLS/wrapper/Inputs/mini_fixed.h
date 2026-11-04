// A minimal fixed-point scalar wrapper with the Vitis ap_fixed<W,I> value
// semantics the attention kernel needs (AP_TRN / AP_WRAP: int and double
// constructors wrap, double -> fixed floors, products are W1+W2 bits and are
// floored back, division truncates toward zero, sqrt of a negative value is
// 0). The production shim lives in the hls-est harness
// (tools/hls-est-validate/shim/exactclass); this header only exercises the
// same cgeist paths under --scalar-wrapper-abi: a wrapper chain, a
// `static const int` member, _BitInt intermediates, by-value returns and
// compound assignment.
#ifndef MINI_FIXED_H
#define MINI_FIXED_H
template <int W, bool S> struct fx_raw { typedef _BitInt(W) type; };
template <int W> struct fx_raw<W, false> { typedef unsigned _BitInt(W) type; };
typedef _BitInt(128) fx_wide;

template <int W, int I, bool S> struct fx_base {
  static const int F = W - I;
  typedef typename fx_raw<W, S>::type raw_t;
  raw_t V;
  template <int W2, int I2, bool S2> static raw_t from(const fx_base<W2, I2, S2> &o) {
    const int F2 = W2 - I2;
    fx_wide v = (fx_wide)o.V;
    if (F2 > F) v = v >> (F2 - F);
    else if (F > F2) v = v << (F - F2);
    return (raw_t)v;
  }
  // floor(d * 2^F), wrapped to W bits (exact for |d * 2^F| < 2^53).
  static raw_t from_double(double d) {
    double s = d * (double)(1LL << F);
    fx_wide t = (fx_wide)s;
    if ((double)t > s)
      t = t - 1;
    return (raw_t)t;
  }
  fx_base() = default;
  fx_base(const fx_base &) = default;
  fx_base &operator=(const fx_base &) = default;
  template <int W2, int I2, bool S2> fx_base(const fx_base<W2, I2, S2> &o) : V(from(o)) {}
  fx_base(int x) : V((raw_t)((fx_wide)x << F)) {}
  fx_base(double d) : V(from_double(d)) {}
  template <int W2, int I2, bool S2> fx_base &operator=(const fx_base<W2, I2, S2> &o) { V = from(o); return *this; }
  operator double() const { return (double)V / (double)(1LL << F); }
  template <int W2, int I2, bool S2>
  fx_base<W + W2, I + I2, S || S2> operator*(const fx_base<W2, I2, S2> &o) const {
    fx_base<W + W2, I + I2, S || S2> r;
    r.V = (typename fx_base<W + W2, I + I2, S || S2>::raw_t)((fx_wide)V * (fx_wide)o.V);
    return r;
  }
  template <int W2, int I2, bool S2>
  fx_base<S2 + W + (W2 - I2 > 0 ? W2 - I2 : 0), S2 + I + (W2 - I2), S || S2>
  operator/(const fx_base<W2, I2, S2> &o) const {
    typedef fx_base<S2 + W + (W2 - I2 > 0 ? W2 - I2 : 0), S2 + I + (W2 - I2), S || S2> R;
    R r;
    r.V = (typename R::raw_t)(((fx_wide)V << (W2 - I2 > 0 ? W2 - I2 : 0)) / (fx_wide)o.V);
    return r;
  }
  template <int W2, int I2, bool S2>
  fx_base<(I > I2 ? I : I2) + 1 + (F > W2 - I2 ? F : W2 - I2), (I > I2 ? I : I2) + 1, true>
  operator+(const fx_base<W2, I2, S2> &o) const {
    typedef fx_base<(I > I2 ? I : I2) + 1 + (F > W2 - I2 ? F : W2 - I2), (I > I2 ? I : I2) + 1, true> R;
    const int F2 = W2 - I2, F3 = F > F2 ? F : F2;
    R r;
    r.V = (typename R::raw_t)(((fx_wide)V << (F3 - F)) + ((fx_wide)o.V << (F3 - F2)));
    return r;
  }
  template <class T> fx_base &operator*=(const T &o) { *this = (*this) * o; return *this; }
  template <class T> fx_base &operator+=(const T &o) { *this = (*this) + o; return *this; }
};
template <int W, int I> struct fx : fx_base<W, I, true> {
  typedef fx_base<W, I, true> Base;
  fx() = default;
  fx(const fx &) = default;
  fx &operator=(const fx &) = default;
  template <class T> fx(const T &v) : Base(v) {}
  template <class T> fx &operator=(const T &v) { Base::operator=(fx(v)); return *this; }
};
namespace hls {
template <int W, int I> fx<W, I> sqrt(fx<W, I> x) {
  fx<W, I> r;
  if (x.V < 0) { r.V = 0; return r; }
  r.V = (typename fx<W, I>::raw_t)(fx_wide)(__builtin_sqrt((double)x) * (double)(1LL << (W - I)) + 0.5);
  return r;
}
} // namespace hls
#endif
