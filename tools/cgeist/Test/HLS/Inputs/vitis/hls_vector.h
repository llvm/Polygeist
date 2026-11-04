// Minimal stand-in for the Vitis HLS hls_vector.h, for cgeist tests only.
#ifndef CGEIST_TEST_HLS_VECTOR_H
#define CGEIST_TEST_HLS_VECTOR_H

#ifdef __cplusplus
#include <stddef.h>

namespace hls {
template <typename T, size_t N> class vector {
public:
  T data[N];
  T &operator[](size_t i) { return data[i]; }
  const T &operator[](size_t i) const { return data[i]; }
  vector &operator+=(const vector &o) {
    for (size_t i = 0; i < N; ++i)
      data[i] += o.data[i];
    return *this;
  }
  vector &operator-=(const vector &o) {
    for (size_t i = 0; i < N; ++i)
      data[i] -= o.data[i];
    return *this;
  }
  vector &operator*=(const vector &o) {
    for (size_t i = 0; i < N; ++i)
      data[i] *= o.data[i];
    return *this;
  }
};
template <typename T, size_t N>
vector<T, N> operator+(vector<T, N> a, const vector<T, N> &b) {
  return a += b;
}
template <typename T, size_t N>
vector<T, N> operator-(vector<T, N> a, const vector<T, N> &b) {
  return a -= b;
}
template <typename T, size_t N>
vector<T, N> operator*(vector<T, N> a, const vector<T, N> &b) {
  return a *= b;
}
} // namespace hls
#endif // __cplusplus
#endif
