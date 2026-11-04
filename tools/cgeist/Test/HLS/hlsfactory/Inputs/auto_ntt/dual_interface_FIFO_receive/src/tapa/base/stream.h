#ifndef TAPA_BASE_STREAM_H_
#define TAPA_BASE_STREAM_H_

namespace tapa {

inline constexpr int kStreamDefaultDepth = 2;

namespace internal {

template <typename T>
struct elem_t {
  T val;
  bool eot;
};

}  // namespace internal

}  // namespace tapa

#endif  // TAPA_BASE_STREAM_H_
