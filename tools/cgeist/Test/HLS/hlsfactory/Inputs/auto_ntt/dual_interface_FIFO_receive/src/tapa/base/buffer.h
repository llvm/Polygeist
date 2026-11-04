#ifndef TAPA_BASE_BUFFER_H
#define TAPA_BASE_BUFFER_H

namespace tapa {

struct normal {};
struct complete {};

template <int ft>
struct cyclic {
  const int factor = ft;
};

template <int ft>
struct block {
  const int factor = ft;
};

template <typename... partitions>
struct array_partition {};

struct bram {};
struct uram {};

template <typename core_type>
struct memcore {};

}  // namespace tapa

#endif  // TAPA_BASE_BUFFER_H