#ifndef TAPA_TRAITS_H_
#define TAPA_TRAITS_H_

#include "tapa/stream.h"

namespace tapa {

template <typename T>
struct elem_type {};
template <typename T>
struct elem_type<istream<T>&> {
  using type = T;
};
template <typename T>
struct elem_type<ostream<T>&> {
  using type = T;
};
template <typename T, uint64_t S>
struct elem_type<istreams<T, S>&> {
  using type = T;
};
template <typename T, uint64_t S>
struct elem_type<ostreams<T, S>&> {
  using type = T;
};

}  // namespace tapa

#define TAPA_ELEM_TYPE(variable) \
  typename ::tapa::elem_type<decltype(variable)>::type

#endif  // TAPA_TRAITS_H_
