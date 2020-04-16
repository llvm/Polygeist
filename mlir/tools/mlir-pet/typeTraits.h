#ifndef PETMLIR_TYPE_TRAITS
#define PETMLIR_TYPE_TRAITS

#include "isl/isl-noexceptions.h"

#include <type_traits>

namespace {
template <typename> struct sfinae_true : public std::true_type {};

template <typename T>
static auto test_has_get_ctx(int)
    -> sfinae_true<decltype(std::declval<T>().get_ctx())>;
template <typename T> static std::false_type test_has_get_ctx(long);

template <typename T> struct has_get_ctx : decltype(test_has_get_ctx<T>(0)) {};
} // namespace

/// \brief Type trait for isl C++ classes.
/// Is true if the class has a ::get_ctx() method, may be extended in the
/// future to account for other isl methods.
template <typename T>
struct is_isl_type
    : public std::integral_constant<bool, has_get_ctx<T>::value> {};

/// \brief Type trait for isl C structures.
/// Is true if the structure can be converted into a C++ isl class instance by
/// calling isl::manage.
template <typename T>
struct is_isl_c_type
    : public std::integral_constant<
          bool, is_isl_type<decltype(isl::manage(std::declval<T>()))>::value> {
};

template <typename T> struct isl_wrap {
  typedef decltype(isl::manage(std::declval<T>())) type;
};

template <typename T> using isl_wrap_t = typename isl_wrap<T>::type;

template <typename T> struct isl_unwrap {
  typedef decltype(std::declval<T>().get()) type;
};

template <typename T> using isl_unwrap_t = typename isl_unwrap<T>::type;

#endif // PETMLIR_TYPE_TRAITS
