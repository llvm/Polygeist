#ifndef TAPA_HOST_UTIL_H_
#define TAPA_HOST_UTIL_H_

#include <climits>

#include <iostream>

#include "tapa/base/util.h"

// FIXME: TAPA should provide its equivalence
#ifndef ap_wait
void ap_wait(void);
#endif

#ifndef ap_wait_n
void ap_wait_n(int);
#endif

namespace tapa {

/// Obtain a value of type @c To by reinterpreting the object representation of
/// @c from.
///
/// @note       This function is slightly different from C++20 @c std::bit_cast.
/// @tparam To  Target type.
/// @param from Source object.
template <typename To, typename From>
inline typename std::enable_if<sizeof(To) == sizeof(From), To>::type  //
bit_cast(From from) noexcept {
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

template <typename T>
inline T reg(T x) {
  return x;
}

}  // namespace tapa

#endif  // TAPA_HOST_UTIL_H_