#ifndef TAPA_XILINX_HLS_UTIL_H_
#define TAPA_XILINX_HLS_UTIL_H_

#include "tapa/base/util.h"

#include <cstring>

#ifdef __SYNTHESIS__

namespace tapa {

template <typename To, typename From>
inline typename std::enable_if<sizeof(To) == sizeof(From), To>::type  //
bit_cast(From from) noexcept {
#pragma HLS inline
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

template <typename T>
T reg(T x) {
#pragma HLS inline off
#pragma HLS pipeline II = 1
#pragma HLS latency min = 1 max = 1
  return x;
}

}  // namespace tapa

#undef TAPA_WHILE_NOT_EOT
#define TAPA_WHILE_NOT_EOT(fifo)                                \
  for (bool tapa_##fifo##_valid;                                \
       !fifo.eot(tapa_##fifo##_valid) || !tapa_##fifo##_valid;) \
  _Pragma("HLS pipeline II = 1") if (tapa_##fifo##_valid)

#undef TAPA_WHILE_NEITHER_EOT
#define TAPA_WHILE_NEITHER_EOT(fifo1, fifo2)                          \
  for (bool tapa_##fifo1##_valid, tapa_##fifo2##_valid;               \
       (!fifo1.eot(tapa_##fifo1##_valid) || !tapa_##fifo1##_valid) && \
       (!fifo2.eot(tapa_##fifo2##_valid) || !tapa_##fifo2##_valid);)  \
  _Pragma("HLS pipeline II = 1") if (tapa_##fifo1##_valid &&          \
                                     tapa_##fifo2##_valid)

#undef TAPA_WHILE_NONE_EOT
#define TAPA_WHILE_NONE_EOT(fifo1, fifo2, fifo3)                              \
  for (bool tapa_##fifo1##_valid, tapa_##fifo2##_valid, tapa_##fifo3##_valid; \
       (!fifo1.eot(tapa_##fifo1##_valid) || !tapa_##fifo1##_valid) &&         \
       (!fifo2.eot(tapa_##fifo2##_valid) || !tapa_##fifo2##_valid) &&         \
       (!fifo3.eot(tapa_##fifo3##_valid) || !tapa_##fifo3##_valid);)          \
  _Pragma("HLS pipeline II = 1") if (tapa_##fifo1##_valid &&                  \
                                     tapa_##fifo2##_valid &&                  \
                                     tapa_##fifo3##_valid)

#else  // __SYNTHESIS__

#include "tapa/host/util.h"

#endif  // __SYNTHESIS__

#endif  // TAPA_XILINX_HLS_UTIL_H_
