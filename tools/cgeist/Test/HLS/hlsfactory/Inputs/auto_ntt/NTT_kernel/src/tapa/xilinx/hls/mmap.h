#ifndef TAPA_XILINX_HLS_MMAP_H_
#define TAPA_XILINX_HLS_MMAP_H_

#include "tapa/base/mmap.h"

#ifdef __SYNTHESIS__

#include <cstddef>

#include "tapa/xilinx/hls/stream.h"

namespace tapa {

template <typename T>
using mmap = T*;

template <typename T>
struct async_mmap {
  using addr_t = int64_t;
  using resp_t = uint8_t;

  tapa::ostream<addr_t> read_addr;
  tapa::istream<T> read_data;
  tapa::ostream<addr_t> write_addr;
  tapa::ostream<T> write_data;
  tapa::istream<resp_t> write_resp;
};

template <typename T, uint64_t S>
class mmaps;

}  // namespace tapa

#else  // __SYNTHESIS__

#include "tapa/host/mmap.h"

#endif  // __SYNTHESIS__

#endif  // TAPA_XILINX_HLS_MMAP_H_
