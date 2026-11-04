#ifndef TAPA_XILINX_HLS_BUFFER_H_
#define TAPA_XILINX_HLS_BUFFER_H_

#include <hls_stream.h>
#include "tapa/base/buffer.h"

namespace tapa {

template <typename T, int n_sections, typename... dims>
class _buffer;

template <typename T, int n_sections, typename... dims>
class section {
 public:
  T& operator()() {
#pragma HLS inline
    last = true;
    return buf_ref.data[section_id];
  }

  const T& operator()() const {
#pragma HLS inline
    last = true;
    return buf_ref.data[section_id];
  }

  ~section() {
#pragma HLS inline
    if (last) {
      buf_ref.sink.write(section_id);
    }
  }

 private:
  using buffer_t = _buffer<T, n_sections, dims...>;
  friend buffer_t;

  section(buffer_t& buf_ref) : buf_ref(buf_ref) {
#pragma HLS inline
    section_id = buf_ref.src.read();
  }

  buffer_t& buf_ref;
  int section_id;
  mutable volatile bool last = true;
};

template <typename T, int n_sections, typename... dims>
class _buffer {
 public:
  using section_t = section<T, n_sections, dims...>;
  section_t acquire() {
#pragma HLS inline
    return *this;
  }

  hls::stream<int> src;
  hls::stream<int> sink;

  // This +1 is not needed, however, if we don't do this,
  // whenever users use single buffers, fake dependency is
  // not respected. Of course, doing it this way adds more
  // address lines, however, that may not matter much. Not
  // sure, but Vivado might be able to optimize it too since
  // the wire won't have an external connection
  // TODO: confirm this from physical design
  T data[n_sections == 1 ? 2 : n_sections];
};

// for HLS interface, ibuffer and obuffer can be totally
// same as we take care of src and sink FIFOs in the
// stitching process
template <typename T, int n_sections, typename... dims>
using ibuffer = _buffer<T, n_sections, dims...>;

template <typename T, int n_sections, typename... dims>
using obuffer = _buffer<T, n_sections, dims...>;

template <typename T, int n_sections, typename... dims>
using buffer = _buffer<T, n_sections, dims...>;

template <typename T, int len, int n_sections, typename... dims>
using ibuffers = ibuffer<T, n_sections, dims...>[len];

template <typename T, int len, int n_sections, typename... dims>
using obuffers = obuffer<T, n_sections, dims...>[len];

template <typename T, int len, int n_sections, typename... dims>
class buffers;

}  // namespace tapa

#endif