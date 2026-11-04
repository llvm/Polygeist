#ifndef TAPA_XILINX_HLS_STREAM_H_
#define TAPA_XILINX_HLS_STREAM_H_

#include "tapa/base/stream.h"

#ifdef __SYNTHESIS__

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <hls_stream.h>

namespace tapa {

template <typename T>
class istream {
 public:
  bool empty() const {
#pragma HLS inline
    return _.empty();
  }

  bool try_eot(bool& is_eot) const {
#pragma HLS inline
    internal::elem_t<T> elem;
    const bool is_success = !empty() && _peek.read_nb(elem);
    is_eot = elem.eot;
    return is_success;
  }

  bool eot(bool& is_success) const {
#pragma HLS inline
    bool eot = false;
    is_success = try_eot(eot);
    return eot;
  }

  bool eot(std::nullptr_t) const {
#pragma HLS inline
    bool is_success;
    return eot(is_success) && is_success;
  }

  bool try_peek(T& value) const {
#pragma HLS inline
    internal::elem_t<T> elem;
    const bool is_success = !empty() && _peek.read_nb(elem);
    value = elem.val;
    return is_success;
  }

  T peek(bool& is_success) const {
#pragma HLS inline
    T val;
    is_success = try_peek(val);
    return val;
  }

  T peek(std::nullptr_t) const {
#pragma HLS inline
    T val;
    try_peek(val);
    return val;
  }

  T peek(bool& is_success, bool& is_eot) const {
#pragma HLS inline
    internal::elem_t<T> peek_val;
    (is_success = !empty()) && _peek.read_nb(peek_val);
    is_eot = peek_val.eot && is_success;
    return peek_val.val;
  }

  bool try_read(T& value) {
#pragma HLS inline
    internal::elem_t<T> elem;
    const bool is_success = _.read_nb(elem);
    value = elem.val;
    return is_success;
  }

  T read() {
#pragma HLS inline
    return _.read().val;
  }

  istream& operator>>(T& value) {
#pragma HLS inline
    value = read();
    return *this;
  }

  T read(bool& is_success) {
#pragma HLS inline
    internal::elem_t<T> elem;
    is_success = _.read_nb(elem);
    return elem.val;
  }

  T read(std::nullptr_t) {
#pragma HLS inline
    internal::elem_t<T> elem;
    _.read_nb(elem);
    return elem.val;
  }

  T read(const T& default_value, bool* is_success = nullptr) {
#pragma HLS inline
    internal::elem_t<T> elem;
    bool is_success_val = _.read_nb(elem);
    if (is_success != nullptr) {
      *is_success = is_success_val;
    }
    return is_success_val ? elem.val : default_value;
  }

  bool try_open() {
#pragma HLS inline
    internal::elem_t<T> elem;
    const bool succeeded = _.read_nb(elem);
    assert(!succeeded || elem.eot);
    return succeeded;
  }

  void open() {
#pragma HLS inline
    const auto elem = _.read();
    assert(elem.eot);
  }

  hls::stream<internal::elem_t<T>> _;
  mutable hls::stream<internal::elem_t<T>> _peek;
};

template <typename T>
class ostream {
 public:
  bool full() const {
#pragma HLS inline
    return _.full();
  }

  bool try_write(const T& value) {
#pragma HLS inline
    return _.write_nb({value, false});
  }

  void write(const T& value) {
#pragma HLS inline
    _.write({value, false});
  }

  ostream& operator<<(const T& value) {
#pragma HLS inline
    write(value);
    return *this;
  }

  bool try_close() {
#pragma HLS inline
    internal::elem_t<T> elem;
    memset(&elem.val, 0, sizeof(elem.val));
    elem.eot = true;
    return _.write_nb(elem);
  }

  void close() {
#pragma HLS inline
    internal::elem_t<T> elem;
    elem.eot = true;
    _.write(elem);
  }

  hls::stream<internal::elem_t<T>> _;
};

/// Defines a communication channel between two task instances.
template <typename T, uint64_t N = kStreamDefaultDepth>
class stream;

template <typename T, uint64_t S>
using istreams = istream<T>[S];

template <typename T, uint64_t S>
using ostreams = ostream<T>[S];

template <typename T, uint64_t S, uint64_t N = kStreamDefaultDepth>
class streams;

}  // namespace tapa

#else  // __SYNTHESIS__

#include "tapa/host/stream.h"

#endif  // __SYNTHESIS__

#endif  // TAPA_XILINX_HLS_STREAM_H_
