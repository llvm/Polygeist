// Minimal stand-in for the Vitis HLS hls_stream.h, for cgeist tests only.
// Stream operations are declared, not defined: cgeist emits them as calls to
// external functions, which is all pragma extraction needs.
#ifndef CGEIST_TEST_HLS_STREAM_H
#define CGEIST_TEST_HLS_STREAM_H

#ifdef __cplusplus
namespace hls {

template <typename T, int DEPTH = 0> class stream {
public:
  stream();
  stream(const char *name);
  T read();
  void read(T &v);
  bool read_nb(T &v);
  void write(const T &v);
  bool write_nb(const T &v);
  bool empty() const;
  bool full() const;
  int size() const;
  void operator>>(T &v);
  void operator<<(const T &v);

private:
  int handle;
};

} // namespace hls
#endif // __cplusplus
#endif
