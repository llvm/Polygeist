// Regression (HLSFactory Eyeriss): `variable=` naming an element of an array
// (typically an array of streams) was rejected as "expected a variable
// name". The storage of the whole array is annotated, recording the element.
// RUN: cgeist %s --function='*' -S --hls-annotate %vitis_include --scalar-wrapper-abi 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate %vitis_include --scalar-wrapper-abi -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: error
#include "hls_stream.h"

struct Tile {
  int rows[4][8];
};

// The path is recorded as written.
// CHECK-LABEL: func.func @_Z6eyerisPi(
// CHECK-DAG: {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "tiles[2].rows"}], hls.preserve, polygeist.varname = "tiles"}
// CHECK-DAG: {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "t.rows[1]"}], polygeist.varname = "t"}
// CHECK-DAG: hls.stream = [{depth = 3 : i32, type = "fifo", variable = "psum[0]"}, {depth = 3 : i32, type = "fifo", variable = "psum[1]"}, {depth = 4 : i32, type = "fifo", variable = "psum[3-1]"}], polygeist.varname = "psum"
void eyeris(int *a) {
  hls::stream<int> psum[3];
#pragma HLS STREAM variable = psum[0] type = fifo depth = 3
#pragma HLS STREAM variable = psum[1] type = fifo depth = 3
#pragma HLS stream variable=psum[3-1] depth=4
  struct Tile t;
  struct Tile tiles[4];
#pragma HLS array_partition variable=t.rows[1] complete
#pragma HLS array_partition variable=tiles[2].rows complete
  for (int i = 0; i < 3; ++i)
    psum[i].write(a[i]);
  for (int i = 0; i < 32; ++i) {
    t.rows[i / 8][i % 8] = psum[i % 3].read();
    tiles[i % 4].rows[i / 8][i % 8] = t.rows[3 - i / 8][i % 8];
  }
  for (int i = 0; i < 32; ++i)
    a[i] = tiles[i % 4].rows[i / 8][i % 8] + t.rows[i / 8][7 - i % 8];
}

#ifdef ERRORS
void errors(int *a) {
  hls::stream<int> s[2];
  int scalar = 0;
// ERR: hlsfactory-subscripted-variable.cpp:[[@LINE+1]]:30: error: index 2 is out of range for 'hls::stream<int>[2]' of size 2 (in '#pragma HLS stream')
#pragma HLS stream variable=s[2] depth=4
// ERR: hlsfactory-subscripted-variable.cpp:[[@LINE+1]]:35: error: subscripted value of type 'int' is not an array (in '#pragma HLS stream')
#pragma HLS stream variable=scalar[0] depth=4
  s[0].write(scalar);
  a[0] = s[0].read();
}
#endif
