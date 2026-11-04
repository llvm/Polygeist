// The pragma `variable=` forms the Vitis HLS headers use (track W11):
// a namespace-qualified global (etc/hls_hotbm_apfixed.h:91
// `variable=fourth_order_double::sin_cos_K0`) and `this` / `*this`
// (hls_stream_39.h:162 `stream variable=this`), which names the object a
// member function runs on: reported and dropped, except for AGGREGATE, where
// `this` (N37, any mode) and `*this` (--xlx-headers) name the member
// function's object (hls_vector.h:110). The qualified name and `*this` are
// accepted only under --xlx-headers; without it they are errors, as before
// W11, and a plain `this` outside AGGREGATE stays an error (the second
// RUN line leaves the qualified name out, since cgeist without --xlx-headers
// cannot address a namespace-qualified global).
// RUN: cgeist %s --xlx-headers --function='*' -S --hls-annotate -o - 2>&1 | FileCheck %s
// RUN: not cgeist %s -DOFF --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

#ifndef OFF
namespace tables {
int rom[256];
}
#endif

struct Vec {
  int d[4];
  int get(int i) const {
#pragma HLS stream variable=this depth=2
    return d[i];
  }
  int first() const {
#pragma HLS stream variable=*this depth=2
    return d[0];
  }
  int last() const {
#ifndef OFF
#pragma HLS aggregate variable=*this
#endif
    return d[3];
  }
};

// CHECK: warning: 'this' in '#pragma HLS stream' names the enclosing object; ignored
// CHECK: warning: '*this' in '#pragma HLS stream' names the enclosing object; ignored
// CHECK-NOT: aggregate' names the enclosing object
// CHECK-NOT: error
// OFF-NOT: names the enclosing object
// OFF-DAG: error: invalid value '*this' for option 'variable' in '#pragma HLS stream': expected a variable name
// OFF-DAG: error: use of undeclared variable 'this' in '#pragma HLS stream'
// OFF-NOT: names the enclosing object
// CHECK: memref.global @_ZN6tables3romE : memref<256xi32> = uninitialized {hls.bind_storage = [{impl = "lutram", type = "rom_1p", variable = "tables::rom"}]}
// CHECK: memref.alloca() {hls.aggregate = [{compact = "auto", from = "_ZNK3Vec4lastEv", variable = "*this"}], polygeist.varname = "v"}
extern "C" int top(int i) {
#ifndef OFF
#pragma HLS bind_storage variable=tables::rom type=rom_1p impl=lutram
#endif
  Vec v;
  for (int k = 0; k < 4; k++)
    v.d[k] = k;
#ifndef OFF
  return tables::rom[i] + v.get(i & 3) + v.first() + v.last();
#else
  return v.get(i & 3) + v.first() + v.last();
#endif
}
