// The members of hls::stream<T> carry the bits Vitis packs T into on the
// FIFO (`hls.stream_element_bits`): a struct is the sum of its members with
// no padding, `bool` is 1 bit, ap types their width, arrays n x element.
// The IR type cannot tell (`bool` is an i8 there). CIRCT VITIS_NOTES N58,
// kernels w9r7_sb_*: a stream of {float; bool} is fifo_w33, of {short; bool}
// fifo_w17.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

template <typename T> struct Data { T data; bool ctrl; };
struct Pix { ap_fixed<20, 5> v[3]; bool last; ap_uint<3> tag; };

// CHECK-DAG: func.func private @_ZN3hls6streamI4DataIfELi0EE5writeERKS2_({{.*}} attributes {{{.*}}hls.stream_element_bits = 33 : i64
// CHECK-DAG: func.func private @_ZN3hls6streamI4DataIsELi0EE4readEv({{.*}} attributes {{{.*}}hls.stream_element_bits = 17 : i64
// CHECK-DAG: func.func private @_ZN3hls6streamI3PixLi0EE5writeERKS1_({{.*}} attributes {{{.*}}hls.stream_element_bits = 64 : i64
// CHECK-DAG: func.func private @_ZN3hls6streamI7ap_uintILi64EELi0EE4readEv({{.*}} attributes {{{.*}}hls.stream_element_bits = 64 : i64
void sb_top(hls::stream<Data<float> > &a, hls::stream<Data<short> > &b,
            hls::stream<Pix> &c, hls::stream<ap_uint<64> > &d, float x) {
  Data<float> f;
  f.data = x;
  f.ctrl = x > 0;
  a.write(f);
  Data<short> s = b.read();
  Pix p;
  p.v[0] = s.data;
  p.v[1] = 0;
  p.v[2] = 1;
  p.last = s.ctrl;
  p.tag = d.read();
  c.write(p);
}
