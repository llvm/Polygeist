// hls::stream from the real Vitis HLS headers (hls_stream_39.h): the object is
// an i32 handle and every member is an external call named as a member of the
// object's own class, so that hls::stream<T, D> keeps D in the name although
// the members are inherited from hls::stream<T, 0>. This is the IR form of the
// stand-in headers (shim/exactclass hls_stream.h) that hls-est's FIFO model
// reads (VITIS_NOTES N31); the headers' __fpga_fifo_* builtins are never
// emitted (OPEN_ISSUES O64 a).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_int.h>
#include <hls_stream.h>

struct pkt {
  ap_uint<12> a;
  int b;
};

// CHECK-LABEL: func.func @_Z3topPKiPi(
// CHECK-DAG: memref.alloca() {{.*}}variable = "s"{{.*}} : memref<1xi32>
// CHECK-DAG: memref.alloca() {{.*}}"p"{{.*}} : memref<1xi32>
// CHECK-DAG: memref.alloca() {{.*}}"t"{{.*}} : memref<1xi32>
// CHECK-DAG: memref.alloca() {{.*}}"arr"{{.*}} : memref<4xi32>
// CHECK-DAG: call @_ZN3hls6streamI7ap_uintILi16EELi0EEC1Ev(
// CHECK-DAG: call @_ZN3hls6streamI3pktLi0EEC1EPKc(
// CHECK-DAG: call @_ZN3hls6streamIiLi64EEC1Ev(
// CHECK-DAG: call @_ZN3hls6streamI7ap_uintILi16EELi0EE5writeERKS2_({{.*}}) : (memref<?xi32>, memref<?xi16>) -> ()
// CHECK-DAG: call @_ZN3hls6streamI3pktLi0EElsERKS1_({{.*}}) : (memref<?xi32>, memref<?x!llvm.struct<(i12, i32)>>) -> ()
// CHECK-DAG: call @_ZN3hls6streamIiLi64EE5writeERKi(
// CHECK-DAG: call @_ZN3hls6streamI7ap_uintILi16EELi0EE4readEv({{.*}}) : (memref<?xi32>) -> i16
// CHECK-DAG: call @_ZN3hls6streamI3pktLi0EErsERS1_(
// CHECK-DAG: call @_ZN3hls6streamIiLi64EE4readERi(
// CHECK-DAG: call @_ZNK3hls6streamI7ap_uintILi16EELi0EE5emptyEv({{.*}}) -> i8
// CHECK-DAG: call @_ZN3hls6streamIiLi8EE5writeERKi(
// CHECK-DAG: call @_ZN3hls6streamIiLi8EE7read_nbERi(
// CHECK-NOT: __fpga_fifo
// CHECK-DAG: func.func private @_ZN3hls6streamIiLi64EE4readERi(memref<?xi32>, memref<?xi32>) attributes {{{.*}}llvm.linkage = #llvm.linkage<external>}
void top(const int in[1024], int out[1024]) {
  hls::stream<ap_uint<16> > s;
#pragma HLS stream variable = s depth = 1024
  hls::stream<pkt> p("p");
  hls::stream<int, 64> t;
  hls::stream<int, 8> arr[4];
  for (int i = 0; i < 1024; i++) {
    s.write(in[i]);
    pkt q;
    q.a = in[i];
    q.b = i;
    p << q;
    t.write(i);
    arr[i % 4].write(i);
  }
  for (int i = 0; i < 1024; i++) {
    ap_uint<16> v = s.read();
    pkt q;
    p >> q;
    int w;
    t.read(w);
    int x = 0;
    arr[i % 4].read_nb(x);
    out[i] = v + q.a + q.b + w + x + (s.empty() ? 1 : 0);
  }
}
