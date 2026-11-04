// Pragmas Vitis 2025.1 accepts and cgeist used to reject (CIRCT OPEN_ISSUES
// O219 C3, C5-C8; each checked with a Vitis kernel w9r22i_c*):
//  - `bind_storage type=xpm_memory` is a valid type (c3_xpm fails only on
//    impl=uram for the part, [HLS 214-370]);
//  - `loop_tripcount` without `max` is ignored (c5: no message, trip count
//    '?');
//  - `variable=&s` drops the `&` ([HLS 207-5544], c6);
//  - array_partition on a scalar of class type (an ap_uint) is ignored (c7);
//  - interface on a local variable is ignored ([HLS 214-238], c8).
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

// CHECK-DAG: hls.bind_storage = [{impl = "bram", type = "xpm_memory", variable = "t"}]
// CHECK-DAG: hls.stream = [{depth = 4 : i32, type = "fifo", variable = "s"}]
// CHECK-NOT: hls.loop_tripcount
// CHECK-NOT: hls.interface = [{{.*}}hist
// WARN-DAG: warning: '#pragma HLS loop_tripcount' without 'max' is ignored, as Vitis ignores it
// WARN-DAG: warning: extra token before variable expression is ignored in '#pragma HLS stream', as Vitis ignores it
// WARN-DAG: warning: '#pragma HLS array_partition' on the scalar 'l_valIn' of type 'ap_uint<64>' is ignored, as Vitis ignores it
// WARN-DAG: warning: 'hist' in '#pragma HLS interface' is not a function parameter; the pragma is ignored, as Vitis ignores it
#include <ap_int.h>
#include <hls_stream.h>
void top(const int in[64], int out[64], hls::stream<int> &sin, hls::stream<int> &sout, ap_uint<64> *x, int n) {
  int t[64], hist[64];
#pragma HLS bind_storage variable = t type = xpm_memory impl = bram
#pragma HLS interface mode = bram port = hist
  for (int i = 0; i < 64; i++) { t[i] = in[i]; hist[i] = in[i]; }
  for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount min = 1 avg = 32
    out[i] = t[63 - i] + hist[i];
  }
  hls::stream<int> s;
#pragma HLS stream variable = &s depth = 4
  for (int i = 0; i < 4; i++) s.write(sin.read());
  for (int i = 0; i < 4; i++) sout.write(s.read());
  ap_uint<64> l_valIn = 0;
#pragma HLS array_partition variable = l_valIn complete
  for (int i = 0; i < 8; i++) l_valIn ^= x[i];
  x[8] = l_valIn;
}
