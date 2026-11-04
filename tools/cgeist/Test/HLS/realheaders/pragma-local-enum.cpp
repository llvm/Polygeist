// An integer pragma value naming an enumerator of an unscoped enum declared in
// the function body, here in a function template:
// `enum { fifo_depth = burst_len * 2 }; #pragma HLS STREAM ... depth = fifo_depth`
// (Vitis_Libraries database scan_col_2.hpp; Vitis 2025.1 synthesises it).
// CIRCT OPEN_ISSUES O219.
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-DAG: hls.loop_tripcount = [{avg = 8 : i32, max = 16 : i32, min = 0 : i32}]
// CHECK-DAG: hls.pipeline = [{II = 3 : i32
template <int N> void work(const int *in, int *out, int n) {
  enum { TRIPS = N * 2, II = 3 };
  for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II = II
#pragma HLS loop_tripcount max = TRIPS
    out[k] = in[k] * 3;
  }
}
void top(const int in[64], int out[64], int n) { work<8>(in, out, n); }
