// VITIS_NOTES N161 (b) (hls-est tree): a loop to unroll completely whose trip
// count is a parameter of its function is counted at the trip count the calls
// give it when every call passes the same constant: Vitis propagates the
// constant into the callee, unrolls the loop there and measures the unrolled
// callee (kernel w9r21rev_r9b_rtunroll_big: `[HLS 214-186] Unrolling loop
// ... in function 'accq<short>' completely with a factor of 16`, kept as
// grp_accq_short_s). A run-time length (top_rt) or two different constants
// (auto-inline-runtime-trip.cpp) leave the loop counted once, and Vitis
// inlines those callees.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report --hls-auto-inline-callsite-constant-trips=false -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

// CHECK-DAG: '{{.*}}accq{{.*}}': parameters every call passes as one constant are propagated before measuring
// CHECK-DAG: caller 'top' callee 'accq' size {{[0-9][0-9][0-9]+}} calls 1: kept (size x calls > threshold)
// CHECK-DAG: caller 'top_rt' callee 'accq_rt' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)
// OFF-NOT: are propagated before measuring
// OFF-DAG: caller 'top' callee 'accq' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)

template <typename T>
void accq(T src[64], T w[64], int &dest, int length) {
  int s = 0;
  for (int i = 0; i < length / 4; i++) {
#pragma HLS unroll
    for (int q = 0; q < 4; q++)
      s += src[4 * i + q] * w[4 * i + q] * (src[4 * i + q] ^ w[(4 * i + q + 1) & 63]) +
           (src[4 * i + q] >> (q + 1)) * (w[4 * i + q] - 3);
  }
  dest = s;
}
void top(short in[100][64], short w[64], int out[100]) {
  short wl[64];
  for (int k = 0; k < 64; k++) wl[k] = w[k];
  for (int r = 0; r < 100; r++) {
#pragma HLS pipeline
    short row[64];
    for (int k = 0; k < 64; k++) row[k] = in[r][k];
    int d;
    accq<short>(row, wl, d, 64);
    out[r] = d;
  }
}
template <typename T>
void accq_rt(T src[64], int &dest, int length) {
  int s = 0;
  for (int i = 0; i < length / 4; i++) {
#pragma HLS unroll
    s += src[4 * i] + src[4 * i + 1] + src[4 * i + 2] + src[4 * i + 3];
  }
  dest = s;
}
void top_rt(short in[100][64], int len[100], int out[100]) {
  for (int r = 0; r < 100; r++) {
#pragma HLS pipeline
    short row[64];
    for (int k = 0; k < 64; k++) row[k] = in[r][k];
    int d;
    accq_rt<short>(row, d, len[r] & 63);
    out[r] = d;
  }
}
