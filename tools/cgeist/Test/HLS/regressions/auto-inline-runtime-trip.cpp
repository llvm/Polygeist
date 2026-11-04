// VITIS_NOTES N161 (hls-est tree): a loop marked for a complete unroll whose
// trip count is a run-time value of its function (`i < length / 4`, length a
// parameter) stays one loop in the automatic-inlining size count when its
// maximum trip count exceeds --hls-auto-inline-unroll-budget: Vitis inlines
// such a callee and unrolls the loop once the trip count is constant
// (class_ece8893_spring_2025 barry_walker lab2: `[HLS 214-178] Inlining
// function 'accumulate_quarter_HLS<...>' into 'softmax_HLS(...)'`). Off: the
// loop is counted at its maximum trip count (2^29) and the callee is kept.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report --hls-auto-inline-runtime-trip-kept=false -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

// CHECK-DAG: caller 'top' callee 'accumulate_quarter' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)
// CHECK-DAG: caller 'top2' callee 'accumulate_quarter' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)
// CHECK-NOT: is not unrolled for measuring
// OFF-DAG: is not unrolled for measuring: over 65536 instructions after the unroll
// OFF-DAG: caller 'top' callee 'accumulate_quarter' size 65537 calls 1: kept (size x calls > threshold)

template <typename In, typename Out>
void accumulate_quarter(In source[], Out &dest, int length) {
  Out s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  for (int i = 0; i < length / 4; i++) {
#pragma HLS unroll
    s1 += source[i];
    s2 += source[i + length / 4];
    s3 += source[i + (length / 4) * 2];
    s4 += source[i + (length / 4) * 3];
  }
  dest = (s1 + s2) + (s3 + s4);
}
void top(int in[16][100], int out[16]) {
  for (int r = 0; r < 16; r++) {
#pragma HLS pipeline
    int row[100];
    for (int j = 0; j < 100; j++) row[j] = in[r][j];
    int s = 0;
    accumulate_quarter<int, int>(row, s, 100);
    out[r] = s;
  }
}
void top2(int in[16][128], int out[16]) {
  for (int r = 0; r < 16; r++) {
#pragma HLS pipeline
    int row[128];
    for (int j = 0; j < 128; j++) row[j] = in[r][j];
    int s = 0;
    accumulate_quarter<int, int>(row, s, 128);
    out[r] = s;
  }
}
