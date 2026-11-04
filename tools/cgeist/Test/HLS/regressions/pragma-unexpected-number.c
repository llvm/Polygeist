// A number where a pragma expects an option name: Vitis 2025.1 warns
// ([HLS 207-5571] "unexpected pragma argument '2', expects identifier") and
// drops the pragma. hlsyn-vitis aes puts `#pragma HLS pipeline` and
// `#pragma HLS unroll 2` on one loop; Vitis pipelines it
// (aes256_encrypt_ecb_Pipeline_L4) and does not unroll it. cgeist rejected
// the pragma with an error (W8b). Any other token where an option name is
// expected is treated the same way (`#pragma HLS pipeline =2`: 207-5571 and
// only the automatic pipelining, kernel w8bfix_peq, W8b review).
// RUN: cgeist %s --function='*' -S --hls-annotate -o - 2>&1 | FileCheck %s

// CHECK: warning: unexpected argument '2' in '#pragma HLS unroll', expected an option name; the pragma is ignored, as Vitis ignores it
// CHECK: warning: unexpected argument '=' in '#pragma HLS pipeline', expected an option name; the pragma is ignored, as Vitis ignores it
// CHECK-LABEL: func.func @f(
// CHECK-NOT:     hls.unroll
// CHECK:         hls.pipeline
void f(int a[64]) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline
#pragma HLS unroll 2
    a[i] = a[i] * 3;
  }
}

// CHECK-LABEL: func.func @g(
// CHECK-NOT:     hls.pipeline
// CHECK:         return
void g(int a[64]) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline =2
    a[i] = a[i] + 1;
  }
}
