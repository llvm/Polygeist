// Regression (HLSFactory DGNN_Booster): `bind_storage` with only `impl` (no
// `type`) is accepted by Vitis but was rejected here. `type` is optional; when
// omitted it is left out of the attribute (the tool chooses).
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK: hls.bind_storage = [{impl = "bram", variable = "weights"}]
float kernel(float *x) {
  float weights[64];
#pragma HLS bind_storage variable=weights impl = bram
  for (int i = 0; i < 64; ++i)
    weights[i] = x[i];
  float s = 0;
  for (int i = 0; i < 64; ++i)
    s += weights[63 - i] * x[i];
  return s;
}
