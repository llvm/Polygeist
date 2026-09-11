/* aten::hardsigmoid. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
void aten_hardsigmoid(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float v = (x[i] + 3.0f) / 6.0f;
    out[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  }
#pragma endscop
}
