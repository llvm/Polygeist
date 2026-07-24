/* aten::hardswish. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
void aten_hardswish(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float v = x[i] + 3.0f;
    v = v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
    out[i] = x[i] * v / 6.0f;
  }
#pragma endscop
}
