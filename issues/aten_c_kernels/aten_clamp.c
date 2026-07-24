/* aten::clamp. Upstream: ATen/native/TensorCompare.cpp. */
#define N 256
void aten_clamp(float x[N], float out[N], float lo, float hi) {
#pragma scop
  for (int i = 0; i < N; ++i)
    out[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
#pragma endscop
}
