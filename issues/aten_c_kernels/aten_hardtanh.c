/* aten::hardtanh. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
void aten_hardtanh(float x[N], float out[N], float lo, float hi) {
#pragma scop
  for (int i = 0; i < N; ++i)
    out[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
#pragma endscop
}
