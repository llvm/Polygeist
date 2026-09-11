/* aten::elu. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
extern float expf(float);
void aten_elu(float x[N], float out[N], float alpha, float scale) {
#pragma scop
  for (int i = 0; i < N; ++i)
    out[i] = scale * (x[i] > 0.0f ? x[i] : alpha * (expf(x[i]) - 1.0f));
#pragma endscop
}
