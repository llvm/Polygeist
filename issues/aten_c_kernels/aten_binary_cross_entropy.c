/* aten::binary_cross_entropy mean reduction. Upstream: ATen/native/Loss.cpp. */
#define N 256
void aten_binary_cross_entropy(float input[N], float target[N], float out[1]) {
#pragma scop
  out[0] = 0.0f;
  for (int i = 0; i < N; ++i)
    out[0] -= target[i] * __builtin_logf(input[i]) +
              (1.0f - target[i]) * __builtin_logf(1.0f - input[i]);
  out[0] /= (float)N;
#pragma endscop
}
