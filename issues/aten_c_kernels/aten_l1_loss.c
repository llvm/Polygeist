/* aten::l1_loss with mean reduction. Upstream: ATen/native/Loss.cpp. */
#define N 256
void aten_l1_loss(float input[N], float target[N], float out[1]) {
#pragma scop
  out[0] = 0.0f;
  for (int i = 0; i < N; ++i) {
    float d = input[i] - target[i];
    out[0] += d < 0.0f ? -d : d;
  }
  out[0] /= (float)N;
#pragma endscop
}
