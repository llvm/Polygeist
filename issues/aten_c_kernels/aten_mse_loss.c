/* aten::mse_loss with mean reduction.
 * Upstream family: aten/src/ATen/native/Loss.cpp.
 */
#define N 256

void aten_mse_loss(float input[N], float target[N], float scratch[N],
                   float out[1]) {
  float sum = 0.0f;
#pragma scop
  for (int i = 0; i < N; ++i) {
    float diff = input[i] - target[i];
    scratch[i] = diff * diff;
  }
  for (int i = 0; i < N; ++i)
    sum += scratch[i];
  out[0] = sum / (float)N;
#pragma endscop
}
