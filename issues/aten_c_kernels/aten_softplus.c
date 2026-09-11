/* aten::softplus. Upstream: ATen/native/cpu/Activation.cpp. */
#define N 256
extern float expf(float);
void aten_softplus(float x[N], float out[N], float beta, float threshold) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float z = beta * x[i];
    out[i] = z > threshold ? x[i] :
             __builtin_logf(1.0f + expf(z)) / beta;
  }
#pragma endscop
}
