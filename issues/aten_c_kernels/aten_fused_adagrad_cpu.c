#ifndef N
#define N 4096
#endif
extern float sqrtf(float);
void aten_fused_adagrad_cpu(float param[N], float grad[N], float state_sum[N],
    float lr, float lr_decay, float weight_decay, float eps, float step,
    float grad_scale, int maximize) {
  float clr = lr / (1.0f + (step - 1.0f) * lr_decay);
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale;
    grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    state_sum[i] += g * g;
    param[i] -= clr * g / (sqrtf(state_sum[i]) + eps);
  }
}
