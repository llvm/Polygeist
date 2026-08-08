#ifndef N
#define N 4096
#endif
extern float sqrtf(float);
void aten_fused_adam_cpu(float param[N], float grad[N], float exp_avg[N],
    float exp_avg_sq[N], float max_exp_avg_sq[N], float lr, float beta1,
    float beta2, float bias1, float bias2_sqrt, float weight_decay, float eps,
    float grad_scale, int maximize, int amsgrad) {
  float step_size = lr / bias1;
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale; grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    exp_avg[i] += (1.0f - beta1) * (g - exp_avg[i]);
    exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
    float variance = exp_avg_sq[i];
    if (amsgrad) {
      max_exp_avg_sq[i] = max_exp_avg_sq[i] > variance ? max_exp_avg_sq[i] : variance;
      variance = max_exp_avg_sq[i];
    }
    param[i] -= step_size * exp_avg[i] / (sqrtf(variance) / bias2_sqrt + eps);
  }
}
