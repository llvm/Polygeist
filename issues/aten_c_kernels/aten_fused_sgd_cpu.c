#ifndef N
#define N 4096
#endif
void aten_fused_sgd_cpu(float param[N], float grad[N], float momentum_buffer[N],
    float lr, float momentum, float dampening, float weight_decay,
    float grad_scale, int maximize, int first_step, int nesterov) {
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale; grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    if (momentum != 0.0f) {
      momentum_buffer[i] = first_step ? g :
          momentum_buffer[i] * momentum + g * (1.0f - dampening);
      g = nesterov ? g + momentum * momentum_buffer[i] : momentum_buffer[i];
    }
    param[i] -= lr * g;
  }
}
