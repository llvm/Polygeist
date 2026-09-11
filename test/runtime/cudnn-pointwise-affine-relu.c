#ifndef POINTWISE_N
#define POINTWISE_N 4194304
#endif

void pointwise_affine_relu(float x[POINTWISE_N],
                           float bias[POINTWISE_N],
                           float alpha,
                           float out[POINTWISE_N]) {
#pragma scop
  for (int i = 0; i < POINTWISE_N; ++i) {
    float affine = alpha * x[i] + bias[i];
    out[i] = affine > 0.0f ? affine : 0.0f;
  }
#pragma endscop
}
