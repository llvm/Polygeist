/* aten::gelu tanh approximation.
 * Upstream family: aten/src/ATen/native/Activation.cpp.
 */
#define N 256
extern float tanhf(float);

void aten_gelu(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    float v = x[i];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
#pragma endscop
}
