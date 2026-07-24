/* aten::relu numerical body.
 * Upstream family: aten/src/ATen/native/Activation.cpp.
 */
#define N 256

void aten_relu(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i)
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
#pragma endscop
}
