/* aten::silu numerical body.
 * Upstream family: aten/src/ATen/native/Activation.cpp.
 */
#define N 256
extern float expf(float);

void aten_silu(float x[N], float out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i)
    out[i] = x[i] / (1.0f + expf(-x[i]));
#pragma endscop
}
