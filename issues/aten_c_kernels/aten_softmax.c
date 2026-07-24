/* aten::_softmax over one contiguous row. */
#define N 128
#define NEG_INF (-3.402823466e38f)
extern float expf(float);

void aten_softmax(float x[N]) {
  float max_value = NEG_INF;
#pragma scop
  for (int i = 0; i < N; ++i)
    max_value = x[i] > max_value ? x[i] : max_value;
  float sum = 0.0f;
  for (int i = 0; i < N; ++i) {
    x[i] = expf(x[i] - max_value);
    sum += x[i];
  }
  for (int i = 0; i < N; ++i)
    x[i] /= sum;
#pragma endscop
}
