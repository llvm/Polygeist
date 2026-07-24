/* aten::mean over one contiguous tensor.
 * Upstream family: aten/src/ATen/native/ReduceOps.cpp.
 */
#define N 256

void aten_mean(double x[N], double out[1]) {
  double sum = 0.0;
#pragma scop
  for (int i = 0; i < N; ++i)
    sum += x[i];
  out[0] = sum / (double)N;
#pragma endscop
}
