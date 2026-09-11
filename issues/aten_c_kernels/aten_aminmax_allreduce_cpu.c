#ifndef N
#define N 4096
#endif
void aten_aminmax_allreduce_cpu(float x[N], float out_min[1], float out_max[1]) {
  float lo = x[0], hi = x[0];
  for (int i = 1; i < N; ++i) {
    lo = x[i] < lo ? x[i] : lo; hi = x[i] > hi ? x[i] : hi;
  }
  out_min[0] = lo; out_max[0] = hi;
}
