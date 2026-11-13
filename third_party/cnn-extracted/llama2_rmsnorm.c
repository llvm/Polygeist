/* llama2_rmsnorm.c — small standalone fixture for the llama2.c RMSNorm
 * kernel shape:
 *   ss = sum(x[i] * x[i])
 *   out[i] = weight[i] * x[i] * rsqrt(ss / N + 1e-5)
 */

#include <math.h>
#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef N
#define N 128
#endif

void kernel_llama2_rmsnorm(int n, DATA_TYPE o[N], DATA_TYPE x[N],
                           DATA_TYPE weight[N]) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int j = 0; j < n; j++) {
    ss += x[j] * x[j];
  }
  ss /= n;
  ss += (DATA_TYPE)1.0e-5;
  ss = (DATA_TYPE)1 / sqrtf(ss);
  for (int j = 0; j < n; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
#pragma endscop
}

static void init_array(DATA_TYPE x[N], DATA_TYPE weight[N]) {
  for (int i = 0; i < N; ++i) {
    x[i] = (DATA_TYPE)((i % 17) - 8) * (DATA_TYPE)0.125;
    weight[i] = (DATA_TYPE)0.5 + (DATA_TYPE)((i % 11) + 1) * (DATA_TYPE)0.03125;
  }
}

static void print_array(DATA_TYPE o[N]) {
  for (int i = 0; i < N; ++i)
    printf("%.8f\n", (double)o[i]);
}

int main(void) {
  DATA_TYPE o[N];
  DATA_TYPE x[N];
  DATA_TYPE weight[N];
  init_array(x, weight);
  kernel_llama2_rmsnorm(N, o, x, weight);
  print_array(o);
  return 0;
}
