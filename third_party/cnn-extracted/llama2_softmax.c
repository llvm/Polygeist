/* llama2_softmax.c — small standalone fixture for the llama2.c row softmax
 * kernel shape:
 *   x[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 */

#include <math.h>
#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef N
#define N 128
#endif

void kernel_llama2_softmax(DATA_TYPE x[N], int n) {
  DATA_TYPE max_val = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  DATA_TYPE sum = (DATA_TYPE)0;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

static void init_array(DATA_TYPE x[N]) {
  for (int i = 0; i < N; ++i)
    x[i] = (DATA_TYPE)((i % 23) - 11) * (DATA_TYPE)0.125;
}

static void print_array(DATA_TYPE x[N]) {
  for (int i = 0; i < N; ++i)
    printf("%.8f\n", (double)x[i]);
}

int main(void) {
  DATA_TYPE x[N];
  init_array(x);
  kernel_llama2_softmax(x, N);
  print_array(x);
  return 0;
}
