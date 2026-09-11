/* llama2_forward_bench.c -- larger Llama2-style forward fixture.
 *
 * Same numeric shape as llama2_tiny_forward.c, but sized large enough that
 * cuBLAS/cuDNN setup overhead is not the entire experiment:
 *
 *   rmsnorm(x, weight) -> hidden
 *   logits = W * hidden
 *   softmax(logits)
 *
 * Defaults are intentionally moderate for Jetson iteration. Override with
 * -DN=4096 -DH=32000 for a Llama-7B-like output projection size.
 */

#include <math.h>
#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef N
#define N 1024
#endif

#ifndef H
#define H 4096
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

#ifndef PRINT_ELEMS
#define PRINT_ELEMS 32
#endif

void kernel_llama2_forward_bench(int n, int h, DATA_TYPE x[N],
                                 DATA_TYPE weight[N], DATA_TYPE w[H][N],
                                 DATA_TYPE hidden[N], DATA_TYPE logits[H]) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }

  ss /= n;
  ss += (DATA_TYPE)1.0e-5;
  ss = (DATA_TYPE)1 / sqrtf(ss);

  for (int i = 0; i < n; ++i) {
    hidden[i] = weight[i] * (ss * x[i]);
  }

  for (int row = 0; row < h; ++row) {
    logits[row] = (DATA_TYPE)0;
  }

  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < n; ++col) {
      logits[row] += w[row][col] * hidden[col];
    }
  }

  DATA_TYPE max_val = logits[0];
  for (int i = 1; i < h; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }

  DATA_TYPE sum = (DATA_TYPE)0;
  for (int i = 0; i < h; ++i) {
    logits[i] = expf(logits[i] - max_val);
    sum += logits[i];
  }

  for (int i = 0; i < h; ++i) {
    logits[i] /= sum;
  }
#pragma endscop
}

static DATA_TYPE x[N];
static DATA_TYPE weight[N];
static DATA_TYPE w[H][N];
static DATA_TYPE hidden[N];
static DATA_TYPE logits[H];

static void init_array(void) {
  for (int i = 0; i < N; ++i) {
    x[i] = (DATA_TYPE)((i % 31) - 15) * (DATA_TYPE)0.0625;
    weight[i] = (DATA_TYPE)0.75 + (DATA_TYPE)((i % 17) + 1) *
                                      (DATA_TYPE)0.015625;
  }
  for (int row = 0; row < H; ++row) {
    for (int col = 0; col < N; ++col) {
      w[row][col] = (DATA_TYPE)(((row * 7 + col * 11) % 29) - 14) *
                    (DATA_TYPE)0.0078125;
    }
  }
}

static void print_array(void) {
  int nprint = PRINT_ELEMS < H ? PRINT_ELEMS : H;
  DATA_TYPE checksum = (DATA_TYPE)0;
  for (int i = 0; i < H; ++i) {
    checksum += logits[i];
  }
  for (int i = 0; i < nprint; ++i) {
    printf("%.8f\n", (double)logits[i]);
  }
  printf("%.8f\n", (double)checksum);
}

int main(void) {
  init_array();
  for (int r = 0; r < REPEAT; ++r) {
    kernel_llama2_forward_bench(N, H, x, weight, w, hidden, logits);
  }
  print_array();
  return 0;
}
