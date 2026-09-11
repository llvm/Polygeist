/* llama2_tiny_forward.c -- self-contained Llama2-style forward fixture.
 *
 * This intentionally avoids checkpoint loading, tokenizer code, mmap, structs,
 * and file I/O.  The goal is to keep the numeric shape of a small inference
 * slice that Polygeist can lift as a whole kernel:
 *
 *   rmsnorm(x, weight) -> hidden
 *   logits = W * hidden
 *   softmax(logits)
 */

#include <math.h>
#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef N
#define N 16
#endif

#ifndef H
#define H 16
#endif

void kernel_llama2_tiny_forward(int n, int h, DATA_TYPE x[N],
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

static void init_array(DATA_TYPE x[N], DATA_TYPE weight[N],
                       DATA_TYPE w[H][N]) {
  for (int i = 0; i < N; ++i) {
    x[i] = (DATA_TYPE)((i % 7) - 3) * (DATA_TYPE)0.25;
    weight[i] = (DATA_TYPE)0.75 + (DATA_TYPE)((i % 5) + 1) * (DATA_TYPE)0.05;
  }
  for (int row = 0; row < H; ++row) {
    for (int col = 0; col < N; ++col) {
      w[row][col] = (DATA_TYPE)(((row * 3 + col * 5) % 13) - 6) *
                    (DATA_TYPE)0.03125;
    }
  }
}

static void print_array(DATA_TYPE logits[H]) {
  for (int i = 0; i < H; ++i) {
    printf("%.8f\n", (double)logits[i]);
  }
}

int main(void) {
  DATA_TYPE x[N];
  DATA_TYPE weight[N];
  DATA_TYPE w[H][N];
  DATA_TYPE hidden[N];
  DATA_TYPE logits[H];

  init_array(x, weight, w);
  kernel_llama2_tiny_forward(N, H, x, weight, w, hidden, logits);
  print_array(logits);
  return 0;
}
