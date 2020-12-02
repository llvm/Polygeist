// File name: main.c

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct OneDMemrefF32 {
  float *ptrToData;
  float *alignedPtrToData;
  long offset;
  long shape[1];
  long stride[1];
};

struct TwoDMemrefF32 {
  float *ptrToData;
  float *alignedPtrToData;
  long offset;
  long shape[2];
  long stride[2];
};

#define N 2000

extern void
_mlir_ciface_kernel_gemver(int, float, float, struct TwoDMemrefF32 *,
                           struct OneDMemrefF32 *, struct OneDMemrefF32 *,
                           struct OneDMemrefF32 *, struct OneDMemrefF32 *,
                           struct OneDMemrefF32 *, struct OneDMemrefF32 *,
                           struct OneDMemrefF32 *, struct OneDMemrefF32 *);
extern void _mlir_ciface_kernel_gemver_new(
    int, float alpha, float beta, struct TwoDMemrefF32 *,
    struct OneDMemrefF32 *, struct OneDMemrefF32 *, struct OneDMemrefF32 *,
    struct OneDMemrefF32 *, struct OneDMemrefF32 *, struct OneDMemrefF32 *,
    struct OneDMemrefF32 *, struct OneDMemrefF32 *);

int main(int argc, char *argv[]) {
  clock_t start, end;
  double orig_time, opt_time;

  int i, j;

  float *A1 = (float *)malloc(sizeof(float) * N * N);
  float *A2 = (float *)malloc(sizeof(float) * N * N);
  float *u1 = (float *)malloc(sizeof(float) * N);
  float *u2 = (float *)malloc(sizeof(float) * N);
  float *v1 = (float *)malloc(sizeof(float) * N);
  float *v2 = (float *)malloc(sizeof(float) * N);
  float *y = (float *)malloc(sizeof(float) * N);
  float *z = (float *)malloc(sizeof(float) * N);
  float *x1 = (float *)malloc(sizeof(float) * N);
  float *w1 = (float *)malloc(sizeof(float) * N);
  float *x2 = (float *)malloc(sizeof(float) * N);
  float *w2 = (float *)malloc(sizeof(float) * N);

  for (i = 0; i < N; i++) {
    u1[i] = (float)i;
    u2[i] = (float)(i + 1) / N / 2.0;
    v1[i] = (float)(i + 1) / N / 4.0;
    v2[i] = (float)(i + 1) / N / 6.0;
    y[i] = (float)(i + 1) / N / 8.0;
    z[i] = (float)(i + 1) / N / 9.0;
    x1[i] = 0.0;
    w1[i] = 0.0;
    x2[i] = 0.0;
    w2[i] = 0.0;
    for (j = 0; j < N; j++) {
      A1[i * N + j] = ((float)i * j) / N;
      A2[i * N + j] = ((float)i * j) / N;
    }
  }

  float alpha = 1.0;
  float beta = 1.0;

  struct TwoDMemrefF32 A1_mem = {A1, A1, 0, {N, N}, {N, 1}};
  struct TwoDMemrefF32 A2_mem = {A2, A2, 0, {N, N}, {N, 1}};
  struct OneDMemrefF32 u1_mem = {u1, u1, 0, {N}, {1}};
  struct OneDMemrefF32 u2_mem = {u2, u2, 0, {N}, {1}};
  struct OneDMemrefF32 v1_mem = {v1, v1, 0, {N}, {1}};
  struct OneDMemrefF32 v2_mem = {v2, v2, 0, {N}, {1}};
  struct OneDMemrefF32 y_mem = {y, y, 0, {N}, {1}};
  struct OneDMemrefF32 z_mem = {z, z, 0, {N}, {1}};
  struct OneDMemrefF32 x1_mem = {x1, x1, 0, {N}, {1}};
  struct OneDMemrefF32 w1_mem = {w1, w1, 0, {N}, {1}};
  struct OneDMemrefF32 x2_mem = {x2, x2, 0, {N}, {1}};
  struct OneDMemrefF32 w2_mem = {w2, w2, 0, {N}, {1}};

  int n = 1000;

  printf("Running original MLIR ...\n");
  start = clock();
  _mlir_ciface_kernel_gemver(n, alpha, beta, &A1_mem, &u1_mem, &v1_mem, &u2_mem,
                             &v2_mem, &w1_mem, &x1_mem, &y_mem, &z_mem);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

  printf("Running Pluto optimised MLIR ...\n");
  start = clock();
  _mlir_ciface_kernel_gemver_new(n, alpha, beta, &A2_mem, &u1_mem, &v1_mem,
                                 &u2_mem, &v2_mem, &w1_mem, &x1_mem, &y_mem,
                                 &z_mem);
  end = clock();
  opt_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", opt_time);

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (fabsf(A1[i * N + j] - A2[i * N + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d A1[i][j] = %f A2[i][j] = %f\n",
                i, j, A1[i * N + j], A2[i * N + j]);
      }

  printf("TEST passed!\n");

  free(A1);
  free(A2);
  free(u1);
  free(u2);
  free(v1);
  free(v2);
  free(y);
  free(z);
  free(x1);
  free(w1);
  free(x2);
  free(w2);

  return 0;
}
