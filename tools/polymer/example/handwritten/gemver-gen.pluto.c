// File name: main.c

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct OneDMemrefF64 {
  double *ptrToData;
  double *alignedPtrToData;
  long offset;
  long shape[1];
  long stride[1];
};

struct TwoDMemrefF64 {
  double *ptrToData;
  double *alignedPtrToData;
  long offset;
  long shape[2];
  long stride[2];
};

#define N 2000

extern void
_mlir_ciface_kernel_gemver(int, double, double, struct TwoDMemrefF64 *,
                           struct OneDMemrefF64 *, struct OneDMemrefF64 *,
                           struct OneDMemrefF64 *, struct OneDMemrefF64 *,
                           struct OneDMemrefF64 *, struct OneDMemrefF64 *,
                           struct OneDMemrefF64 *, struct OneDMemrefF64 *);
extern void _mlir_ciface_kernel_gemver_new(
    int, double alpha, double beta, struct TwoDMemrefF64 *,
    struct OneDMemrefF64 *, struct OneDMemrefF64 *, struct OneDMemrefF64 *,
    struct OneDMemrefF64 *, struct OneDMemrefF64 *, struct OneDMemrefF64 *,
    struct OneDMemrefF64 *, struct OneDMemrefF64 *);

static void kernel_gemver(int n, double alpha, double beta, double *A,
                          double *u1, double *v1, double *u2, double *v2,
                          double *w, double *x, double *y, double *z) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i * N + j] = A[i * N + j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j * N + i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] + alpha * A[i * N + j] * x[j];
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double orig_time, opt_time;

  int i, j;

  double *A1 = (double *)malloc(sizeof(double) * N * N);
  double *A2 = (double *)malloc(sizeof(double) * N * N);
  double *A3 = (double *)malloc(sizeof(double) * N * N);
  double *u1 = (double *)malloc(sizeof(double) * N);
  double *u2 = (double *)malloc(sizeof(double) * N);
  double *v1 = (double *)malloc(sizeof(double) * N);
  double *v2 = (double *)malloc(sizeof(double) * N);
  double *y = (double *)malloc(sizeof(double) * N);
  double *z = (double *)malloc(sizeof(double) * N);
  double *x1 = (double *)malloc(sizeof(double) * N);
  double *w1 = (double *)malloc(sizeof(double) * N);
  double *x2 = (double *)malloc(sizeof(double) * N);
  double *w2 = (double *)malloc(sizeof(double) * N);

  for (i = 0; i < N; i++) {
    u1[i] = (double)i;
    u2[i] = (double)(i + 1) / N / 2.0;
    v1[i] = (double)(i + 1) / N / 4.0;
    v2[i] = (double)(i + 1) / N / 6.0;
    y[i] = (double)(i + 1) / N / 8.0;
    z[i] = (double)(i + 1) / N / 9.0;
    x1[i] = 0.0;
    w1[i] = 0.0;
    x2[i] = 0.0;
    w2[i] = 0.0;
    for (j = 0; j < N; j++) {
      A1[i * N + j] = ((double)i * j) / N;
      A2[i * N + j] = ((double)i * j) / N;
      A3[i * N + j] = ((double)i * j) / N;
    }
  }

  double alpha = 1.0;
  double beta = 1.0;

  struct TwoDMemrefF64 A1_mem = {A1, A1, 0, {N, N}, {N, 1}};
  struct TwoDMemrefF64 A2_mem = {A2, A2, 0, {N, N}, {N, 1}};
  struct OneDMemrefF64 u1_mem = {u1, u1, 0, {N}, {1}};
  struct OneDMemrefF64 u2_mem = {u2, u2, 0, {N}, {1}};
  struct OneDMemrefF64 v1_mem = {v1, v1, 0, {N}, {1}};
  struct OneDMemrefF64 v2_mem = {v2, v2, 0, {N}, {1}};
  struct OneDMemrefF64 y_mem = {y, y, 0, {N}, {1}};
  struct OneDMemrefF64 z_mem = {z, z, 0, {N}, {1}};
  struct OneDMemrefF64 x1_mem = {x1, x1, 0, {N}, {1}};
  struct OneDMemrefF64 w1_mem = {w1, w1, 0, {N}, {1}};
  struct OneDMemrefF64 x2_mem = {x2, x2, 0, {N}, {1}};
  struct OneDMemrefF64 w2_mem = {w2, w2, 0, {N}, {1}};

  int n = 2000;
  printf("Running original C ...\n");
  start = clock();
  kernel_gemver(n, alpha, beta, A3, u1, v1, u2, v2, w1, x1, y, z);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

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
      if (fabs(A3[i * N + j] - A1[i * N + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d A1[i][j] = %f A3[i][j] = %f\n",
                i, j, A1[i * N + j], A3[i * N + j]);
        return 1;
      }
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (fabs(A3[i * N + j] - A2[i * N + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d A2[i][j] = %f A3[i][j] = %f\n",
                i, j, A2[i * N + j], A3[i * N + j]);
        return 1;
      }

  printf("TEST passed!\n");

  free(A1);
  free(A2);
  free(A3);
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
