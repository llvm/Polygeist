// File name: main.c

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

extern void _mlir_ciface_kernel_lu(int, struct TwoDMemrefF64 *);
extern void _mlir_ciface_kernel_lu_new(int, struct TwoDMemrefF64 *);

static void init_array(int n, double *A) {
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      A[i * n + j] = (double)(-j % n) / n + 1;
    for (j = i + 1; j < n; j++) {
      A[i * n + j] = 0;
    }
    A[i * n + i] = 1;
  }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r, s, t;
  double *B = (double *)malloc(sizeof(double) * N * N);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      B[r * n + s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
        B[r * n + s] += A[r * n + t] * A[s * n + t];
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      A[r * n + s] = B[r * n + s];
  free(B);
}

static void kernel_lu(int n, double *A) {
  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      for (k = 0; k < j; k++) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
      }
      A[i * n + j] /= A[j * n + j];
    }
    for (j = i; j < n; j++) {
      for (k = 0; k < i; k++) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double orig_time, opt_time;

  int i, j;

  double *A1 = (double *)malloc(sizeof(double) * N * N);
  double *A2 = (double *)malloc(sizeof(double) * N * N);
  double *A3 = (double *)malloc(sizeof(double) * N * N);

  int n = 2000;
  init_array(n, A1);
  memcpy(A2, A1, sizeof(double) * N * N);
  memcpy(A3, A1, sizeof(double) * N * N);

  struct TwoDMemrefF64 A1_mem = {A1, A1, 0, {N, N}, {N, 1}};
  struct TwoDMemrefF64 A2_mem = {A2, A2, 0, {N, N}, {N, 1}};

  printf("Running original C ...\n");
  start = clock();
  kernel_lu(n, A3);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

  printf("Running original MLIR ...\n");
  start = clock();
  _mlir_ciface_kernel_lu(n, &A1_mem);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

  printf("Running Pluto optimised MLIR ...\n");
  start = clock();
  _mlir_ciface_kernel_lu_new(n, &A2_mem);
  end = clock();
  opt_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", opt_time);

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      if (fabs(A3[i * n + j] - A1[i * n + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d A1[i][j] = %f A3[i][j] = %f\n",
                i, j, A1[i * n + j], A3[i * n + j]);
        return 1;
      }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      if (fabs(A3[i * n + j] - A2[i * n + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d A2[i][j] = %f A3[i][j] = %f\n",
                i, j, A2[i * n + j], A3[i * n + j]);
        return 1;
      }

  printf("TEST passed!\n");

  free(A1);
  free(A2);
  free(A3);

  return 0;
}
