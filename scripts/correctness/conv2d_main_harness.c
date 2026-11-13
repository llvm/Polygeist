/* conv2d_main_harness.c — minimal main for the extracted conv2d kernel.
 *
 * The polybenchGpu-extracted/conv2d.c file has no main (that's the point of
 * the extraction). We provide a minimal one that initialises A with the
 * polybench-style A[i][j] = (i+j)/nj formula, calls kernel_conv2d, and
 * dumps the interior of B to stderr so a diff vs a reference build can
 * confirm correctness.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

extern void kernel_conv2d(int ni, int nj, double *A, double *B);

int main(int argc, char **argv) {
  int ni = NI, nj = NJ;
  /* Heap-allocate so we don't blow the stack for larger NI/NJ. */
  double *A = (double*)malloc((size_t)ni * (size_t)nj * sizeof(double));
  double *B = (double*)malloc((size_t)ni * (size_t)nj * sizeof(double));
  if (!A || !B) { fprintf(stderr, "alloc failed\n"); return 1; }

  /* Init A[i][j] = (i + j) / nj  — same as polybench's init_array. */
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      A[(size_t)i * (size_t)nj + (size_t)j] = ((double)(i + j)) / (double)nj;
  memset(B, 0, (size_t)ni * (size_t)nj * sizeof(double));

  kernel_conv2d(ni, nj, A, B);

  /* Dump interior of B (skip border) to stderr — polybench-style. */
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: B\n");
  for (int i = 1; i < ni - 1; ++i) {
    for (int j = 1; j < nj - 1; ++j) {
      if (((i - 1) * (nj - 2) + (j - 1)) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", B[(size_t)i * (size_t)nj + (size_t)j]);
    }
  }
  fprintf(stderr, "\nend   dump: B\n");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");

  free(A); free(B);
  return 0;
}
