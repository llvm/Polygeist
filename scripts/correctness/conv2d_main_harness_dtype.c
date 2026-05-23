/* conv2d_main_harness_dtype.c — dtype-parameterized main for the extracted
 * conv2d kernel. Compile with -DCTYPE=<scalar C type> (e.g. -DCTYPE=int or
 * -DCTYPE=short) and -DFMT=<printf fmt> (e.g. -DFMT='\"%d \"'). Falls back
 * to double + %.2lf when nothing is defined, matching the original f64
 * harness's behavior.
 *
 * Initialises A with a deterministic, dtype-appropriate fill, calls
 * kernel_conv2d, and dumps the interior of B to stderr.
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

#ifndef CTYPE
#define CTYPE double
#endif

/* Pick a sensible printf format from CTYPE_KIND. Caller defines exactly one
 * of -DCTYPE_KIND_INT, -DCTYPE_KIND_FLOAT, -DCTYPE_KIND_HALF; default is
 * float-style. Avoids the shell-quoting nightmare of passing a format
 * string through a -D macro. */
#if defined(CTYPE_KIND_INT)
  #define FMT "%d "
#elif defined(CTYPE_KIND_HALF)
  #define FMT "%.3f "
#else
  #define FMT "%.2f "
#endif

extern void kernel_conv2d(int ni, int nj, CTYPE *A, CTYPE *B);

int main(int argc, char **argv) {
  int ni = NI, nj = NJ;
  CTYPE *A = (CTYPE*)malloc((size_t)ni * (size_t)nj * sizeof(CTYPE));
  CTYPE *B = (CTYPE*)malloc((size_t)ni * (size_t)nj * sizeof(CTYPE));
  if (!A || !B) { fprintf(stderr, "alloc failed\n"); return 1; }

  /* Init A[i][j] = ((i+j) % 16) — small bounded values so int kernels don't
   * overflow at this NJ. For float dtypes this gives the same input domain
   * as the polybench (i+j)/nj formula up to a constant scale. */
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      A[(size_t)i * (size_t)nj + (size_t)j] = (CTYPE)((i + j) % 16);
  memset(B, 0, (size_t)ni * (size_t)nj * sizeof(CTYPE));

  kernel_conv2d(ni, nj, A, B);

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: B\n");
  for (int i = 1; i < ni - 1; ++i) {
    for (int j = 1; j < nj - 1; ++j) {
      if (((i - 1) * (nj - 2) + (j - 1)) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, FMT, B[(size_t)i * (size_t)nj + (size_t)j]);
    }
  }
  fprintf(stderr, "\nend   dump: B\n");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");

  free(A); free(B);
  return 0;
}
