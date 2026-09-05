#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void parboil_basic_sgemm(
    char transa, char transb, int m, int n, int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta,
    float *C, int ldc);

int main(void) {
  const int m = 2, n = 2, k = 3;
  float *a = (float *)malloc(6 * sizeof(float));
  float *b = (float *)malloc(6 * sizeof(float));
  float *c = (float *)malloc(4 * sizeof(float));
  if (!a || !b || !c) return 100;
  const float a_values[6] = {1, 4, 2, 5, 3, 6};
  const float b_values[6] = {7, 8, 9, 10, 11, 12};
  for (int i = 0; i < 6; ++i) a[i] = a_values[i], b[i] = b_values[i];
  for (int i = 0; i < 4; ++i) c[i] = 1.0f;

  parboil_basic_sgemm('N', 'T', m, n, k, 2.0f,
                       a, m, b, n, 3.0f, c, m);
  int ok = fabsf(c[0] - 119.0f) < 1.0e-4f &&
           fabsf(c[1] - 281.0f) < 1.0e-4f &&
           fabsf(c[2] - 131.0f) < 1.0e-4f &&
           fabsf(c[3] - 311.0f) < 1.0e-4f;
  if (!ok)
    fprintf(stderr, "unexpected C = [%g, %g, %g, %g]\n",
            c[0], c[1], c[2], c[3]);
  free(a);
  free(b);
  free(c);
  if (!ok) return 1;
  puts("parboil-source-faithful-sgemm: PASS");
  return 0;
}
