#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t at(int z, int y, int x, int h, int w) {
  return ((size_t)z * (size_t)h + (size_t)y) * (size_t)w + (size_t)x;
}

static void reference(int inD, int inH, int inW,
                      int outD, int outH, int outW,
                      int stride, int inOff, int outOff,
                      const double c[4], double alpha, double beta,
                      const double *input, const double *addend,
                      double *output) {
  if (output != addend)
    memcpy(output, addend,
           (size_t)outD * (size_t)outH * (size_t)outW * sizeof(double));
  int convD = outD - 2 * outOff;
  int convH = outH - 2 * outOff;
  int convW = outW - 2 * outOff;
  for (int oz = 0; oz < convD; ++oz)
    for (int oy = 0; oy < convH; ++oy)
      for (int ox = 0; ox < convW; ++ox) {
        double sum = 0.0;
        int iz = inOff + oz * stride;
        int iy = inOff + oy * stride;
        int ix = inOff + ox * stride;
        for (int dz = 0; dz < 3; ++dz)
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx)
              sum += c[(dz != 1) + (dy != 1) + (dx != 1)] *
                     input[at(iz + dz, iy + dy, ix + dx, inH, inW)];
        size_t dst = at(outOff + oz, outOff + oy, outOff + ox, outH, outW);
        output[dst] = alpha * sum + beta * addend[dst];
      }
}

static int run_case(const char *name, int inN, int outN, int stride,
                    int inOff, int outOff, const double c[4],
                    double alpha, double beta, int alias_addend) {
  size_t inCount = (size_t)inN * inN * inN;
  size_t outCount = (size_t)outN * outN * outN;
  double *input = (double *)malloc(inCount * sizeof(double));
  double *addend = (double *)malloc(outCount * sizeof(double));
  double *actual = alias_addend ? addend : (double *)malloc(outCount * sizeof(double));
  double *expected = (double *)malloc(outCount * sizeof(double));
  double *referenceAddend = (double *)malloc(outCount * sizeof(double));
  if (!input || !addend || !actual || !expected || !referenceAddend)
    return 1;
  for (size_t i = 0; i < inCount; ++i)
    input[i] = ((double)((int)(i % 29) - 14)) / 17.0;
  for (size_t i = 0; i < outCount; ++i)
    addend[i] = ((double)((int)(i % 13) - 6)) / 11.0;
  memcpy(referenceAddend, addend, outCount * sizeof(double));
  if (!alias_addend)
    memset(actual, 0, outCount * sizeof(double));
  memcpy(expected, referenceAddend, outCount * sizeof(double));
  reference(inN, inN, inN, outN, outN, outN, stride, inOff, outOff,
            c, alpha, beta, input, referenceAddend, expected);
  polygeist_cudnn_stencil3d_symmetric_f64(
      inN, inN, inN, outN, outN, outN, stride, stride, stride,
      inOff, inOff, inOff, outOff, outOff, outOff,
      c[0], c[1], c[2], c[3], alpha, beta, input, addend, actual);
  double maxError = 0.0;
  for (size_t i = 0; i < outCount; ++i) {
    double error = fabs(actual[i] - expected[i]);
    if (error > maxError)
      maxError = error;
  }
  printf("%s max_abs=%.3e %s\n", name, maxError,
         maxError < 1.0e-11 ? "PASS" : "FAIL");
  free(input);
  if (!alias_addend)
    free(actual);
  free(addend);
  free(expected);
  free(referenceAddend);
  return maxError < 1.0e-11 ? 0 : 1;
}

int main(void) {
  const double resid[4] = {-8.0 / 3.0, 0.0, 1.0 / 6.0, 1.0 / 12.0};
  const double psinv[4] = {-3.0 / 8.0, 1.0 / 32.0, -1.0 / 64.0, 0.0};
  const double restrict3[4] = {0.5, 0.25, 0.125, 0.0625};
  int failed = 0;
  failed |= run_case("resid", 7, 7, 1, 0, 1, resid, -1.0, 1.0, 0);
  failed |= run_case("psinv", 7, 7, 1, 0, 1, psinv, 1.0, 1.0, 1);
  failed |= run_case("rprj3", 9, 5, 2, 1, 1, restrict3, 1.0, 0.0, 1);
  return failed;
}
