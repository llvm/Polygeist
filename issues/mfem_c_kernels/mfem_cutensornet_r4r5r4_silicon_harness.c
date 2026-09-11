#include <math.h>
#include <stdint.h>
#include <stdio.h>

extern void mfem_cutensornet_r4r5r4(
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t);
extern void mfem_cutensornet_r5r4r4(
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t);
extern void mfem_cutensornet_r5r5r4_broadcast(
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t,
    double *, double *, int64_t, int64_t, int64_t);

int main(void) {
  double a[48], b[96], c[24];
  for (int64_t i = 0; i < 48; ++i)
    a[i] = (double)((i * 7 + 3) % 29 - 14) / 11.0;
  for (int64_t i = 0; i < 96; ++i)
    b[i] = (double)((i * 5 + 1) % 23 - 11) / 13.0;
  for (int64_t i = 0; i < 24; ++i)
    c[i] = -777.0;

  mfem_cutensornet_r4r5r4(
      a, a, 0, 48, 1, b, b, 0, 96, 1, c, c, 0, 24, 1);

  double max_error = 0.0;
  for (int64_t d0 = 0; d0 < 2; ++d0)
    for (int64_t d1 = 0; d1 < 3; ++d1)
      for (int64_t d3 = 0; d3 < 2; ++d3)
        for (int64_t d2 = 0; d2 < 2; ++d2) {
          double expected = 0.0;
          for (int64_t d4 = 0; d4 < 4; ++d4) {
            int64_t a_offset = d0 * 24 + d1 * 8 + d4 * 2 + d2;
            int64_t b_offset =
                d0 * 48 + d3 * 24 + d2 * 12 + d1 * 4 + d4;
            expected += a[a_offset] * b[b_offset];
          }
          int64_t c_offset = d0 * 12 + d1 * 4 + d3 * 2 + d2;
          double error = fabs(c[c_offset] - expected);
          if (error > max_error)
            max_error = error;
        }
  printf("compiler_generated_r4r5r4 max_error=%.17g %s\n", max_error,
         max_error <= 1.0e-11 ? "PASS" : "FAIL");
  int failures = max_error > 1.0e-11;

  {
    double a5[188], b4[48], c4[24];
    for (int64_t i = 0; i < 188; ++i)
      a5[i] = (double)((i * 7 + 3) % 29 - 14) / 11.0;
    for (int64_t i = 0; i < 48; ++i)
      b4[i] = (double)((i * 5 + 1) % 23 - 11) / 13.0;
    for (int64_t i = 0; i < 24; ++i)
      c4[i] = -777.0;
    mfem_cutensornet_r5r4r4(
        a5, a5, 0, 188, 1, b4, b4, 0, 48, 1,
        c4, c4, 0, 24, 1);
    max_error = 0.0;
    for (int64_t d0 = 0; d0 < 2; ++d0)
      for (int64_t d1 = 0; d1 < 3; ++d1)
        for (int64_t d2 = 0; d2 < 2; ++d2)
          for (int64_t d3 = 0; d3 < 2; ++d3) {
            double expected = 0.0;
            for (int64_t d4 = 0; d4 < 4; ++d4) {
              int64_t a_offset =
                  d0 * 96 + d1 * 32 + d2 * 16 + d3 * 8 + d4;
              int64_t b_offset = d0 * 24 + d1 * 8 + d4 * 2 + d3;
              expected += a5[a_offset] * b4[b_offset];
            }
            int64_t c_offset = d0 * 12 + d1 * 4 + d2 * 2 + d3;
            double error = fabs(c4[c_offset] - expected);
            if (error > max_error)
              max_error = error;
          }
    printf("compiler_generated_r5r4r4 max_error=%.17g %s\n", max_error,
           max_error <= 1.0e-11 ? "PASS" : "FAIL");
    failures += max_error > 1.0e-11;
  }

  {
    double a5[48], b5[16], c4[24];
    for (int64_t i = 0; i < 48; ++i)
      a5[i] = (double)((i * 7 + 3) % 29 - 14) / 11.0;
    for (int64_t i = 0; i < 16; ++i)
      b5[i] = (double)((i * 5 + 1) % 23 - 11) / 13.0;
    for (int64_t i = 0; i < 24; ++i)
      c4[i] = -777.0;
    mfem_cutensornet_r5r5r4_broadcast(
        a5, a5, 0, 48, 1, b5, b5, 0, 16, 1,
        c4, c4, 0, 24, 1);
    max_error = 0.0;
    for (int64_t d0 = 0; d0 < 2; ++d0)
      for (int64_t d1 = 0; d1 < 3; ++d1)
        for (int64_t d2 = 0; d2 < 2; ++d2)
          for (int64_t d3 = 0; d3 < 2; ++d3) {
            double expected = 0.0;
            for (int64_t d4 = 0; d4 < 4; ++d4) {
              int64_t a_offset = d0 * 24 + d1 * 8 + d3 * 4 + d4;
              int64_t b_offset = d2 * 8 + d3 * 4 + d4;
              expected += a5[a_offset] * b5[b_offset];
            }
            int64_t c_offset = d0 * 12 + d1 * 4 + d2 * 2 + d3;
            double error = fabs(c4[c_offset] - expected);
            if (error > max_error)
              max_error = error;
          }
    printf("compiler_generated_r5r5r4_broadcast max_error=%.17g %s\n",
           max_error, max_error <= 1.0e-11 ? "PASS" : "FAIL");
    failures += max_error > 1.0e-11;
  }

  printf("compiler_generated_variants failures=%d\n", failures);
  return failures != 0;
}
