#include <math.h>
#include <stdio.h>

#include "../runtime/polygeist_cublas_rt.h"

static int close_double(double a, double b) {
  double d = fabs(a - b);
  return d < 1.0e-9;
}

static int close_float(float a, float b) {
  float d = fabsf(a - b);
  return d < 1.0e-4f;
}

int main(void) {
  const double in64[8] = {
      1.0, 0.0,
      2.0, 0.0,
      0.0, 0.0,
     -1.0, 0.0,
  };
  const double expect64[8] = {
      2.0, 0.0,
      1.0, -3.0,
      0.0, 0.0,
      1.0, 3.0,
  };
  double out64[8] = {0};
  polygeist_cufft_z2z_1d(4, 0, in64, out64);
  for (int i = 0; i < 8; ++i) {
    if (!close_double(out64[i], expect64[i])) {
      fprintf(stderr, "z2z mismatch at %d: got %.17g expected %.17g\n",
              i, out64[i], expect64[i]);
      return 1;
    }
  }

  const float in32[8] = {
      1.0f, 0.0f,
      2.0f, 0.0f,
      0.0f, 0.0f,
     -1.0f, 0.0f,
  };
  const float expect32[8] = {
      2.0f, 0.0f,
      1.0f, -3.0f,
      0.0f, 0.0f,
      1.0f, 3.0f,
  };
  float out32[8] = {0};
  polygeist_cufft_c2c_1d(4, 0, in32, out32);
  for (int i = 0; i < 8; ++i) {
    if (!close_float(out32[i], expect32[i])) {
      fprintf(stderr, "c2c mismatch at %d: got %.9g expected %.9g\n",
              i, out32[i], expect32[i]);
      return 1;
    }
  }

  printf("cufft_runtime_smoke ok\n");
  return 0;
}
