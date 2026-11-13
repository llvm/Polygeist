#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define KP 4
#define KQ 5

extern void tensor_product_3d_f64(
    const double psi[KQ * KP], const double u[KP * KP * KP],
    double out[KQ * KQ * KQ]);

int main(void) {
  double psi[KQ * KP], u[KP * KP * KP], out[KQ * KQ * KQ];
  for (int i = 0; i < KQ * KP; ++i) psi[i] = (double)(i - 7) / 13.0;
  for (int i = 0; i < KP * KP * KP; ++i)
    u[i] = (double)(i % 11 - 5) / 9.0;

  tensor_product_3d_f64(psi, u, out);

  double maxError = 0.0;
  for (int a = 0; a < KQ; ++a)
    for (int b = 0; b < KQ; ++b)
      for (int c = 0; c < KQ; ++c) {
        double reference = 0.0;
        for (int i = 0; i < KP; ++i)
          for (int j = 0; j < KP; ++j)
            for (int k = 0; k < KP; ++k)
              reference += psi[a * KP + i] * psi[b * KP + j] *
                           psi[c * KP + k] * u[(i * KP + j) * KP + k];
        double error = fabs(out[(a * KQ + b) * KQ + c] - reference);
        if (error > maxError) maxError = error;
      }
  printf("cutensornet f64 e2e max_err=%.17g\n", maxError);
  return maxError <= 1.0e-12 ? 0 : 1;
}
