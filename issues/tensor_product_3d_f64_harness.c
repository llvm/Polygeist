#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define KP 4
#define KQ 5

void polygeist_cutensornet_tensor_product_3d_f64(
    int32_t kq, int32_t kp, const double *psi, const double *u, double *out);

int main(void) {
  double psi[KQ * KP], u[KP * KP * KP], out[KQ * KQ * KQ];
  for (int i = 0; i < KQ * KP; ++i) psi[i] = (double)(i - 7) / 13.0;
  for (int i = 0; i < KP * KP * KP; ++i)
    u[i] = (double)(i % 11 - 5) / 9.0;

  polygeist_cutensornet_tensor_product_3d_f64(KQ, KP, psi, u, out);

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
  printf("cutensornet f64 tensor product max_err=%.17g\n", maxError);
  return maxError <= 1.0e-12 ? 0 : 1;
}
