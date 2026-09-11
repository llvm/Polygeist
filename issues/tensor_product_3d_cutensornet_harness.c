#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define KP 4
#define KQ 5

// Bare-pointer memref ABI for three memref<?xf32> arguments.
extern void tensor_product_3d(
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t);

int main(void) {
  float psi[KQ * KP], u[KP * KP * KP], out[KQ * KQ * KQ];
  for (int i = 0; i < KQ * KP; ++i)
    psi[i] = (float)(i - 7) / 13.0f;
  for (int i = 0; i < KP * KP * KP; ++i)
    u[i] = (float)(i % 11 - 5) / 9.0f;

  tensor_product_3d(
      psi, psi, 0, KQ * KP, 1,
      u, u, 0, KP * KP * KP, 1,
      out, out, 0, KQ * KQ * KQ, 1);

  float maxErr = 0.0f;
  for (int a = 0; a < KQ; ++a)
    for (int b = 0; b < KQ; ++b)
      for (int c = 0; c < KQ; ++c) {
        float ref = 0.0f;
        for (int i = 0; i < KP; ++i)
          for (int j = 0; j < KP; ++j)
            for (int k = 0; k < KP; ++k)
              ref += psi[a * KP + i] * psi[b * KP + j] *
                     psi[c * KP + k] * u[(i * KP + j) * KP + k];
        float err = fabsf(out[(a * KQ + b) * KQ + c] - ref);
        if (err > maxErr) maxErr = err;
      }
  printf("cutensornet tensor product max_err=%g\n", maxErr);
  return maxErr <= 2.0e-5f ? 0 : 1;
}
