/* C-level validation for the application extraction's element-count scaling.
 * This deliberately bypasses Polygeist and the library matcher: every
 * scratch-sliced normalization is compared with its direct loop algorithm. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif

#include "stage_kernels.h"

#define MAX_DATA (2250 * MFEM_BENCH_NE)
#define MAX_FIELD (192 * MFEM_BENCH_NE)

static double data[MAX_DATA], x[MAX_FIELD], initial[MAX_FIELD];
static double direct_y[MAX_FIELD], sliced_y[MAX_FIELD];
static double B[20], G[20];

static void initialize(void) {
  for (int i = 0; i < 20; ++i) {
    B[i] = ((i * 17 + 5) % 29 - 14.0) / 53.0;
    G[i] = ((i * 11 + 7) % 31 - 15.0) / 47.0;
  }
  for (int i = 0; i < MAX_DATA; ++i)
    data[i] = ((i * 13 + 3) % 37 - 18.0) / 71.0;
  for (int i = 0; i < MAX_FIELD; ++i) {
    x[i] = ((i * 19 + 2) % 41 - 20.0) / 83.0;
    initial[i] = ((i * 7 + 1) % 23 - 11.0) / 97.0;
  }
}

static double compare(int extent) {
  double result = 0.0;
  for (int i = 0; i < extent; ++i)
    result = fmax(result, fabs(direct_y[i] - sliced_y[i]));
  return result;
}

static void reset(void) {
  memcpy(direct_y, initial, sizeof(initial));
  memcpy(sliced_y, initial, sizeof(initial));
}

int main(void) {
  initialize();

  reset();
  mfem_pa_vector_mass_apply_3d_direct(B, data, x, direct_y);
  mfem_pa_vector_mass_apply_3d_sliced(B, data, x, sliced_y);
  printf("stage=vector_mass ne=%d max_abs=%.17g\n", MFEM_BENCH_NE,
         compare(192 * MFEM_BENCH_NE));

  reset();
  mfem_pa_vector_diffusion_apply_3d_direct(B, G, data, x, direct_y);
  mfem_pa_vector_diffusion_apply_3d_sliced(B, G, data, x, sliced_y);
  printf("stage=vector_diffusion ne=%d max_abs=%.17g\n", MFEM_BENCH_NE,
         compare(192 * MFEM_BENCH_NE));

  reset();
  mfem_pa_vector_convection_nl_apply_3d_direct(B, G, data, x, direct_y);
  mfem_pa_vector_convection_nl_apply_3d_sliced(B, G, data, x, sliced_y);
  printf("stage=vector_convection ne=%d max_abs=%.17g\n", MFEM_BENCH_NE,
         compare(192 * MFEM_BENCH_NE));

  reset();
  mfem_pa_discrete_gradient_apply_3d_direct(B, G, data, x, direct_y);
  mfem_pa_discrete_gradient_apply_3d_sliced(B, G, data, x, sliced_y);
  printf("stage=discrete_gradient ne=%d max_abs=%.17g\n", MFEM_BENCH_NE,
         compare(192 * MFEM_BENCH_NE));
  return 0;
}
