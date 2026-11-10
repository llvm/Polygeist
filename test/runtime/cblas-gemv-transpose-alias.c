// RUN: clang -O2 -DPOLYGEIST_CPU_USE_CBLAS -I%polygeist_src_root/runtime %s %polygeist_src_root/runtime/polygeist_cublas_rt_cpu.c -lopenblas -lm -o %t
// RUN: env OPENBLAS_NUM_THREADS=1 %t

#include "polygeist_cublas_rt.h"
#include <math.h>

int main(void) {
  const float a[6] = {1, 2, 3, 4, 5, 6};
  float same[3] = {1, 2, 3};
  float overlap[3] = {1, 2, 3};
  const float separate[3] = {1, 2, 3};
  float out[2] = {2, 3};
  polygeist_cublas_sgemv_T(3, 2, 1, a, 2, same, 0, same);
  if (fabsf(same[0] - 22) > 1e-6f || fabsf(same[1] - 28) > 1e-6f)
    return 1;
  polygeist_cublas_sgemv_T(3, 2, 0.5f, a, 2, overlap, 2, overlap + 1);
  if (fabsf(overlap[1] - 15) > 1e-6f || fabsf(overlap[2] - 20) > 1e-6f)
    return 2;
  polygeist_cublas_sgemv_T(3, 2, 0.5f, a, 2, separate, 2, out);
  return fabsf(out[0] - 15) > 1e-6f || fabsf(out[1] - 20) > 1e-6f;
}
