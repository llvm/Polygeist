#include "../../runtime/polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  enum { NX = 7, NY = 6, NZ = 5, N = NX * NY * NZ };
  float input[N], output[N];
  for (int i = 0; i < N; ++i) {
    input[i] = 1.0f;
    output[i] = -9.0f;
  }
  // Invoke the same shape twice so the second call exercises the persistent
  // cuDNN descriptor/algorithm/workspace/device-buffer cache.
  for (int iteration = 0; iteration < 2; ++iteration) {
    for (int i = 0; i < N; ++i)
      output[i] = -9.0f;
    polygeist_cudnn_stencil3d_7pt_f32_flat(
        input, output, 2.0f, 1.0f, NY, NX, NX - 2, NY - 2, NZ - 2);
  }
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        int interior = x > 0 && x + 1 < NX && y > 0 && y + 1 < NY &&
                       z > 0 && z + 1 < NZ;
        float expected = interior ? 4.0f : -9.0f;
        if (fabsf(output[(z * NY + y) * NX + x] - expected) > 1.0e-5f) {
          fprintf(stderr, "FAIL at (%d,%d,%d): got %g expected %g\n",
                  x, y, z, output[(z * NY + y) * NX + x], expected);
          return 1;
        }
      }
  puts("PASS cudnnStencil3D7pt_f32_flat");
  return 0;
}
