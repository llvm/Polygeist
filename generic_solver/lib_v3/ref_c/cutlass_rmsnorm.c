/* CUTLASS test/unit/util/rms_norm.cu rmsnorm_host, specializing the tensor
 * ElementType alias to float and M=1. TensorRef.at becomes contiguous C array
 * access. cutlass::fast_sqrt(float)'s host branch is sqrtf; epsilon=1e-5f.
 * This is an equivalent reference for cuDNN's proprietary RMSNorm operation.
 * Its inp/sqrt_var*g association is retained; equivalence to reciprocal-mul
 * application forms therefore requires the explicitly relaxed numeric model.
 * Upstream source and fast_math implementation hashes are pinned in manifest.
 */
#include <math.h>
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float input[], float weight[], float output[]) {
  float square_sum = 0;
  for (int i = 0; i < n; ++i) {float inp = input[i]; square_sum += inp * inp;}
  float sq_mean = square_sum / (float)n;
  float sqrt_var = sqrtf(sq_mean + 1.0e-5f);
  for (int i = 0; i < n; ++i) {float inp = input[i], g = weight[i]; output[i] = inp / sqrt_var * g;}
}
