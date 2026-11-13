/* ggml-cpu/vec.cpp ggml_vec_swiglu_f32 scalar tail, with the actual
 * ggml_silu_f32 from vec.h inlined. CPU SIMD selection is specialized to the
 * scalar tail; FP32 types, arithmetic order and expf are unchanged.
 * Upstream paths/hashes are pinned in the manifest.
 */
#include <math.h>
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float x[], float gate[], float output[]) {
  for (int i = 0; i < n; ++i) output[i] = (x[i] / (1.0f + expf(-x[i]))) * gate[i];
}
