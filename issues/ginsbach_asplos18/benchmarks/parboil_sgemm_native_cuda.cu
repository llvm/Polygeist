#include <cuda_runtime.h>
#include <iostream>

#include "../../../third_party/gpu-parboil/benchmarks/sgemm/src/cuda/sgemm_kernel.cu"

extern "C" void parboil_basic_sgemm(
    char transa, char transb, int m, int n, int k, float alpha,
    const float *a, int lda, const float *b, int ldb, float beta, float *c,
    int ldc) {
  regtileSgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
