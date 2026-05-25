// polygeist_cublas_rt_cpu.c — reference CPU implementation of the runtime
// shim ABI. No CUDA dependency. Used for end-to-end correctness validation
// on machines without a GPU.
//
// The math is intentionally the slowest possible 3-loop gemm: the goal is
// to validate the lowering pass and the runtime call shape, not to be fast.

#include "polygeist_cublas_rt.h"

#include <stdio.h>
#include <time.h>

void polygeist_cublas_init(void) { /* no-op */ }
void polygeist_cublas_destroy(void) { /* no-op */ }

void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc) {
  // C[i,j] = alpha * sum_k A[i,k] * B[k,j] + beta * C[i,j]
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int32_t k = 0; k < K; ++k) {
        acc += A[(size_t)i * (size_t)lda + (size_t)k] *
               B[(size_t)k * (size_t)ldb + (size_t)j];
      }
      double *c = &C[(size_t)i * (size_t)ldc + (size_t)j];
      *c = alpha * acc + beta * (*c);
    }
  }
}

void polygeist_cublas_memset_zero_2d(int32_t M, int32_t N,
                                       double *A, int32_t lda) {
  for (int32_t i = 0; i < M; ++i) {
    double *row = &A[(size_t)i * (size_t)lda];
    for (int32_t j = 0; j < N; ++j) row[j] = 0.0;
  }
}

void polygeist_cublas_memset_zero_1d(int32_t N, double *v) {
  for (int32_t i = 0; i < N; ++i) v[i] = 0.0;
}

void polygeist_cublas_dgemv(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y) {
  // Row-major y[i] = alpha * sum_j A[i,j] * x[j] + beta * y[i]
  for (int32_t i = 0; i < M; ++i) {
    double acc = 0.0;
    for (int32_t j = 0; j < N; ++j)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[j];
    y[i] = alpha * acc + beta * y[i];
  }
}

void polygeist_cublas_daxpby(int32_t N, double alpha, const double *x,
                              double beta, double *y) {
  for (int32_t i = 0; i < N; ++i) y[i] = alpha * x[i] + beta * y[i];
}

void polygeist_cublas_daxpy_unit(int32_t N, const double *x, double *y) {
  for (int32_t i = 0; i < N; ++i) y[i] += x[i];
}

void polygeist_cublas_dger_rank2(int32_t M, int32_t N,
                                   const double *u1, const double *v1,
                                   const double *u2, const double *v2,
                                   double *A, int32_t lda) {
  for (int32_t i = 0; i < M; ++i) {
    double *row = &A[(size_t)i * (size_t)lda];
    for (int32_t j = 0; j < N; ++j)
      row[j] += u1[i] * v1[j] + u2[i] * v2[j];
  }
}

void polygeist_cublas_dgemv_T(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y) {
  // Row-major y[j] = alpha * sum_i A[i,j] * x[i] + beta * y[j]
  // (M is A's first dim = x's length; N is A's second dim = y's length)
  for (int32_t j = 0; j < N; ++j) {
    double acc = 0.0;
    for (int32_t i = 0; i < M; ++i)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[i];
    y[j] = alpha * acc + beta * y[j];
  }
}

void polygeist_cublas_dscal_2d(int32_t M, int32_t N, double scale,
                                 double *A, int32_t lda) {
  for (int32_t i = 0; i < M; ++i) {
    double *row = &A[(size_t)i * (size_t)lda];
    for (int32_t j = 0; j < N; ++j) row[j] *= scale;
  }
}

// Reference CPU impl of the polybench 3x3 9-tap conv2d. Same weights as the
// upstream kernel_conv2d in third_party/polybenchGpu/OpenMP/stencils/.
void polygeist_cudnn_conv2d_polybench9tap(
    int32_t M, int32_t N, const double *A, double *B) {
  polygeist_cudnn_conv2d_3x3_f64(M, N,
     0.2,  0.5, -0.8,
    -0.3,  0.6, -0.9,
     0.4,  0.7,  0.1,
    A, B);
}

// Generic 3x3 conv2d — filter weights passed at runtime by the caller
// (the matcher surfaces them from the linalg.generic body, the lowering
// pass forwards them here). Works for polybench, Sobel, Gaussian, or any
// other 3x3 weighted conv.
void polygeist_cudnn_conv2d_3x3_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2,
    double w3, double w4, double w5,
    double w6, double w7, double w8,
    const double *A, double *B) {
  const double w[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      double acc = 0.0;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += w[(dy + 1) * 3 + (dx + 1)] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

void polygeist_cudnn_conv2d_3x3_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2,
    float w3, float w4, float w5,
    float w6, float w7, float w8,
    const float *A, float *B) {
  const float w[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      float acc = 0.0f;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += w[(dy + 1) * 3 + (dx + 1)] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

// FP16 / BF16: accumulate in float to avoid catastrophic precision loss in
// 9-tap stencils (half's 11-bit mantissa is not enough for sums of nine
// products). Inputs/outputs/weights stay in the half precision type so the
// ABI matches MLIR's f16 / bf16 lowering. Guarded the same way as the
// header declarations — see polygeist_cublas_rt.h.
#if defined(__FLT16_MAX__)
void polygeist_cudnn_conv2d_3x3_f16(
    int32_t M, int32_t N,
    _Float16 w0, _Float16 w1, _Float16 w2,
    _Float16 w3, _Float16 w4, _Float16 w5,
    _Float16 w6, _Float16 w7, _Float16 w8,
    const _Float16 *A, _Float16 *B) {
  const float w[9] = { (float)w0, (float)w1, (float)w2,
                       (float)w3, (float)w4, (float)w5,
                       (float)w6, (float)w7, (float)w8 };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      float acc = 0.0f;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += w[(dy + 1) * 3 + (dx + 1)] *
                 (float)A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = (_Float16)acc;
    }
  }
}
#endif  // __FLT16_MAX__

#if defined(__BFLT16_MAX__) || defined(__ARM_FEATURE_BF16) || \
    defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || defined(__BF16__)
// GCC's aarch64 `__bf16` doesn't permit direct casts to/from float, so we
// do the bf16↔float conversion via bit reinterpretation: bf16 is the top
// 16 bits of an IEEE-754 fp32 (truncate-to-zero rounding). This is the
// portable trick that NVIDIA uses internally too.
static inline float _bf16_to_float(__bf16 b) {
  uint16_t bits;
  __builtin_memcpy(&bits, &b, sizeof(bits));
  uint32_t f_bits = ((uint32_t)bits) << 16;
  float f;
  __builtin_memcpy(&f, &f_bits, sizeof(f));
  return f;
}
static inline __bf16 _float_to_bf16(float f) {
  uint32_t f_bits;
  __builtin_memcpy(&f_bits, &f, sizeof(f_bits));
  // Round-to-nearest-even bias before truncating low 16 bits.
  uint32_t rounded = f_bits + 0x7FFF + ((f_bits >> 16) & 1);
  uint16_t bits = (uint16_t)(rounded >> 16);
  __bf16 out;
  __builtin_memcpy(&out, &bits, sizeof(out));
  return out;
}

void polygeist_cudnn_conv2d_3x3_bf16(
    int32_t M, int32_t N,
    __bf16 w0, __bf16 w1, __bf16 w2,
    __bf16 w3, __bf16 w4, __bf16 w5,
    __bf16 w6, __bf16 w7, __bf16 w8,
    const __bf16 *A, __bf16 *B) {
  const float w[9] = {
      _bf16_to_float(w0), _bf16_to_float(w1), _bf16_to_float(w2),
      _bf16_to_float(w3), _bf16_to_float(w4), _bf16_to_float(w5),
      _bf16_to_float(w6), _bf16_to_float(w7), _bf16_to_float(w8) };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      float acc = 0.0f;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += w[(dy + 1) * 3 + (dx + 1)] *
                 _bf16_to_float(A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)]);
      B[(size_t)i * (size_t)N + (size_t)j] = _float_to_bf16(acc);
    }
  }
}
#endif  // bf16 support

// INT32 / INT16: simple integer accumulation. cuDNN INT32 has no tensor-core
// path, but is bit-exact integer correctness; INT16 here mirrors what the
// CUDA shim does (upcast to INT32 internally). Wraparound semantics follow
// 2's-complement; overflow is undefined per C but in practice ints wrap.
void polygeist_cudnn_conv2d_3x3_i32(
    int32_t M, int32_t N,
    int32_t w0, int32_t w1, int32_t w2,
    int32_t w3, int32_t w4, int32_t w5,
    int32_t w6, int32_t w7, int32_t w8,
    const int32_t *A, int32_t *B) {
  const int32_t w[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      int64_t acc = 0;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += (int64_t)w[(dy + 1) * 3 + (dx + 1)] *
                 (int64_t)A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = (int32_t)acc;
    }
  }
}

void polygeist_cudnn_conv2d_3x3_i16(
    int32_t M, int32_t N,
    int16_t w0, int16_t w1, int16_t w2,
    int16_t w3, int16_t w4, int16_t w5,
    int16_t w6, int16_t w7, int16_t w8,
    const int16_t *A, int16_t *B) {
  const int32_t w[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      int64_t acc = 0;
      for (int32_t dy = -1; dy <= 1; ++dy)
        for (int32_t dx = -1; dx <= 1; ++dx)
          acc += (int64_t)w[(dy + 1) * 3 + (dx + 1)] *
                 (int64_t)A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = (int16_t)acc;
    }
  }
}

// CPU stub timing — wall-clock via clock_gettime(CLOCK_MONOTONIC). Useful
// for sanity but not for GPU perf numbers.

static struct timespec g_t0;

void polygeist_cublas_time_begin(void) {
  clock_gettime(CLOCK_MONOTONIC, &g_t0);
}

double polygeist_cublas_time_end_ms(void) {
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double dt_ns = (double)(t1.tv_sec - g_t0.tv_sec) * 1.0e9 +
                 (double)(t1.tv_nsec - g_t0.tv_nsec);
  return dt_ns / 1.0e6;
}
