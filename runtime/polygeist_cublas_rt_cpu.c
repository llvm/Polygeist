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

// ----------------------------------------------------------------------------
// Extracted-darknet batched CNN primitives (CPU reference impls). NCHW
// FP32 layout. Each is a straight-forward nested loop — slow, but useful
// for end-to-end correctness validation against the CUDA / cuDNN path.
// ----------------------------------------------------------------------------

void polygeist_cudnn_conv2d_batched(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F, float *Out) {
  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t oc = 0; oc < OC; ++oc)
      for (int32_t oh = 0; oh < OH; ++oh)
        for (int32_t ow = 0; ow < OW; ++ow) {
          float acc = 0.0f;
          for (int32_t ic = 0; ic < IC; ++ic)
            for (int32_t kh = 0; kh < K; ++kh)
              for (int32_t kw = 0; kw < K; ++kw) {
                size_t a_idx = ((size_t)b * IC + ic) * H * W +
                               (size_t)(oh + kh) * W + (ow + kw);
                size_t f_idx = ((size_t)oc * IC + ic) * K * K +
                               (size_t)kh * K + kw;
                acc += A[a_idx] * F[f_idx];
              }
          Out[((size_t)b * OC + oc) * OH * OW +
              (size_t)oh * OW + ow] = acc;
        }
}

void polygeist_cudnn_maxpool_batched(
    int32_t B, int32_t C, int32_t H, int32_t W, int32_t OH, int32_t OW,
    const float *A, float *Out) {
  // Derive K, S from H/OH for the typical pool=K=stride case.
  // OH = (H - K) / S + 1. For K == S: OH = H / S → S = H / OH, K = S.
  // For K != S (e.g. ResNet stem: K=3, S=2): can't recover both from
  // shape alone. We rely on the harness to pass shape consistent with
  // K = H - (OH - 1) * S = H - (OH - 1) * (H / OH) for the K==S case.
  // For K!=S, the harness should set S=H/OH and emit K via a side channel
  // — but for the extracted kernels in this PR both shapes use K==S
  // (MINI: K=S=2; LARGE: harness uses K=2, S=2 to match the simpler form).
  int32_t S = H / OH;
  int32_t K = (S > 0) ? S : 2;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t oh = 0; oh < OH; ++oh)
        for (int32_t ow = 0; ow < OW; ++ow) {
          float m = -3.40282347e38f;
          for (int32_t kh = 0; kh < K; ++kh)
            for (int32_t kw = 0; kw < K; ++kw) {
              size_t a_idx = ((size_t)b * C + c) * H * W +
                             (size_t)(oh * S + kh) * W + (ow * S + kw);
              float v = A[a_idx];
              if (v > m) m = v;
            }
          Out[((size_t)b * C + c) * OH * OW +
              (size_t)oh * OW + ow] = m;
        }
}

void polygeist_cudnn_batchnorm_inference(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out) {
  for (int32_t b = 0; b < B; ++b)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t h = 0; h < H; ++h)
        for (int32_t w = 0; w < W; ++w) {
          size_t idx = ((size_t)b * C + c) * H * W +
                       (size_t)h * W + w;
          Out[idx] = scale[c] * (A[idx] - mean[c]) * inv_std[c] + bias[c];
        }
}

void polygeist_cudnn_add_tensor_batched(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A, float *Out) {
  size_t n = (size_t)B * C * H * W;
  for (size_t i = 0; i < n; ++i) Out[i] += A[i];
}

void polygeist_cublas_memset_zero_2d_f32(int32_t M, int32_t N, float *A, int32_t lda) {
  if (lda == N) {
    memset(A, 0, (size_t)M * (size_t)N * sizeof(float));
  } else {
    for (int32_t i = 0; i < M; ++i)
      memset(&A[(size_t)i * (size_t)lda], 0, (size_t)N * sizeof(float));
  }
}

void polygeist_cublas_sgemm_1x1conv(
    int32_t B, int32_t IC, int32_t OC, int32_t HW,
    const float *A, const float *F, float *C) {
  /* C[b][oc][p] = sum_ic A[b][ic][p] * F[oc][ic] for p in 0..HW-1. */
  for (int32_t b = 0; b < B; ++b)
    for (int32_t oc = 0; oc < OC; ++oc)
      for (int32_t p = 0; p < HW; ++p) {
        float acc = 0.0f;
        for (int32_t ic = 0; ic < IC; ++ic) {
          size_t a_idx = ((size_t)b * IC + ic) * HW + p;
          size_t f_idx = (size_t)oc * IC + ic;
          acc += A[a_idx] * F[f_idx];
        }
        C[((size_t)b * OC + oc) * HW + p] = acc;
      }
}

void polygeist_cublas_dsyrk(int32_t N, int32_t K, const float *A, float *C) {
  /* C = AᵀA where A is K×N (row-major); C is N×N (row-major). */
  for (int32_t m = 0; m < N; ++m)
    for (int32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; ++k)
        acc += A[(size_t)k * N + m] * A[(size_t)k * N + n];
      C[(size_t)m * N + n] = acc;
    }
}

void polygeist_cublaslt_matmul_bias_relu(
    int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, const float *bias,
    float *C) {
  for (int32_t m = 0; m < M; ++m)
    for (int32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      float v = acc + bias[n];
      C[(size_t)m * N + n] = v > 0.0f ? v : 0.0f;
    }
}

void polygeist_cudnn_conv_bias_relu_add_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *bias, const float *Z,
    float *Out) {
  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t oc = 0; oc < OC; ++oc)
      for (int32_t oh = 0; oh < OH; ++oh)
        for (int32_t ow = 0; ow < OW; ++ow) {
          float acc = 0.0f;
          for (int32_t ic = 0; ic < IC; ++ic)
            for (int32_t kh = 0; kh < K; ++kh)
              for (int32_t kw = 0; kw < K; ++kw) {
                size_t a_idx = ((size_t)b * IC + ic) * H * W +
                               (size_t)(oh + kh) * W + (ow + kw);
                size_t f_idx = ((size_t)oc * IC + ic) * K * K +
                               (size_t)kh * K + kw;
                acc += A[a_idx] * F[f_idx];
              }
          size_t z_idx = ((size_t)b * OC + oc) * OH * OW +
                         (size_t)oh * OW + ow;
          float val = acc + bias[oc] + Z[z_idx];
          Out[z_idx] = val > 0.0f ? val : 0.0f;
        }
}

void polygeist_cudnn_conv_bn_relu_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out) {
  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t oc = 0; oc < OC; ++oc)
      for (int32_t oh = 0; oh < OH; ++oh)
        for (int32_t ow = 0; ow < OW; ++ow) {
          /* Conv accumulate. */
          float acc = 0.0f;
          for (int32_t ic = 0; ic < IC; ++ic)
            for (int32_t kh = 0; kh < K; ++kh)
              for (int32_t kw = 0; kw < K; ++kw) {
                size_t a_idx = ((size_t)b * IC + ic) * H * W +
                               (size_t)(oh + kh) * W + (ow + kw);
                size_t f_idx = ((size_t)oc * IC + ic) * K * K +
                               (size_t)kh * K + kw;
                acc += A[a_idx] * F[f_idx];
              }
          /* BN inference. */
          float bn = scale[oc] * (acc - mean[oc]) * inv_std[oc] + bias[oc];
          /* ReLU. */
          float relu = bn > 0.0f ? bn : 0.0f;
          Out[((size_t)b * OC + oc) * OH * OW +
              (size_t)oh * OW + ow] = relu;
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
