// polygeist_cublas_rt_cpu.c — CPU implementation of the runtime shim ABI.
// No CUDA dependency. By default this uses reference loops for correctness
// validation. Define POLYGEIST_CPU_USE_CBLAS to route BLAS-like kernels to an
// optimized CBLAS implementation such as OpenBLAS, BLIS, MKL, ArmPL, or NVPL.

#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef POLYGEIST_CPU_USE_CBLAS
#include <cblas.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

void polygeist_cublas_init(void) { /* no-op */ }
void polygeist_cublas_destroy(void) { /* no-op */ }
void polygeist_cublas_pipeline_begin(void) { /* no-op */ }
void polygeist_cublas_pipeline_end(void) { /* no-op */ }

void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
  return;
#endif
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

void polygeist_cublas_sgemm(
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float *A, int32_t lda,
    const float *B, int32_t ldb,
    float beta,
    float *C, int32_t ldc) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
  return;
#endif
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; ++k) {
        acc += A[(size_t)i * (size_t)lda + (size_t)k] *
               B[(size_t)k * (size_t)ldb + (size_t)j];
      }
      float *c = &C[(size_t)i * (size_t)ldc + (size_t)j];
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

void polygeist_cublas_memset_zero_1d_f32(int32_t N, float *v) {
  for (int32_t i = 0; i < N; ++i) v[i] = 0.0f;
}

void polygeist_cublas_dgemv(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, lda, x, 1, beta, y,
              1);
  return;
#endif
  // Row-major y[i] = alpha * sum_j A[i,j] * x[j] + beta * y[i]
  for (int32_t i = 0; i < M; ++i) {
    double acc = 0.0;
    for (int32_t j = 0; j < N; ++j)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[j];
    y[i] = alpha * acc + beta * y[i];
  }
}

void polygeist_cublas_sgemv(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, lda, x, 1, beta, y,
              1);
  return;
#endif
  for (int32_t i = 0; i < M; ++i) {
    float acc = 0.0f;
    for (int32_t j = 0; j < N; ++j)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[j];
    y[i] = alpha * acc + beta * y[i];
  }
}

void polygeist_cublas_daxpby(int32_t N, double alpha, const double *x,
                              double beta, double *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  if (x == y) {
    cblas_dscal(N, alpha + beta, y, 1);
  } else {
    cblas_dscal(N, beta, y, 1);
    cblas_daxpy(N, alpha, x, 1, y, 1);
  }
  return;
#endif
  for (int32_t i = 0; i < N; ++i) y[i] = alpha * x[i] + beta * y[i];
}

void polygeist_cublas_daxpy_unit(int32_t N, const double *x, double *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_daxpy(N, 1.0, x, 1, y, 1);
  return;
#endif
  for (int32_t i = 0; i < N; ++i) y[i] += x[i];
}

void polygeist_cublas_dger_rank2(int32_t M, int32_t N,
                                   const double *u1, const double *v1,
                                   const double *u2, const double *v2,
                                   double *A, int32_t lda) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_dger(CblasRowMajor, M, N, 1.0, u1, 1, v1, 1, A, lda);
  cblas_dger(CblasRowMajor, M, N, 1.0, u2, 1, v2, 1, A, lda);
  return;
#endif
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
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_dgemv(CblasRowMajor, CblasTrans, M, N, alpha, A, lda, x, 1, beta, y,
              1);
  return;
#endif
  // Row-major y[j] = alpha * sum_i A[i,j] * x[i] + beta * y[j]
  // (M is A's first dim = x's length; N is A's second dim = y's length)
  for (int32_t j = 0; j < N; ++j) {
    double acc = 0.0;
    for (int32_t i = 0; i < M; ++i)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[i];
    y[j] = alpha * acc + beta * y[j];
  }
}

void polygeist_cublas_sgemv_T(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemv(CblasRowMajor, CblasTrans, M, N, alpha, A, lda, x, 1, beta, y,
              1);
  return;
#endif
  for (int32_t j = 0; j < N; ++j) {
    float acc = 0.0f;
    for (int32_t i = 0; i < M; ++i)
      acc += A[(size_t)i * (size_t)lda + (size_t)j] * x[i];
    y[j] = alpha * acc + beta * y[j];
  }
}

void polygeist_cublas_dscal_2d(int32_t M, int32_t N, double scale,
                                 double *A, int32_t lda) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  if (lda == N) {
    cblas_dscal((int32_t)((size_t)M * (size_t)N), scale, A, 1);
  } else {
    for (int32_t i = 0; i < M; ++i)
      cblas_dscal(N, scale, &A[(size_t)i * (size_t)lda], 1);
  }
  return;
#endif
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

void polygeist_cudnn_conv2d_5x5_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2, double w3, double w4,
    double w5, double w6, double w7, double w8, double w9,
    double w10, double w11, double w12, double w13, double w14,
    double w15, double w16, double w17, double w18, double w19,
    double w20, double w21, double w22, double w23, double w24,
    const double *A, double *B) {
  const double w[25] = {
      w0, w1, w2, w3, w4, w5, w6, w7, w8, w9,
      w10, w11, w12, w13, w14, w15, w16, w17, w18, w19,
      w20, w21, w22, w23, w24};
  for (int32_t i = 0; i < M - 4; ++i) {
    for (int32_t j = 0; j < N - 4; ++j) {
      double acc = 0.0;
      for (int32_t dy = 0; dy < 5; ++dy)
        for (int32_t dx = 0; dx < 5; ++dx)
          acc += w[dy * 5 + dx] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

void polygeist_cudnn_conv2d_5x5_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2, float w3, float w4,
    float w5, float w6, float w7, float w8, float w9,
    float w10, float w11, float w12, float w13, float w14,
    float w15, float w16, float w17, float w18, float w19,
    float w20, float w21, float w22, float w23, float w24,
    const float *A, float *B) {
  const float w[25] = {
      w0, w1, w2, w3, w4, w5, w6, w7, w8, w9,
      w10, w11, w12, w13, w14, w15, w16, w17, w18, w19,
      w20, w21, w22, w23, w24};
  for (int32_t i = 0; i < M - 4; ++i) {
    for (int32_t j = 0; j < N - 4; ++j) {
      float acc = 0.0f;
      for (int32_t dy = 0; dy < 5; ++dy)
        for (int32_t dx = 0; dx < 5; ++dx)
          acc += w[dy * 5 + dx] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

void polygeist_cudnn_conv2d_ntap_f64(
    int32_t M, int32_t N, int32_t K,
    const double *W, const double *A, double *B) {
  int32_t out_h = M - (K - 1);
  int32_t out_w = N - (K - 1);
  for (int32_t i = 0; i < out_h; ++i) {
    for (int32_t j = 0; j < out_w; ++j) {
      double acc = 0.0;
      for (int32_t dy = 0; dy < K; ++dy)
        for (int32_t dx = 0; dx < K; ++dx)
          acc += W[(size_t)dy * (size_t)K + (size_t)dx] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

void polygeist_cudnn_conv2d_ntap_f32(
    int32_t M, int32_t N, int32_t K,
    const float *W, const float *A, float *B) {
  int32_t out_h = M - (K - 1);
  int32_t out_w = N - (K - 1);
  for (int32_t i = 0; i < out_h; ++i) {
    for (int32_t j = 0; j < out_w; ++j) {
      float acc = 0.0f;
      for (int32_t dy = 0; dy < K; ++dy)
        for (int32_t dx = 0; dx < K; ++dx)
          acc += W[(size_t)dy * (size_t)K + (size_t)dx] *
                 A[(size_t)(i + dy) * (size_t)N + (size_t)(j + dx)];
      B[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}

void polygeist_cudnn_conv3d_ntap_f64(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const double *W, const double *A, double *B) {
  for (int32_t z = 0; z < outD; ++z) {
    for (int32_t y = 0; y < outH; ++y) {
      for (int32_t x = 0; x < outW; ++x) {
        double acc = 0.0;
        for (int32_t dz = 0; dz < K; ++dz)
          for (int32_t dy = 0; dy < K; ++dy)
            for (int32_t dx = 0; dx < K; ++dx)
              acc += W[((size_t)dz * (size_t)K + (size_t)dy) *
                           (size_t)K + (size_t)dx] *
                     A[((size_t)(z + dz) * (size_t)inH +
                        (size_t)(y + dy)) * (size_t)inW +
                       (size_t)(x + dx)];
        B[((size_t)z * (size_t)outH + (size_t)y) * (size_t)outW +
          (size_t)x] = acc;
      }
    }
  }
}

void polygeist_cudnn_conv3d_ntap_f32(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const float *W, const float *A, float *B) {
  for (int32_t z = 0; z < outD; ++z) {
    for (int32_t y = 0; y < outH; ++y) {
      for (int32_t x = 0; x < outW; ++x) {
        float acc = 0.0f;
        for (int32_t dz = 0; dz < K; ++dz)
          for (int32_t dy = 0; dy < K; ++dy)
            for (int32_t dx = 0; dx < K; ++dx)
              acc += W[((size_t)dz * (size_t)K + (size_t)dy) *
                           (size_t)K + (size_t)dx] *
                     A[((size_t)(z + dz) * (size_t)inH +
                        (size_t)(y + dy)) * (size_t)inW +
                       (size_t)(x + dx)];
        B[((size_t)z * (size_t)outH + (size_t)y) * (size_t)outW +
          (size_t)x] = acc;
      }
    }
  }
}

void polygeist_custom_stencil3d_7pt_flat_f64(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6) {
  for (int32_t i = 0; i < N; ++i) {
    double extra_v = extra ? extra[i] : 0.0;
    double scale = coeff ? coeff[i] : 1.0;
    double base = base0 * a0[i] + (extra ? base_extra * extra_v : 0.0);
    double inner = c0 * a0[i] + c1 * a1[i] + c2 * a2[i] +
                   c3 * a3[i] + c4 * a4[i] + c5 * a5[i] +
                   c6 * a6[i] + (extra ? coeff_extra * extra_v : 0.0);
    out[i] = base + scale * inner;
  }
}

void polygeist_custom_stencil3d_7pt_flat_f32(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6) {
  for (int32_t i = 0; i < N; ++i) {
    float extra_v = extra ? extra[i] : 0.0f;
    float scale = coeff ? coeff[i] : 1.0f;
    float base = base0 * a0[i] + (extra ? base_extra * extra_v : 0.0f);
    float inner = c0 * a0[i] + c1 * a1[i] + c2 * a2[i] +
                  c3 * a3[i] + c4 * a4[i] + c5 * a5[i] +
                  c6 * a6[i] + (extra ? coeff_extra * extra_v : 0.0f);
    out[i] = base + scale * inner;
  }
}

void polygeist_cufft_z2z_1d(
    int32_t N, int32_t inverse, const double *A, double *B) {
  if (N <= 0) return;
  const double sign = inverse ? 1.0 : -1.0;
  for (int32_t k = 0; k < N; ++k) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int32_t n = 0; n < N; ++n) {
      double angle = sign * 2.0 * M_PI * (double)k * (double)n / (double)N;
      double c = cos(angle);
      double s = sin(angle);
      double ar = A[(size_t)2 * (size_t)n + 0];
      double ai = A[(size_t)2 * (size_t)n + 1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    B[(size_t)2 * (size_t)k + 0] = sum_re;
    B[(size_t)2 * (size_t)k + 1] = sum_im;
  }
}

void polygeist_cufft_c2c_1d(
    int32_t N, int32_t inverse, const float *A, float *B) {
  if (N <= 0) return;
  const float sign = inverse ? 1.0f : -1.0f;
  for (int32_t k = 0; k < N; ++k) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      float angle = sign * 2.0f * (float)M_PI * (float)k * (float)n / (float)N;
      float c = cosf(angle);
      float s = sinf(angle);
      float ar = A[(size_t)2 * (size_t)n + 0];
      float ai = A[(size_t)2 * (size_t)n + 1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    B[(size_t)2 * (size_t)k + 0] = sum_re;
    B[(size_t)2 * (size_t)k + 1] = sum_im;
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

// PVA-routed INT8/INT16 conv CPU stubs. These mirror the PVA Solutions
// Conv2d operator's hardware semantics, which differ from a "raw" integer
// multiply-add and from the centered conv emitted by the polybench source.
// Verified empirically against a Jetson PVA run; the model is:
//   1. PVA Conv2d operates on the full M×N input → full M×N output, with
//      CENTERED kernel anchor. Output(y, x) = Σ kernel(ky, kx) *
//      input(y + ky - K/2, x + kx - K/2).
//   2. Border policy: REPLICATE — out-of-range input coords clamp to
//      [0, M) × [0, N).
//   3. Kernel coefficients reinterpreted as UNSIGNED 8/16-bit even though
//      our weights arrive signed. A polybench -8 weight becomes 248, -9
//      becomes 247, -3 becomes 253. (PVA uses Q-format kernels with all
//      coefficients ≥ 0; the hardware ignores the sign bit.)
//   4. Accumulator: int64.
//   5. Q-format rescale: dst = (acc + (1 << (qbits-1))) >> qbits, with
//      qbits = 8 for int8 and 16 for int16.
//   6. Saturate to the signed range of the image dtype.
// Per-arg contract from the matcher's lowering: B points to &B[1][1] of
// the original output array (not &B[0][0]), and stride = N. The shim
// therefore writes only the (M-2)×(N-2) interior — output(i, j) for i,j
// in [0, M-2) × [0, N-2). The matched harness's dump reads the same
// interior region in B's coordinates ([1, M-1) × [1, N-1)), so the two
// agree element-for-element.
static inline int32_t pva_clamp(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void polygeist_pva_conv2d_3x3_i8(
    int32_t M, int32_t N,
    int8_t w0, int8_t w1, int8_t w2,
    int8_t w3, int8_t w4, int8_t w5,
    int8_t w6, int8_t w7, int8_t w8,
    const int8_t *A, int8_t *B) {
  const uint8_t w[9] = {
      (uint8_t)w0, (uint8_t)w1, (uint8_t)w2,
      (uint8_t)w3, (uint8_t)w4, (uint8_t)w5,
      (uint8_t)w6, (uint8_t)w7, (uint8_t)w8 };
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int64_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += (int64_t)w[ky * 3 + kx] *
                 (int64_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int64_t dst = (acc + 128) >> 8;
      if (dst >  127) dst =  127;
      if (dst < -128) dst = -128;
      B[(size_t)i * (size_t)N + (size_t)j] = (int8_t)dst;
    }
  }
}

// PVA BoxFilter — uniform 1/K² filter (no coefficient tensor). PVA hardware
// applies the same centered anchor + REPLICATE border policy as conv2d. Per
// the BoxFilter doc, the output is the integer mean of the K² neighbours,
// computed as `(sum + K²/2) >> log2(K²)` for K∈{3,5,7}... except 9 isn't a
// power of two, so the actual round-to-nearest is `(sum + 4) / 9` for K=3.
// Empirically verified against silicon below.
static void box_filter_3x3_kernel_i8(int32_t M, int32_t N,
                                      const int8_t *A, int8_t *B) {
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int32_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += (int32_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int32_t dst = (acc + 4) / 9;  // rounded mean
      if (dst >  127) dst =  127;
      if (dst < -128) dst = -128;
      B[(size_t)i * (size_t)N + (size_t)j] = (int8_t)dst;
    }
  }
}

void polygeist_pva_boxfilter_3x3_i8(int32_t M, int32_t N,
                                     const int8_t *A, int8_t *B) {
  box_filter_3x3_kernel_i8(M, N, A, B);
}

// GaussianFilter — sigma=1.0, K=3 hardcoded. Canonical discrete Gaussian
// kernel for sigma=1, K=3 is approximately
//   [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16
// PVA's hardware computes the kernel internally and likely matches this
// (we'll verify empirically and tweak if a few LSBs diverge — first-pass
// model captures the math). REPLICATE border, integer truncation on the
// /16 divide, saturate to dtype range.
static void gaussian_3x3_kernel_i8(int32_t M, int32_t N,
                                    const int8_t *A, int8_t *B) {
  static const int32_t w[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int32_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += w[ky * 3 + kx] *
                 (int32_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int32_t dst = (acc + 8) >> 4;  // /16 with rounding
      if (dst >  127) dst =  127;
      if (dst < -128) dst = -128;
      B[(size_t)i * (size_t)N + (size_t)j] = (int8_t)dst;
    }
  }
}

void polygeist_pva_gaussian_3x3_i8(int32_t M, int32_t N,
                                    const int8_t *A, int8_t *B) {
  gaussian_3x3_kernel_i8(M, N, A, B);
}

void polygeist_pva_gaussian_3x3_i16(int32_t M, int32_t N,
                                     const int16_t *A, int16_t *B) {
  static const int32_t w[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int32_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += w[ky * 3 + kx] *
                 (int32_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int32_t dst = (acc + 8) >> 4;
      if (dst >  32767) dst =  32767;
      if (dst < -32768) dst = -32768;
      B[(size_t)i * (size_t)N + (size_t)j] = (int16_t)dst;
    }
  }
}

void polygeist_pva_boxfilter_3x3_i16(int32_t M, int32_t N,
                                      const int16_t *A, int16_t *B) {
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int32_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += (int32_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int32_t dst = (acc + 4) / 9;
      if (dst >  32767) dst =  32767;
      if (dst < -32768) dst = -32768;
      B[(size_t)i * (size_t)N + (size_t)j] = (int16_t)dst;
    }
  }
}

// BilateralFilter — non-linear edge-preserving filter. Faithful CPU
// modeling requires implementing PVA's exact fixed-point spatial+range
// weight tables, which is impractical without spec docs. The CPU stub
// here is a "no-op pass-through" that lets us validate the PVA shim
// runs cleanly + the output isn't garbage (mean stays in input range,
// non-NaN, etc.). Real correctness comes from spot-checking the PVA
// output visually or against a reference float64 bilateral implementation.
void polygeist_pva_bilateral_3x3_i8(int32_t M, int32_t N,
                                     const int8_t *A, int8_t *B) {
  for (int32_t i = 0; i < M - 2; ++i)
    for (int32_t j = 0; j < N - 2; ++j)
      B[(size_t)i * (size_t)N + (size_t)j] = A[(size_t)(i + 1) * (size_t)N + (size_t)(j + 1)];
}

void polygeist_pva_bilateral_3x3_i16(int32_t M, int32_t N,
                                      const int16_t *A, int16_t *B) {
  for (int32_t i = 0; i < M - 2; ++i)
    for (int32_t j = 0; j < N - 2; ++j)
      B[(size_t)i * (size_t)N + (size_t)j] = A[(size_t)(i + 1) * (size_t)N + (size_t)(j + 1)];
}

// HistogramEqualization CPU stub — runs the textbook histogram-equalization
// algorithm on the FULL M×N image as uint8 (matching PVA's reinterpret),
// then writes the (M-2)×(N-2) interior to B starting at &B[1][1] to match
// the matcher's pointer-shift convention.
void polygeist_pva_histeq_i8(int32_t M, int32_t N,
                              const int8_t *A, int8_t *B) {
  size_t total = (size_t)M * (size_t)N;
  int32_t hist[256] = {0};
  for (size_t k = 0; k < total; ++k) hist[(uint8_t)A[k]]++;
  int32_t cdf[256];
  cdf[0] = hist[0];
  for (int b = 1; b < 256; ++b) cdf[b] = cdf[b - 1] + hist[b];
  int32_t cdf_min = 0;
  for (int b = 0; b < 256; ++b) if (cdf[b]) { cdf_min = cdf[b]; break; }
  int32_t denom = (int32_t)total - cdf_min;
  if (denom <= 0) denom = 1;
  uint8_t lut[256];
  for (int b = 0; b < 256; ++b) {
    int32_t v = (cdf[b] - cdf_min) * 255 / denom;
    if (v < 0) v = 0; if (v > 255) v = 255;
    lut[b] = (uint8_t)v;
  }
  // PVA writes lut[A[r][c]] at output position (r, c). The matcher passes
  // B = &B_orig[1][1], so dump-position (i_dump, j_dump) for i,j in [1, N-1)
  // reads PVA output at (i_dump-1, j_dump-1) — that's A[i_dump-1][j_dump-1]
  // through the LUT. Shim-local iteration i,j in [0, M-2) maps directly.
  for (int32_t i = 0; i < M - 2; ++i)
    for (int32_t j = 0; j < N - 2; ++j) {
      uint8_t in = (uint8_t)A[(size_t)i * (size_t)N + (size_t)j];
      B[(size_t)i * (size_t)N + (size_t)j] = (int8_t)lut[in];
    }
}

void polygeist_pva_conv2d_3x3_i16(
    int32_t M, int32_t N,
    int16_t w0, int16_t w1, int16_t w2,
    int16_t w3, int16_t w4, int16_t w5,
    int16_t w6, int16_t w7, int16_t w8,
    const int16_t *A, int16_t *B) {
  const uint16_t w[9] = {
      (uint16_t)w0, (uint16_t)w1, (uint16_t)w2,
      (uint16_t)w3, (uint16_t)w4, (uint16_t)w5,
      (uint16_t)w6, (uint16_t)w7, (uint16_t)w8 };
  for (int32_t i = 0; i < M - 2; ++i) {
    for (int32_t j = 0; j < N - 2; ++j) {
      int64_t acc = 0;
      for (int32_t ky = 0; ky < 3; ++ky) {
        int32_t iy = pva_clamp(i + ky - 1, 0, M - 1);
        for (int32_t kx = 0; kx < 3; ++kx) {
          int32_t ix = pva_clamp(j + kx - 1, 0, N - 1);
          acc += (int64_t)w[ky * 3 + kx] *
                 (int64_t)A[(size_t)iy * (size_t)N + (size_t)ix];
        }
      }
      int64_t dst = (acc + (1LL << 15)) >> 16;
      if (dst >  32767) dst =  32767;
      if (dst < -32768) dst = -32768;
      B[(size_t)i * (size_t)N + (size_t)j] = (int16_t)dst;
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

void polygeist_cudnn_conv2d_im2col_gemm_f32(
    int32_t IC, int32_t H, int32_t W, int32_t OC,
    int32_t K, int32_t S, int32_t P,
    const float *A, const float *F, float *Out) {
  const int32_t OH = (H + 2 * P - K) / S + 1;
  const int32_t OW = (W + 2 * P - K) / S + 1;
  for (int32_t oc = 0; oc < OC; ++oc)
    for (int32_t oh = 0; oh < OH; ++oh)
      for (int32_t ow = 0; ow < OW; ++ow) {
        float acc = 0.0f;
        for (int32_t ic = 0; ic < IC; ++ic)
          for (int32_t kh = 0; kh < K; ++kh)
            for (int32_t kw = 0; kw < K; ++kw) {
              int32_t ih = oh * S + kh - P;
              int32_t iw = ow * S + kw - P;
              if (ih < 0 || iw < 0 || ih >= H || iw >= W)
                continue;
              size_t a_idx = ((size_t)ic * H + ih) * W + iw;
              size_t f_idx = ((size_t)oc * IC + ic) * K * K +
                             (size_t)kh * K + kw;
              acc += A[a_idx] * F[f_idx];
            }
        Out[((size_t)oc * OH + oh) * OW + ow] = acc;
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
#ifdef POLYGEIST_CPU_USE_CBLAS
  for (int32_t b = 0; b < B; ++b) {
    const float *Ab = &A[(size_t)b * (size_t)IC * (size_t)HW];
    float *Cb = &C[(size_t)b * (size_t)OC * (size_t)HW];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, OC, HW, IC, 1.0f,
                F, IC, Ab, HW, 0.0f, Cb, HW);
  }
  return;
#endif
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
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, K, 1.0f, A, N,
              A, N, 0.0f, C, N);
  return;
#endif
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
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K,
              B, N, 0.0f, C, N);
  for (int32_t m = 0; m < M; ++m)
    for (int32_t n = 0; n < N; ++n) {
      float v = C[(size_t)m * (size_t)N + (size_t)n] + bias[n];
      C[(size_t)m * (size_t)N + (size_t)n] = v > 0.0f ? v : 0.0f;
    }
  return;
#endif
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

void polygeist_rmsnorm_f32(
    int32_t N, const float *X, const float *Weight, float *Out) {
  float ss = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    ss += X[i] * X[i];
  float scale = 1.0f / sqrtf(ss / (float)N + 1.0e-5f);
  for (int32_t i = 0; i < N; ++i)
    Out[i] = Weight[i] * (scale * X[i]);
}

void polygeist_rmsnorm_unweighted_f32(
    int32_t N, const float *X, float *Out) {
  float ss = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    ss += X[i] * X[i];
  float scale = 1.0f / sqrtf(ss / (float)N + 1.0e-5f);
  for (int32_t i = 0; i < N; ++i)
    Out[i] = scale * X[i];
}

void polygeist_cublas_dot_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  float acc = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    acc += X[i] * Y[i];
  *Out = acc;
}

void polygeist_cuda_gelu_tanh_f32(
    int32_t N, const float *X, float *Out) {
  for (int32_t i = 0; i < N; ++i) {
    float v = X[i];
    float inner = 0.7978845608028654f *
                  (v + 0.044715f * v * v * v);
    Out[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

void polygeist_whisper_exp_shift_sum_f32(
    int32_t N, const float *X, float max_val, float *Out, float *Sum) {
  float sum = 0.0f;
  for (int32_t i = 0; i < N; ++i) {
    float v = expf(X[i] - max_val);
    Out[i] = v;
    sum += v;
  }
  *Sum = sum;
}

void polygeist_cudnn_softmax_forward_f32(int32_t N, float *X) {
  if (N <= 0) return;
  float max_val = X[0];
  for (int32_t i = 1; i < N; ++i)
    if (X[i] > max_val) max_val = X[i];
  float sum = 0.0f;
  for (int32_t i = 0; i < N; ++i) {
    X[i] = expf(X[i] - max_val);
    sum += X[i];
  }
  for (int32_t i = 0; i < N; ++i)
    X[i] /= sum;
}

void polygeist_cudnn_softmax_forward_out_f32(
    int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  memcpy(Out, X, (size_t)N * sizeof(float));
  polygeist_cudnn_softmax_forward_f32(N, Out);
}

void polygeist_cuda_copy_f32(int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  memcpy(Out, X, (size_t)N * sizeof(float));
}

void polygeist_cuda_add_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  for (int32_t i = 0; i < N; ++i)
    Out[i] = X[i] + Y[i];
}

void polygeist_cuda_mask_select_f32(
    int32_t N, int32_t pos, const float *Scores, float *Out) {
  const float neg_inf = -3.4028234663852886e38f;
  for (int32_t i = 0; i < N; ++i)
    Out[i] = (i > pos) ? neg_inf : Scores[i];
}

void polygeist_cuda_swiglu_f32(
    int32_t N, const float *Gate, const float *Up, float *Out) {
  for (int32_t i = 0; i < N; ++i) {
    float g = Gate[i];
    Out[i] = (g / (1.0f + expf(-g))) * Up[i];
  }
}

void polygeist_cuda_rope_mulmul_f32(
    int32_t M, int32_t N, const float *A, const float *B,
    const float *C, const float *D, float *Out, int32_t add) {
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      size_t idx = (size_t)i * (size_t)N + (size_t)j;
      float p0 = A[idx] * B[j];
      float p1 = C[idx] * D[j];
      Out[idx] = add ? (p0 + p1) : (p0 - p1);
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
