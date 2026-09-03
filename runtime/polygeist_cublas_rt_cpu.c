// polygeist_cublas_rt_cpu.c — CPU implementation of the runtime shim ABI.
// No CUDA dependency. By default this uses reference loops for correctness
// validation. Define POLYGEIST_CPU_USE_CBLAS to route BLAS-like kernels to an
// optimized CBLAS implementation such as OpenBLAS, BLIS, MKL, ArmPL, or NVPL.

#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
int32_t polygeist_cuda_graph_begin(int64_t graph_id) {
  (void)graph_id;
  return 1;
}
void polygeist_cuda_graph_end(int64_t graph_id) { (void)graph_id; }
void *polygeist_cuda_graph_stream(void) { return NULL; }

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

void polygeist_cublas_sgemm_transpose(
    int32_t M, int32_t N, int32_t K,
    int32_t transA, int32_t transB,
    float alpha,
    const float *A, int32_t lda,
    const float *B, int32_t ldb,
    float beta,
    float *C, int32_t ldc) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sgemm(CblasRowMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return;
#endif
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; ++k) {
        float av = transA ? A[(size_t)k * lda + i]
                          : A[(size_t)i * lda + k];
        float bv = transB ? B[(size_t)j * ldb + k]
                          : B[(size_t)k * ldb + j];
        acc += av * bv;
      }
      float *c = &C[(size_t)i * ldc + j];
      *c = alpha * acc + beta * *c;
    }
  }
}

void polygeist_cublas_sgemm_strided_batched_broadcast_rhs(
    int32_t batch, int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, float *C) {
  for (int32_t b = 0; b < batch; ++b) {
    const float *Ab = A + (size_t)b * (size_t)M * (size_t)K;
    float *Cb = C + (size_t)b * (size_t)M * (size_t)N;
#ifdef POLYGEIST_CPU_USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, Ab, K, B, N, 0.0f, Cb, N);
#else
    for (int32_t i = 0; i < M; ++i) {
      for (int32_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int32_t k = 0; k < K; ++k)
          acc += Ab[(size_t)i * (size_t)K + (size_t)k] *
                 B[(size_t)k * (size_t)N + (size_t)j];
        Cb[(size_t)i * (size_t)N + (size_t)j] = acc;
      }
    }
#endif
  }
}

void polygeist_cublas_sgemm_strided_batched(
    int32_t batch, int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, float *C) {
  for (int32_t b = 0; b < batch; ++b)
    polygeist_cublas_sgemm(M, N, K, 1.0f,
        A + (size_t)b * M * K, K,
        B + (size_t)b * K * N, N, 0.0f,
        C + (size_t)b * M * N, N);
}

void polygeist_cublas_dgemm_outer_product(
    int32_t M, int32_t N,
    const double *u, const double *v, double *C) {
  for (int32_t i = 0; i < M; ++i)
    for (int32_t j = 0; j < N; ++j)
      C[(size_t)i * (size_t)N + (size_t)j] = u[i] * v[j];
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

void polygeist_cublas_saxpby(int32_t N, float alpha, const float *x,
                              float beta, float *y) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  if (x == y) {
    cblas_sscal(N, alpha + beta, y, 1);
  } else {
    cblas_sscal(N, beta, y, 1);
    cblas_saxpy(N, alpha, x, 1, y, 1);
  }
  return;
#endif
  for (int32_t i = 0; i < N; ++i) y[i] = alpha * x[i] + beta * y[i];
}

void polygeist_cublas_sscal(int32_t N, float scale, float *x) {
#ifdef POLYGEIST_CPU_USE_CBLAS
  cblas_sscal(N, scale, x, 1);
  return;
#endif
  for (int32_t i = 0; i < N; ++i) x[i] *= scale;
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

void polygeist_cudnn_conv2d_uniform_window_f32(
    int32_t N, int32_t C, int32_t H, int32_t W,
    int32_t OH, int32_t OW, float weight,
    int32_t KH, int32_t KW, int32_t SH, int32_t SW,
    int32_t DH, int32_t DW, int32_t PH, int32_t PW,
    const float *input, float *output) {
  for (int32_t n = 0; n < N; ++n)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t oh = 0; oh < OH; ++oh)
        for (int32_t ow = 0; ow < OW; ++ow) {
          float sum = 0.0f;
          for (int32_t kh = 0; kh < KH; ++kh)
            for (int32_t kw = 0; kw < KW; ++kw) {
              int32_t ih = oh * SH + kh * DH - PH;
              int32_t iw = ow * SW + kw * DW - PW;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += weight * input[
                    ((size_t)n * (size_t)C + (size_t)c) *
                        (size_t)H * (size_t)W +
                    (size_t)ih * (size_t)W + (size_t)iw];
            }
          output[((size_t)n * (size_t)C + (size_t)c) *
                     (size_t)OH * (size_t)OW +
                 (size_t)oh * (size_t)OW + (size_t)ow] = sum;
        }
}

void polygeist_cudnn_adaptive_pool_f32(
    int32_t operation, int32_t rank, int32_t N, int32_t C,
    int32_t I0, int32_t I1, int32_t I2,
    int32_t O0, int32_t O1, int32_t O2,
    const void *ptr0, void *ptr1, void *ptr2) {
  (void)rank;
  const float *source = (const float *)ptr0;
  const int32_t *indices_in = operation == 3 ? (const int32_t *)ptr1 : NULL;
  float *values_out = (float *)(operation == 3 ? ptr2 : ptr1);
  int32_t *indices_out = operation == 2 ? (int32_t *)ptr2 : NULL;
  size_t input_spatial = (size_t)I0 * I1 * I2;
  size_t output_spatial = (size_t)O0 * O1 * O2;
  int fixed_average = operation == 4 || operation == 5;
  if (operation == 1 || operation == 3 || operation == 5)
    memset(values_out, 0,
           (size_t)N * C * input_spatial * sizeof(float));

  for (int32_t nc = 0; nc < N * C; ++nc)
    for (int32_t o0 = 0; o0 < O0; ++o0) {
      int32_t k0 = fixed_average ? I0 / O0 : 0;
      int32_t k1 = fixed_average ? I1 / O1 : 0;
      int32_t k2 = fixed_average ? I2 / O2 : 0;
      int32_t s0 = fixed_average ? o0 * k0 : (o0 * I0) / O0;
      int32_t e0 = fixed_average ? s0 + k0 :
          ((o0 + 1) * I0 + O0 - 1) / O0;
      for (int32_t o1 = 0; o1 < O1; ++o1) {
        int32_t s1 = fixed_average ? o1 * k1 : (o1 * I1) / O1;
        int32_t e1 = fixed_average ? s1 + k1 :
            ((o1 + 1) * I1 + O1 - 1) / O1;
        for (int32_t o2 = 0; o2 < O2; ++o2) {
          int32_t s2 = fixed_average ? o2 * k2 : (o2 * I2) / O2;
          int32_t e2 = fixed_average ? s2 + k2 :
              ((o2 + 1) * I2 + O2 - 1) / O2;
          size_t out_index = (size_t)nc * output_spatial +
              ((size_t)o0 * O1 + o1) * O2 + o2;
          if (operation == 0 || operation == 4) {
            float sum = 0.0f;
            int32_t count = 0;
            for (int32_t i0 = s0; i0 < e0; ++i0)
              for (int32_t i1 = s1; i1 < e1; ++i1)
                for (int32_t i2 = s2; i2 < e2; ++i2) {
                  size_t in_spatial = ((size_t)i0 * I1 + i1) * I2 + i2;
                  sum += source[(size_t)nc * input_spatial + in_spatial];
                  ++count;
                }
            values_out[out_index] = sum / (float)count;
          } else if (operation == 1 || operation == 5) {
            float contribution = source[out_index] /
                (float)((e0 - s0) * (e1 - s1) * (e2 - s2));
            for (int32_t i0 = s0; i0 < e0; ++i0)
              for (int32_t i1 = s1; i1 < e1; ++i1)
                for (int32_t i2 = s2; i2 < e2; ++i2) {
                  size_t in_spatial = ((size_t)i0 * I1 + i1) * I2 + i2;
                  values_out[(size_t)nc * input_spatial + in_spatial] +=
                      contribution;
                }
          } else if (operation == 2) {
            int32_t best = (s0 * I1 + s1) * I2 + s2;
            float value = source[(size_t)nc * input_spatial + best];
            for (int32_t i0 = s0; i0 < e0; ++i0)
              for (int32_t i1 = s1; i1 < e1; ++i1)
                for (int32_t i2 = s2; i2 < e2; ++i2) {
                  int32_t candidate = (i0 * I1 + i1) * I2 + i2;
                  float next = source[(size_t)nc * input_spatial + candidate];
                  if (next > value) {
                    value = next;
                    best = candidate;
                  }
                }
            values_out[out_index] = value;
            indices_out[out_index] = best;
          } else if (operation == 3) {
            int32_t destination = indices_in[out_index];
            if (destination < 0 || (size_t)destination >= input_spatial) {
              fprintf(stderr, "adaptive max-pool index out of range: %d\n",
                      destination);
              abort();
            }
            values_out[(size_t)nc * input_spatial + destination] +=
                source[out_index];
          }
        }
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

void polygeist_cutensornet_tensor_product_3d_f32(
    int32_t KQ, int32_t KP, const float *psi, const float *u, float *out) {
  for (int32_t a = 0; a < KQ; ++a)
    for (int32_t b = 0; b < KQ; ++b)
      for (int32_t c = 0; c < KQ; ++c) {
        float sum = 0.0f;
        for (int32_t i = 0; i < KP; ++i)
          for (int32_t j = 0; j < KP; ++j)
            for (int32_t k = 0; k < KP; ++k)
              sum += psi[(size_t)a * KP + i] *
                     psi[(size_t)b * KP + j] *
                     psi[(size_t)c * KP + k] *
                     u[((size_t)i * KP + j) * KP + k];
        out[((size_t)a * KQ + b) * KQ + c] = sum;
      }
}

void polygeist_cutensornet_tensor_product_3d_f64(
    int32_t KQ, int32_t KP, const double *psi, const double *u, double *out) {
  for (int32_t a = 0; a < KQ; ++a)
    for (int32_t b = 0; b < KQ; ++b)
      for (int32_t c = 0; c < KQ; ++c) {
        double sum = 0.0;
        for (int32_t i = 0; i < KP; ++i)
          for (int32_t j = 0; j < KP; ++j)
            for (int32_t k = 0; k < KP; ++k)
              sum += psi[(size_t)a * KP + i] *
                     psi[(size_t)b * KP + j] *
                     psi[(size_t)c * KP + k] *
                     u[((size_t)i * KP + j) * KP + k];
        out[((size_t)a * KQ + b) * KQ + c] = sum;
      }
}

void polygeist_cutensornet_contraction2_f64(
    const double *A, const double *B, double *C, const int64_t *metadata) {
  enum { MAX_RANK = 64, TENSOR_FIELDS = 3 * MAX_RANK };
  int64_t ranks[3] = {metadata[0], metadata[1], metadata[2]};
  int64_t extents[3][MAX_RANK];
  int64_t strides[3][MAX_RANK];
  int64_t modes[3][MAX_RANK];
  int64_t modeExtents[MAX_RANK];
  int present[3][MAX_RANK] = {{0}};
  for (int mode = 0; mode < MAX_RANK; ++mode)
    modeExtents[mode] = 1;

  for (int tensor = 0; tensor < 3; ++tensor) {
    if (ranks[tensor] < 0 || ranks[tensor] > MAX_RANK) {
      fprintf(stderr, "polygeist runtime: invalid contraction rank\n");
      return;
    }
    int64_t base = 3 + (int64_t)tensor * TENSOR_FIELDS;
    for (int64_t dim = 0; dim < ranks[tensor]; ++dim) {
      extents[tensor][dim] = metadata[base + dim];
      strides[tensor][dim] = metadata[base + MAX_RANK + dim];
      modes[tensor][dim] = metadata[base + 2 * MAX_RANK + dim];
      int64_t mode = modes[tensor][dim];
      int64_t extent = extents[tensor][dim];
      if (mode < 0 || mode >= MAX_RANK || extent <= 0) {
        fprintf(stderr, "polygeist runtime: invalid contraction metadata\n");
        return;
      }
      if (modeExtents[mode] != 1 && modeExtents[mode] != extent) {
        fprintf(stderr,
                "polygeist runtime: inconsistent contraction mode extent\n");
        return;
      }
      modeExtents[mode] = extent;
      present[tensor][mode] = 1;
    }
  }

  int64_t total = 1;
  for (int mode = 0; mode < MAX_RANK; ++mode)
    total *= modeExtents[mode];
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t coordinates[MAX_RANK];
    int64_t remaining = linear;
    for (int mode = MAX_RANK - 1; mode >= 0; --mode) {
      coordinates[mode] = remaining % modeExtents[mode];
      remaining /= modeExtents[mode];
    }

    int64_t offsets[3] = {0, 0, 0};
    for (int tensor = 0; tensor < 3; ++tensor)
      for (int64_t dim = 0; dim < ranks[tensor]; ++dim)
        offsets[tensor] +=
            coordinates[modes[tensor][dim]] * strides[tensor][dim];

    int firstReductionPoint = 1;
    for (int mode = 0; mode < MAX_RANK; ++mode)
      if (!present[2][mode] &&
          (present[0][mode] || present[1][mode]) &&
          coordinates[mode] != 0)
        firstReductionPoint = 0;
    if (firstReductionPoint)
      C[offsets[2]] = 0.0;
    C[offsets[2]] += A[offsets[0]] * B[offsets[1]];
  }
}

void polygeist_cutensornet_contraction2_f64_device(
    const double *A, const double *B, double *C, const int64_t *metadata) {
  // The CPU runtime has no distinct device address space. Keep the symbol
  // available so lowering/link tests can exercise the ABI.
  polygeist_cutensornet_contraction2_f64(A, B, C, metadata);
}

enum {
  POLYGEIST_NETWORK_MAX_INPUTS = 32,
  POLYGEIST_NETWORK_MAX_MODES = 64
};

static void polygeist_cutensornet_network_cpu(
    const int64_t *pointer_values, const int64_t *metadata, int use_f64) {
  if (!pointer_values || !metadata || metadata[0] != 1) {
    fprintf(stderr, "polygeist runtime: invalid tensor-network ABI\n");
    return;
  }
  int64_t num_inputs = metadata[1];
  int accumulate = metadata[2] != 0;
  int64_t num_tensors = num_inputs + 1;
  if (num_inputs < 2 || num_inputs > POLYGEIST_NETWORK_MAX_INPUTS) {
    fprintf(stderr, "polygeist runtime: invalid tensor-network input count\n");
    return;
  }

  int64_t ranks[POLYGEIST_NETWORK_MAX_INPUTS + 1] = {0};
  int64_t extents[POLYGEIST_NETWORK_MAX_INPUTS + 1]
                 [POLYGEIST_NETWORK_MAX_MODES] = {{0}};
  int64_t strides[POLYGEIST_NETWORK_MAX_INPUTS + 1]
                 [POLYGEIST_NETWORK_MAX_MODES] = {{0}};
  int64_t modes[POLYGEIST_NETWORK_MAX_INPUTS + 1]
               [POLYGEIST_NETWORK_MAX_MODES] = {{0}};
  int present[POLYGEIST_NETWORK_MAX_INPUTS + 1]
             [POLYGEIST_NETWORK_MAX_MODES] = {{0}};
  int64_t mode_extents[POLYGEIST_NETWORK_MAX_MODES];
  int mode_seen[POLYGEIST_NETWORK_MAX_MODES] = {0};
  for (int mode = 0; mode < POLYGEIST_NETWORK_MAX_MODES; ++mode)
    mode_extents[mode] = 1;

  int64_t cursor = 3 + num_tensors;
  for (int64_t tensor = 0; tensor < num_tensors; ++tensor) {
    ranks[tensor] = metadata[3 + tensor];
    if (ranks[tensor] < 0 ||
        ranks[tensor] > POLYGEIST_NETWORK_MAX_MODES) {
      fprintf(stderr, "polygeist runtime: invalid tensor-network rank\n");
      return;
    }
    for (int64_t dim = 0; dim < ranks[tensor]; ++dim) {
      int64_t extent = metadata[cursor++];
      int64_t stride = metadata[cursor++];
      int64_t mode = metadata[cursor++];
      if (extent <= 0 || stride < 0 || mode < 0 ||
          mode >= POLYGEIST_NETWORK_MAX_MODES ||
          (mode_seen[mode] && mode_extents[mode] != extent)) {
        fprintf(stderr, "polygeist runtime: invalid tensor-network metadata\n");
        return;
      }
      extents[tensor][dim] = extent;
      strides[tensor][dim] = stride;
      modes[tensor][dim] = mode;
      present[tensor][mode] = 1;
      mode_extents[mode] = extent;
      mode_seen[mode] = 1;
    }
  }

  int64_t total = 1;
  for (int mode = 0; mode < POLYGEIST_NETWORK_MAX_MODES; ++mode) {
    if (mode_extents[mode] > INT64_MAX / total) {
      fprintf(stderr, "polygeist runtime: tensor-network extent overflow\n");
      return;
    }
    total *= mode_extents[mode];
  }
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t coordinates[POLYGEIST_NETWORK_MAX_MODES];
    int64_t remaining = linear;
    for (int mode = POLYGEIST_NETWORK_MAX_MODES - 1; mode >= 0; --mode) {
      coordinates[mode] = remaining % mode_extents[mode];
      remaining /= mode_extents[mode];
    }
    int64_t offsets[POLYGEIST_NETWORK_MAX_INPUTS + 1] = {0};
    for (int64_t tensor = 0; tensor < num_tensors; ++tensor)
      for (int64_t dim = 0; dim < ranks[tensor]; ++dim)
        offsets[tensor] +=
            coordinates[modes[tensor][dim]] * strides[tensor][dim];

    int first_reduction_point = 1;
    for (int mode = 0; mode < POLYGEIST_NETWORK_MAX_MODES; ++mode)
      if (!present[num_inputs][mode] && coordinates[mode] != 0)
        first_reduction_point = 0;

    if (use_f64) {
      double product = 1.0;
      for (int64_t tensor = 0; tensor < num_inputs; ++tensor)
        product *= ((const double *)(uintptr_t)pointer_values[tensor])
                       [offsets[tensor]];
      double *output =
          (double *)(uintptr_t)pointer_values[num_inputs];
      if (first_reduction_point) {
        if (accumulate)
          output[offsets[num_inputs]] += product;
        else
          output[offsets[num_inputs]] = product;
      } else {
        output[offsets[num_inputs]] += product;
      }
    } else {
      float product = 1.0f;
      for (int64_t tensor = 0; tensor < num_inputs; ++tensor)
        product *= ((const float *)(uintptr_t)pointer_values[tensor])
                       [offsets[tensor]];
      float *output = (float *)(uintptr_t)pointer_values[num_inputs];
      if (first_reduction_point) {
        if (accumulate)
          output[offsets[num_inputs]] += product;
        else
          output[offsets[num_inputs]] = product;
      } else {
        output[offsets[num_inputs]] += product;
      }
    }
  }
}

void polygeist_cutensornet_network_f32(
    const int64_t *pointers, const int64_t *metadata) {
  polygeist_cutensornet_network_cpu(pointers, metadata, 0);
}
void polygeist_cutensornet_network_f32_device(
    const int64_t *pointers, const int64_t *metadata) {
  polygeist_cutensornet_network_cpu(pointers, metadata, 0);
}
void polygeist_cutensornet_network_f64(
    const int64_t *pointers, const int64_t *metadata) {
  polygeist_cutensornet_network_cpu(pointers, metadata, 1);
}
void polygeist_cutensornet_network_f64_device(
    const int64_t *pointers, const int64_t *metadata) {
  polygeist_cutensornet_network_cpu(pointers, metadata, 1);
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

void polygeist_cudnn_conv1d_bias_f32(
    int32_t B, int32_t IC, int32_t OC, int32_t L, int32_t K,
    const float *input, const float *filter, const float *bias, float *output) {
  int32_t OL = L - K + 1;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t oc = 0; oc < OC; ++oc)
      for (int32_t ol = 0; ol < OL; ++ol) {
        float acc = bias[oc];
        for (int32_t ic = 0; ic < IC; ++ic)
          for (int32_t k = 0; k < K; ++k)
            acc += input[((size_t)b * IC + ic) * L + ol + k] *
                   filter[((size_t)oc * IC + ic) * K + k];
        output[((size_t)b * OC + oc) * OL + ol] = acc;
      }
}

void polygeist_cudnn_conv2d_dilated_f32(
    int32_t IC, int32_t OC, int32_t H, int32_t W, int32_t KH, int32_t KW,
    int32_t DH, int32_t DW, const float *input, const float *filter,
    float *output) {
  int32_t OH = H - (KH - 1) * DH;
  int32_t OW = W - (KW - 1) * DW;
  for (int32_t oc = 0; oc < OC; ++oc)
    for (int32_t oh = 0; oh < OH; ++oh)
      for (int32_t ow = 0; ow < OW; ++ow) {
        float acc = 0.0f;
        for (int32_t ic = 0; ic < IC; ++ic)
          for (int32_t kh = 0; kh < KH; ++kh)
            for (int32_t kw = 0; kw < KW; ++kw)
              acc += input[((size_t)ic * H + oh + kh * DH) * W +
                           ow + kw * DW] *
                     filter[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
        output[((size_t)oc * OH + oh) * OW + ow] = acc;
      }
}

void polygeist_cublas_gemmex_i8_i32(
    int32_t M, int32_t N, int32_t K, const int8_t *A, const int8_t *B,
    int32_t *C) {
  for (int32_t i = 0; i < M; ++i)
    for (int32_t j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int32_t k = 0; k < K; ++k)
        acc += (int32_t)A[(size_t)i * K + k] *
               (int32_t)B[(size_t)k * N + j];
      C[(size_t)i * N + j] = acc;
    }
}

void polygeist_cublas_snrm2_f32(
    int32_t N, const float *input, float *output) {
  double sum = 0.0;
  for (int32_t i = 0; i < N; ++i)
    sum += (double)input[i] * (double)input[i];
  output[0] = (float)sqrt(sum);
}

void polygeist_cublas_joint_maxabs_product_f32(
    int32_t N, const float *a, const float *b, float *output) {
  float ma = 0.0f, mb = 0.0f;
  for (int32_t i = 0; i < N; ++i) {
    float av = fabsf(a[i]), bv = fabsf(b[i]);
    if (av > ma) ma = av;
    if (bv > mb) mb = bv;
  }
  output[0] = ma * mb;
}

void polygeist_cudnn_feature_mask_scale_f32(
    int32_t N, int32_t C, int32_t H, int32_t W, float scale,
    const float *input, const float *mask, float *output) {
  for (int32_t n = 0; n < N; ++n)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t h = 0; h < H; ++h)
        for (int32_t w = 0; w < W; ++w) {
          size_t index = ((size_t)n * C + c) * H * W + (size_t)h * W + w;
          output[index] = input[index] * mask[(size_t)n * C + c] * scale;
        }
}

void polygeist_cudnn_conv_transpose2d_f32(
    int32_t B, int32_t IC, int32_t OC, int32_t H, int32_t W,
    int32_t KH, int32_t KW, const float *input, const float *filter,
    float *output) {
  int32_t OH = H + KH - 1, OW = W + KW - 1;
  memset(output, 0, (size_t)B * OC * OH * OW * sizeof(float));
  for (int32_t b = 0; b < B; ++b)
    for (int32_t ic = 0; ic < IC; ++ic)
      for (int32_t ih = 0; ih < H; ++ih)
        for (int32_t iw = 0; iw < W; ++iw)
          for (int32_t oc = 0; oc < OC; ++oc)
            for (int32_t kh = 0; kh < KH; ++kh)
              for (int32_t kw = 0; kw < KW; ++kw)
                output[((size_t)b * OC + oc) * OH * OW +
                       (size_t)(ih + kh) * OW + iw + kw] +=
                    input[((size_t)b * IC + ic) * H * W +
                          (size_t)ih * W + iw] *
                    filter[((size_t)ic * OC + oc) * KH * KW +
                           (size_t)kh * KW + kw];
}

void polygeist_cudnn_conv_transpose3d_f32(
    int32_t IC, int32_t OC, int32_t D, int32_t H, int32_t W,
    int32_t KD, int32_t KH, int32_t KW, const float *input,
    const float *filter, float *output) {
  int32_t OD=D+KD-1,OH=H+KH-1,OW=W+KW-1;
  memset(output,0,(size_t)OC*OD*OH*OW*sizeof(float));
  for(int32_t ic=0;ic<IC;++ic)for(int32_t z=0;z<D;++z)
    for(int32_t y=0;y<H;++y)for(int32_t x=0;x<W;++x)
      for(int32_t oc=0;oc<OC;++oc)for(int32_t kz=0;kz<KD;++kz)
        for(int32_t ky=0;ky<KH;++ky)for(int32_t kx=0;kx<KW;++kx)
          output[(((size_t)oc*OD+z+kz)*OH+y+ky)*OW+x+kx] +=
              input[((size_t)ic*D+z)*H*W+(size_t)y*W+x] *
              filter[(((size_t)ic*OC+oc)*KD+kz)*KH*KW+(size_t)ky*KW+kx];
}

void polygeist_cudnn_conv_backward_filter3d_f32(
    int32_t IC, int32_t OC,int32_t ID,int32_t IH,int32_t IW,
    int32_t OD,int32_t OH,int32_t OW,int32_t KD,int32_t KH,int32_t KW,
    const float *input,const float *grad_output,float *grad_filter) {
  for(int32_t oc=0;oc<OC;++oc)for(int32_t ic=0;ic<IC;++ic)
    for(int32_t kz=0;kz<KD;++kz)for(int32_t ky=0;ky<KH;++ky)
      for(int32_t kx=0;kx<KW;++kx){float acc=0;
        for(int32_t z=0;z<OD;++z)for(int32_t y=0;y<OH;++y)
          for(int32_t x=0;x<OW;++x)
            acc += input[((size_t)ic*ID+z+kz)*IH*IW+(size_t)(y+ky)*IW+x+kx] *
                   grad_output[((size_t)oc*OD+z)*OH*OW+(size_t)y*OW+x];
        grad_filter[(((size_t)oc*IC+ic)*KD+kz)*KH*KW+(size_t)ky*KW+kx]=acc;}
}

void polygeist_cudnn_depthwise_conv2d_f32(
    int32_t B, int32_t C, int32_t H, int32_t W, int32_t KH, int32_t KW,
    const float *input, const float *filter, const float *bias, float *output) {
  int32_t py = KH / 2, px = KW / 2;
  for (int32_t b = 0; b < B; ++b)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t y = 0; y < H; ++y)
        for (int32_t x = 0; x < W; ++x) {
          float acc = bias[c];
          for (int32_t ky = 0; ky < KH; ++ky)
            for (int32_t kx = 0; kx < KW; ++kx) {
              int32_t iy = y + ky - py, ix = x + kx - px;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                acc += input[((size_t)b * C + c) * H * W +
                             (size_t)iy * W + ix] *
                       filter[((size_t)c * KH + ky) * KW + kx];
            }
          output[((size_t)b * C + c) * H * W + (size_t)y * W + x] = acc;
        }
}

void polygeist_cutensor_kronecker_product2d_f32(
    int32_t A, int32_t B, int32_t C, int32_t D,
    const float *x, const float *y, float *output) {
  for (int32_t a = 0; a < A; ++a)
    for (int32_t c = 0; c < C; ++c)
      for (int32_t b = 0; b < B; ++b)
        for (int32_t d = 0; d < D; ++d)
          output[((size_t)a * C + c) * B * D + (size_t)b * D + d] =
              x[(size_t)a * B + b] * y[(size_t)c * D + d];
}

void polygeist_cudnn_binary_cross_entropy_mean_f32(
    int32_t N, const float *input, const float *target, float *output) {
  float sum = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    sum -= target[i] * logf(input[i]) +
           (1.0f - target[i]) * logf(1.0f - input[i]);
  output[0] = sum / (float)N;
}

void polygeist_cudnn_conv_tbc_f32(
    int32_t T, int32_t B, int32_t I, int32_t O, int32_t K,
    const float *input, const float *filter, float *output) {
  int32_t TO = T - K + 1;
  for (int32_t t = 0; t < TO; ++t)
    for (int32_t b = 0; b < B; ++b)
      for (int32_t o = 0; o < O; ++o) {
        float acc = 0.0f;
        for (int32_t k = 0; k < K; ++k)
          for (int32_t i = 0; i < I; ++i)
            acc += input[((size_t)(t + k) * B + b) * I + i] *
                   filter[((size_t)k * I + i) * O + o];
        output[((size_t)t * B + b) * O + o] = acc;
      }
}
void polygeist_cudnn_conv_tbc_backward_f32(
    int32_t T,int32_t B,int32_t I,int32_t O,int32_t K,
    const float *grad,const float *filter,float *output) {
  int32_t TO=T+K-1;memset(output,0,(size_t)TO*B*I*sizeof(float));
  for(int32_t t=0;t<T;++t)for(int32_t b=0;b<B;++b)
    for(int32_t o=0;o<O;++o)for(int32_t k=0;k<K;++k)
      for(int32_t i=0;i<I;++i)
        output[((size_t)(t+k)*B+b)*I+i]+=
          grad[((size_t)t*B+b)*O+o]*filter[((size_t)k*I+i)*O+o];
}

void polygeist_cudnn_transform_bias_rescale_qkv_f32(
    int32_t B, int32_t S, int32_t H, int32_t D, float scale,
    const float *qkv, const float *bias, float *q, float *k, float *v) {
  float *outputs[3] = {q, k, v};
  for (int32_t b = 0; b < B; ++b)
    for (int32_t s = 0; s < S; ++s)
      for (int32_t h = 0; h < H; ++h)
        for (int32_t d = 0; d < D; ++d)
          for (int32_t part = 0; part < 3; ++part) {
            float value = qkv[((((size_t)b * S + s) * 3 + part) * H + h) *
                              D + d] +
                          bias[((size_t)part * H + h) * D + d];
            if (part == 0) value *= scale;
            outputs[part][(((size_t)b * H + h) * S + s) * D + d] = value;
          }
}

void polygeist_cudnn_addr_elementwise_f32(
    int32_t N, float beta, float alpha, const float *self,
    const float *x, const float *y, float *output) {
  for (int32_t i = 0; i < N; ++i)
    output[i] = beta == 0.0f ? alpha * x[i] * y[i]
                             : beta * self[i] + alpha * x[i] * y[i];
}

void polygeist_cudnn_log_sigmoid_f32(
    int32_t N, const float *x, float *output, float *buffer) {
  for (int32_t i = 0; i < N; ++i) {
    buffer[i] = expf(-fabsf(x[i]));
    output[i] = fminf(x[i], 0.0f) - log1pf(buffer[i]);
  }
}

void polygeist_cudnn_conv3d_channels_f32(
    int32_t IC, int32_t inD, int32_t inH, int32_t inW,
    int32_t OC, int32_t kD, int32_t kH, int32_t kW,
    const float *input, const float *filter, const float *bias, float *output) {
  const int32_t outD = inD - kD + 1;
  const int32_t outH = inH - kH + 1;
  const int32_t outW = inW - kW + 1;
  for (int32_t oc = 0; oc < OC; ++oc)
    for (int32_t od = 0; od < outD; ++od)
      for (int32_t oh = 0; oh < outH; ++oh)
        for (int32_t ow = 0; ow < outW; ++ow) {
          float acc = bias ? bias[oc] : 0.0f;
          for (int32_t ic = 0; ic < IC; ++ic)
            for (int32_t kd = 0; kd < kD; ++kd)
              for (int32_t kh = 0; kh < kH; ++kh)
                for (int32_t kw = 0; kw < kW; ++kw) {
                  size_t inIdx = (((size_t)ic * inD + (od + kd)) * inH +
                                  (oh + kh)) * inW + (ow + kw);
                  size_t fIdx = ((((size_t)oc * IC + ic) * kD + kd) * kH +
                                 kh) * kW + kw;
                  acc += input[inIdx] * filter[fIdx];
                }
          output[((size_t)oc * outD + od) * outH * outW +
                 (size_t)oh * outW + ow] = acc;
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

void polygeist_cudnn_batchnorm_backward_f32(
    int32_t N, int32_t C, int32_t spatial, int32_t full_outputs,
    const float *grad, const float *x, const float *mean,
    const float *invstd, const float *weight, float *dx,
    float *dweight, float *dbias) {
  int32_t m = N * spatial;
  for (int32_t c = 0; c < C; ++c) {
    float sum_g = 0.0f, sum_gx = 0.0f;
    for (int32_t n = 0; n < N; ++n)
      for (int32_t s = 0; s < spatial; ++s) {
        size_t index = ((size_t)n * C + c) * spatial + s;
        sum_g += grad[index];
        sum_gx += grad[index] * (x[index] - mean[c]);
      }
    if (full_outputs) {
      dbias[c] = sum_g;
      dweight[c] = sum_gx * invstd[c];
    }
    float scale = full_outputs ? weight[c] : 1.0f;
    float factor = scale * invstd[c] / (float)m;
    for (int32_t n = 0; n < N; ++n)
      for (int32_t s = 0; s < spatial; ++s) {
        size_t index = ((size_t)n * C + c) * spatial + s;
        float centered = x[index] - mean[c];
        dx[index] = factor * ((float)m * grad[index] - sum_g -
            centered * invstd[c] * invstd[c] * sum_gx);
      }
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

void polygeist_cublas_dot_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  float acc = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    acc += X[i] * Y[i];
  *Out = acc;
}

void polygeist_cublas_dot_f64(
    int32_t N, const double *X, const double *Y, double *Out) {
  double acc = 0.0;
  for (int32_t i = 0; i < N; ++i)
    acc += X[i] * Y[i];
  *Out = acc;
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

void polygeist_cuda_copy_strided_2d_f32(
    int32_t rows, int32_t cols,
    int32_t src_row_stride, int32_t src_col_stride,
    int32_t dst_row_stride, int32_t dst_col_stride,
    const float *X, float *Out) {
  for (int32_t i = 0; i < rows; ++i)
    for (int32_t j = 0; j < cols; ++j)
      Out[(size_t)i * dst_row_stride + (size_t)j * dst_col_stride] =
          X[(size_t)i * src_row_stride + (size_t)j * src_col_stride];
}

void polygeist_cublas_broadcast_1d_to_2d_f32(
    int32_t axis, int32_t rows, int32_t cols,
    const float *X, float *Out) {
  for (int32_t i = 0; i < rows; ++i)
    for (int32_t j = 0; j < cols; ++j)
      Out[(size_t)i * cols + j] = X[axis == 0 ? i : j];
}

void polygeist_cuda_add_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  for (int32_t i = 0; i < N; ++i)
    Out[i] = X[i] + Y[i];
}

void polygeist_cudnn_pointwise_affine_relu_f32(
    int32_t N, float alpha, const float *X, const float *Bias, float *Out) {
  for (int32_t i = 0; i < N; ++i) {
    float value = alpha * X[i] + Bias[i];
    Out[i] = value > 0.0f ? value : 0.0f;
  }
}

void polygeist_cudnn_pointwise_graph_f32(
    int32_t N,
    int64_t graph0, int64_t graph1, int64_t graph2, int64_t graph3,
    int64_t graph4, int64_t graph5, int64_t graph6, int64_t graph7,
    int64_t graph8, int64_t graph9, int64_t graph10, int64_t graph11,
    int32_t num_nodes,
    float s0, float s1, float s2, float s3,
    float s4, float s5, float s6, float s7,
    int32_t stride0, int32_t stride1, int32_t stride2, int32_t stride3,
    int32_t out_stride,
    const float *In0, const float *In1, const float *In2, const float *In3,
    float *Out) {
  const float *inputs[4] = {In0, In1, In2, In3};
  const int32_t strides[4] = {stride0, stride1, stride2, stride3};
  const float scalars[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
  uint64_t words[12] = {
      (uint64_t)graph0, (uint64_t)graph1,
      (uint64_t)graph2, (uint64_t)graph3,
      (uint64_t)graph4, (uint64_t)graph5,
      (uint64_t)graph6, (uint64_t)graph7,
      (uint64_t)graph8, (uint64_t)graph9,
      (uint64_t)graph10, (uint64_t)graph11};
  for (int32_t i = 0; i < N; ++i) {
    float refs[36];
    for (int j = 0; j < 4; ++j) refs[j] = inputs[j][(int64_t)i * strides[j]];
    for (int j = 0; j < 8; ++j) refs[4 + j] = scalars[j];
    for (int node = 0; node < num_nodes; ++node) {
      uint32_t inst = (uint32_t)(words[node / 2] >> (32 * (node % 2)));
      int op = (inst >> 24) & 0xff;
      float lhs = refs[(inst >> 16) & 0xff];
      float rhs = refs[(inst >> 8) & 0xff];
      float third = refs[inst & 0xff];
      float value = NAN;
      switch (op) {
      case 1: value = lhs + rhs; break;
      case 2: value = lhs * rhs; break;
      case 3: value = lhs - rhs; break;
      case 4: value = lhs / rhs; break;
      case 5: value = lhs > 0.0f ? lhs : 0.0f; break;
      case 6: value = tanhf(lhs); break;
      case 7: value = expf(lhs); break;
      case 8: value = sqrtf(lhs); break;
      case 9: value = fabsf(lhs); break;
      case 10: value = fmaxf(lhs, rhs); break;
      case 11: value = fminf(lhs, rhs); break;
      case 12: value = logf(lhs); break;
      case 13: value = sinf(lhs); break;
      case 14: value = cosf(lhs); break;
      case 15: value = 1.0f / lhs; break;
      case 16: value = floorf(lhs); break;
      case 17: value = ceilf(lhs); break;
      case 18: value = erff(lhs); break;
      case 19: value = powf(lhs, rhs); break;
      case 20: value = fmodf(lhs, rhs); break;
      case 21: value = -lhs; break;
      case 22: value = tanf(lhs); break;
      case 23: value = lhs == rhs ? 1.0f : 0.0f; break;
      case 24: value = lhs != rhs ? 1.0f : 0.0f; break;
      case 25: value = lhs > rhs ? 1.0f : 0.0f; break;
      case 26: value = lhs >= rhs ? 1.0f : 0.0f; break;
      case 27: value = lhs < rhs ? 1.0f : 0.0f; break;
      case 28: value = lhs <= rhs ? 1.0f : 0.0f; break;
      case 29: value = lhs != 0.0f ? rhs : third; break;
      case 30: value = (lhs != 0.0f && rhs != 0.0f) ? 1.0f : 0.0f; break;
      case 31: value = (lhs != 0.0f || rhs != 0.0f) ? 1.0f : 0.0f; break;
      case 32: value = lhs == 0.0f ? 1.0f : 0.0f; break;
      case 33: value = lhs; break;
      case 34: value = atan2f(lhs, rhs); break;
      case 35: value = lhs > 0.0f ? rhs : 0.0f; break;
      default: break;
      }
      refs[12 + node] = value;
    }
    Out[(int64_t)i * out_stride] = refs[11 + num_nodes];
  }
}

void polygeist_cub_inclusive_sum1d_f32(
    int32_t n, const float *input, float *final_value, float *output) {
  float value = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    value += input[i];
    output[i] = value;
  }
  if (final_value) *final_value = value;
}

void polygeist_cub_segmented_inclusive_product2d_f32(
    int32_t rows, int32_t cols, const float *input,
    float *final_values, float *output) {
  for (int32_t row = 0; row < rows; ++row) {
    float value = 1.0f;
    for (int32_t col = 0; col < cols; ++col) {
      value *= input[(int64_t)row * cols + col];
      output[(int64_t)row * cols + col] = value;
    }
    final_values[row] = value;
  }
}

void polygeist_cub_exclusive_sum1d_i32(
    int32_t n, const int32_t *input, int32_t *output) {
  int32_t value = 0;
  for (int32_t i = 0; i < n; ++i) {
    output[i] = value;
    value += input[i];
  }
  output[n] = value;
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

static float polygeist_cutensor_unary_eval_f32(int32_t op, float x) {
  switch (op) {
  case POLYGEIST_CUTENSOR_UNARY_ABS: return fabsf(x);
  case POLYGEIST_CUTENSOR_UNARY_ACOS: return acosf(x);
  case POLYGEIST_CUTENSOR_UNARY_ACOSH: return acoshf(x);
  case POLYGEIST_CUTENSOR_UNARY_ASIN: return asinf(x);
  case POLYGEIST_CUTENSOR_UNARY_ASINH: return asinhf(x);
  case POLYGEIST_CUTENSOR_UNARY_ATAN: return atanf(x);
  case POLYGEIST_CUTENSOR_UNARY_ATANH: return atanhf(x);
  case POLYGEIST_CUTENSOR_UNARY_CEIL: return ceilf(x);
  case POLYGEIST_CUTENSOR_UNARY_COS: return cosf(x);
  case POLYGEIST_CUTENSOR_UNARY_COSH: return coshf(x);
  case POLYGEIST_CUTENSOR_UNARY_EXP: return expf(x);
  case POLYGEIST_CUTENSOR_UNARY_FLOOR: return floorf(x);
  case POLYGEIST_CUTENSOR_UNARY_LOG: return logf(x);
  case POLYGEIST_CUTENSOR_UNARY_MISH:
    return x * tanhf(log1pf(expf(x)));
  case POLYGEIST_CUTENSOR_UNARY_NEG: return -x;
  case POLYGEIST_CUTENSOR_UNARY_RECIPROCAL: return 1.0f / x;
  case POLYGEIST_CUTENSOR_UNARY_RELU: return x > 0.0f ? x : 0.0f;
  case POLYGEIST_CUTENSOR_UNARY_SIGMOID:
    return 1.0f / (1.0f + expf(-x));
  case POLYGEIST_CUTENSOR_UNARY_SILU:
    return x / (1.0f + expf(-x));
  case POLYGEIST_CUTENSOR_UNARY_SIN: return sinf(x);
  case POLYGEIST_CUTENSOR_UNARY_SINH: return sinhf(x);
  case POLYGEIST_CUTENSOR_UNARY_SQRT: return sqrtf(x);
  case POLYGEIST_CUTENSOR_UNARY_TAN: return tanf(x);
  case POLYGEIST_CUTENSOR_UNARY_TANH: return tanhf(x);
  default: return NAN;
  }
}

void polygeist_cudnn_reduce_f32(
    int32_t op, int32_t n, const float *x, float *out) {
  float acc = *out;
  for (int32_t i = 0; i < n; ++i) {
    if (op == 0) acc += x[i];
    else if (op == 1) acc *= x[i];
    else if (op == 2) acc = x[i] < acc ? x[i] : acc;
    else if (op == 3) acc = x[i] > acc ? x[i] : acc;
  }
  *out = acc;
}

void polygeist_cudnn_reduce_f64(
    int32_t op, int32_t n, const double *x, double *out) {
  double acc = *out;
  for (int32_t i = 0; i < n; ++i) {
    if (op == 0) acc += x[i];
    else if (op == 1) acc *= x[i];
    else if (op == 2) acc = x[i] < acc ? x[i] : acc;
    else if (op == 3) acc = x[i] > acc ? x[i] : acc;
  }
  *out = acc;
}

void polygeist_cudnn_reduce_diagonal_f32(
    int32_t rows, int32_t cols, int32_t row_stride, int32_t col_stride,
    const float *x, float *out) {
  int32_t n = rows < cols ? rows : cols;
  float acc = *out;
  int32_t stride = row_stride + col_stride;
  for (int32_t i = 0; i < n; ++i) acc += x[(size_t)i * stride];
  *out = acc;
}

void polygeist_cub_segmented_reduce_i32(
    int32_t op, int32_t rows, int32_t cols,
    const int32_t *x, int32_t *out) {
  for (int32_t row = 0; row < rows; ++row) {
    int32_t acc = op == 0 ? 1 : 0;
    for (int32_t col = 0; col < cols; ++col) {
      int32_t value = x[(size_t)row * cols + col];
      if (op == 0) acc = (acc != 0 && value != 0) ? 1 : 0;
      else if (op == 1) acc = (acc != 0 || value != 0) ? 1 : 0;
      else acc ^= value;
    }
    out[row] = acc;
  }
}

void polygeist_cub_segmented_reduce_f32(
    int32_t op, int32_t rows, int32_t cols, const float *x, float *out) {
  for (int32_t row = 0; row < rows; ++row) {
    float acc = op == 0 ? 0.0f : x[(size_t)row * cols];
    int32_t begin = op == 0 ? 0 : 1;
    for (int32_t col = begin; col < cols; ++col) {
      float value = x[(size_t)row * cols + col];
      if (op == 0) acc += value;
      else if (op == 1) acc = value < acc ? value : acc;
      else acc = value > acc ? value : acc;
    }
    out[row] = acc;
  }
}

void polygeist_cub_segmented_argreduce_f32(
    int32_t op, int32_t rows, int32_t cols,
    const float *x, int32_t *out) {
  for (int32_t row = 0; row < rows; ++row) {
    int32_t best = 0;
    float value = x[(size_t)row * cols];
    for (int32_t col = 1; col < cols; ++col) {
      float candidate = x[(size_t)row * cols + col];
      if ((op == 0 && candidate > value) ||
          (op == 1 && candidate < value)) {
        best = col;
        value = candidate;
      }
    }
    out[row] = best;
  }
}

void polygeist_cub_segmented_prefix_sum_f32(
    int32_t rows, int32_t cols, const float *x,
    const int32_t *lengths, float *out) {
  for (int32_t row = 0; row < rows; ++row) {
    int32_t end = lengths[row] < 0 ? 0 : lengths[row];
    if (end > cols) end = cols;
    float acc = 0.0f;
    for (int32_t col = 0; col < end; ++col)
      acc += x[(size_t)row * cols + col];
    out[row] = acc;
  }
}

void polygeist_cudnn_sinc_f32(int32_t n, const float *x, float *out) {
  const float pi = 3.14159265358979323846f;
  for (int32_t i = 0; i < n; ++i)
    out[i] = x[i] == 0.0f ? 1.0f : sinf(pi * x[i]) / (pi * x[i]);
}

void polygeist_cub_segmented_sort_descending_f32_i32(
    int32_t rows,int32_t cols,int32_t top,const float *input,
    float *values,int32_t *indices){
  if(top<0)top=0;if(top>cols)top=cols;
  float *scratch=(float*)malloc((size_t)cols*sizeof(float));
  int32_t *order=(int32_t*)malloc((size_t)cols*sizeof(int32_t));
  if(!scratch||!order)abort();
  for(int32_t row=0;row<rows;++row){
    for(int32_t col=0;col<cols;++col){scratch[col]=input[(int64_t)row*cols+col];order[col]=col;}
    for(int32_t col=1;col<cols;++col){float v=scratch[col];int32_t index=order[col],j=col-1;
      while(j>=0&&scratch[j]<v){scratch[j+1]=scratch[j];order[j+1]=order[j];--j;}
      scratch[j+1]=v;order[j+1]=index;}
    for(int32_t col=0;col<top;++col){values[(int64_t)row*top+col]=scratch[col];indices[(int64_t)row*top+col]=order[col];}
  }
  free(order);free(scratch);
}
void polygeist_cub_segment_reduce_lengths_f32(
    int32_t n,int32_t segments,int32_t op,const float *input,
    const int32_t *lengths,float *output){
  int32_t position=0;
  for(int32_t segment=0;segment<segments;++segment){float value=op==2?-3.402823466e38f:(op==3?3.402823466e38f:0.0f);
    for(int32_t i=0;i<lengths[segment]&&position<n;++i){float x=input[position++];
      if(op==0||op==1)value+=x;else if(op==2)value=value>x?value:x;else value=value<x?value:x;}
    if(op==1&&lengths[segment]>0)value/=lengths[segment];output[segment]=value;}
}
void polygeist_cub_segmented_prefix_logical_and_i32(
    int32_t rows, int32_t cols, const int32_t *x,
    const int32_t *lengths, int32_t *out) {
  for (int32_t row = 0; row < rows; ++row) {
    int32_t end = lengths[row] < 0 ? 0 : lengths[row];
    if (end > cols) end = cols;
    int32_t acc = 1;
    for (int32_t col = 0; col < end; ++col)
      acc = (acc != 0 && x[(size_t)row * cols + col] != 0) ? 1 : 0;
    out[row] = acc;
  }
}

void polygeist_cub_count_nonzero1d_f32(int32_t n,const float*in,int32_t*out){int32_t v=0;for(int32_t i=0;i<n;i++)v+=in[i]!=0.0f;*out=v;}
void polygeist_cub_segmented_count_nonzero2d_f32(int32_t r,int32_t c,const float*in,int32_t*out){for(int32_t i=0;i<r;i++){int32_t v=0;for(int32_t j=0;j<c;j++)v+=in[(int64_t)i*c+j]!=0.0f;out[i]=v;}}
void polygeist_cub_equal_all1d_f32(int32_t n,const float*a,const float*b,int32_t*out){int32_t v=1;for(int32_t i=0;i<n;i++)v=v&&(a[i]==b[i]);*out=v;}
void polygeist_cutensor_permute_f32(int32_t rank,const int64_t*ie,const int64_t*is,const int32_t*im,const int64_t*oe,const int64_t*os,const int32_t*om,const float*in,float*out){
  if(rank<1||rank>64)return;int64_t total=1;for(int d=0;d<rank;d++)total*=oe[d];
  for(int64_t linear=0;linear<total;linear++){int64_t rem=linear,coord[64]={0},oo=0,io=0;for(int d=rank-1;d>=0;d--){int64_t c=rem%oe[d];rem/=oe[d];coord[om[d]]=c;oo+=c*os[d];}for(int d=0;d<rank;d++)io+=coord[im[d]]*is[d];out[oo]=in[io];}
}

void polygeist_cutensor_unary_f32(
    int32_t op, int32_t n, const float *x, float *out) {
  for (int32_t i = 0; i < n; ++i)
    out[i] = polygeist_cutensor_unary_eval_f32(op, x[i]);
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
