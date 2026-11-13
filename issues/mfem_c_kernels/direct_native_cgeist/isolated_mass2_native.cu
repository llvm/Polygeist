// Standalone extraction of MFEM SmemPAMassApply2D_Element.
//
// Computational source:
//   mfem/fem/integ/bilininteg_mass_kernels.hpp
//   revision 951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5
//
// The kernel body below preserves MFEM's CUDA thread assignment, shared-memory
// staging, synchronization, indexing, and accumulating output semantics. The
// surrounding MFEM Array/Vector/DeviceTensor and forall dispatch machinery is
// replaced by a standalone harness and equivalent lightweight tensor views.

#include <cuda_runtime.h>

#ifndef POLYGEIST_TRANSLATION_ONLY
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#endif

using real_t = double;

namespace extracted_mfem {

constexpr int D1D = 4;
constexpr int Q1D = 5;
constexpr int NBZ = 8; // mfem::mass::NBZ(4)

#define MFEM_THREAD_ID(k) threadIdx.k
#define MFEM_SHARED __shared__
#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_FOREACH_THREAD(i, k, N)                                        \
  for (int i = threadIdx.k; i < (N); i += blockDim.k)

// Copied structurally from upstream SmemPAMassApply2D_Element<4,5,8,true>.
__device__ __forceinline__ void mass_apply_2d_element(
    const int e, const int NE, const real_t *b_, const real_t *d_,
    const real_t *x_, real_t *y_) {
  constexpr int MDQ = Q1D;

  const int tidz = MFEM_THREAD_ID(z);

  MFEM_SHARED real_t BBt[Q1D * D1D];
  real_t (*B)[D1D] = reinterpret_cast<real_t (*)[D1D]>(BBt);
  real_t (*Bt)[Q1D] = reinterpret_cast<real_t (*)[Q1D]>(BBt);
  MFEM_SHARED real_t sm0[NBZ][MDQ * MDQ];
  MFEM_SHARED real_t sm1[NBZ][MDQ * MDQ];
  real_t (*X)[D1D] = reinterpret_cast<real_t (*)[D1D]>(sm0 + tidz);
  real_t (*DQ)[Q1D] = reinterpret_cast<real_t (*)[Q1D]>(sm1 + tidz);
  real_t (*QQ)[Q1D] = reinterpret_cast<real_t (*)[Q1D]>(sm0 + tidz);
  real_t (*QD)[D1D] = reinterpret_cast<real_t (*)[D1D]>(sm1 + tidz);

  MFEM_FOREACH_THREAD(dy, y, D1D) {
    MFEM_FOREACH_THREAD(dx, x, D1D) {
      X[dy][dx] = x_[dx + D1D * (dy + D1D * e)];
    }
  }
  if (tidz == 0) {
    MFEM_FOREACH_THREAD(dy, y, D1D) {
      MFEM_FOREACH_THREAD(q, x, Q1D) { B[q][dy] = b_[q + Q1D * dy]; }
    }
  }
  MFEM_SYNC_THREAD;
  MFEM_FOREACH_THREAD(dy, y, D1D) {
    MFEM_FOREACH_THREAD(qx, x, Q1D) {
      real_t dq = 0.0;
      for (int dx = 0; dx < D1D; ++dx) { dq += X[dy][dx] * B[qx][dx]; }
      DQ[dy][qx] = dq;
    }
  }
  MFEM_SYNC_THREAD;
  MFEM_FOREACH_THREAD(qy, y, Q1D) {
    MFEM_FOREACH_THREAD(qx, x, Q1D) {
      real_t qq = 0.0;
      for (int dy = 0; dy < D1D; ++dy) { qq += DQ[dy][qx] * B[qy][dy]; }
      QQ[qy][qx] = qq * d_[qx + Q1D * (qy + Q1D * e)];
    }
  }
  MFEM_SYNC_THREAD;
  if (tidz == 0) {
    MFEM_FOREACH_THREAD(dy, y, D1D) {
      MFEM_FOREACH_THREAD(q, x, Q1D) { Bt[dy][q] = b_[q + Q1D * dy]; }
    }
  }
  MFEM_SYNC_THREAD;
  MFEM_FOREACH_THREAD(qy, y, Q1D) {
    MFEM_FOREACH_THREAD(dx, x, D1D) {
      real_t dq = 0.0;
      for (int qx = 0; qx < Q1D; ++qx) { dq += QQ[qy][qx] * Bt[dx][qx]; }
      QD[qy][dx] = dq;
    }
  }
  MFEM_SYNC_THREAD;
  MFEM_FOREACH_THREAD(dy, y, D1D) {
    MFEM_FOREACH_THREAD(dx, x, D1D) {
      real_t dd = 0.0;
      for (int qy = 0; qy < Q1D; ++qy) { dd += QD[qy][dx] * Bt[dy][qy]; }
      y_[dx + D1D * (dy + D1D * e)] += dd;
    }
  }
}

__global__ void mass_apply_2d_kernel(const int NE, const real_t *b,
                                     const real_t *D, const real_t *x,
                                     real_t *y) {
  // This is the indexing used by upstream mfem::CuKernel2D.
  const int e = blockIdx.x * blockDim.z + threadIdx.z;
  if (e >= NE) { return; }
  mass_apply_2d_element(e, NE, b, D, x, y);
}

#ifdef POLYGEIST_TRANSLATION_ONLY
// A plain host entry point keeps the CUDA computation reachable when cgeist
// translates this file without the native validation harness.
extern "C" void mfem_isolated_mass2_entry(const int NE, const real_t *b,
                                           const real_t *D, const real_t *x,
                                           real_t *y) {
  const dim3 block(Q1D, Q1D, NBZ);
  const dim3 grid((NE + NBZ - 1) / NBZ);
  mass_apply_2d_kernel<<<grid, block>>>(NE, b, D, x, y);
}
#endif

#undef MFEM_FOREACH_THREAD
#undef MFEM_SYNC_THREAD
#undef MFEM_SHARED
#undef MFEM_THREAD_ID

} // namespace extracted_mfem

#ifndef POLYGEIST_TRANSLATION_ONLY
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t status_ = (expr);                                       \
    if (status_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status_));                              \
      return 2;                                                              \
    }                                                                        \
  } while (false)

static double input_value(int i, int salt) {
  const int v = ((i * 17 + salt * 13 + 5) % 97) - 48;
  return static_cast<double>(v) / 113.0;
}

static void reference_mass_apply_2d(int NE, const real_t *b, const real_t *D,
                                    const real_t *x, real_t *y) {
  using namespace extracted_mfem;
  for (int e = 0; e < NE; ++e) {
    real_t dq[D1D][Q1D];
    real_t qq[Q1D][Q1D];
    real_t qd[Q1D][D1D];
    for (int dy = 0; dy < D1D; ++dy) {
      for (int qx = 0; qx < Q1D; ++qx) {
        dq[dy][qx] = 0.0;
        for (int dx = 0; dx < D1D; ++dx) {
          dq[dy][qx] += x[dx + D1D * (dy + D1D * e)] *
                        b[qx + Q1D * dx];
        }
      }
    }
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int qx = 0; qx < Q1D; ++qx) {
        qq[qy][qx] = 0.0;
        for (int dy = 0; dy < D1D; ++dy) {
          qq[qy][qx] += dq[dy][qx] * b[qy + Q1D * dy];
        }
        qq[qy][qx] *= D[qx + Q1D * (qy + Q1D * e)];
      }
    }
    for (int qy = 0; qy < Q1D; ++qy) {
      for (int dx = 0; dx < D1D; ++dx) {
        qd[qy][dx] = 0.0;
        for (int qx = 0; qx < Q1D; ++qx) {
          qd[qy][dx] += qq[qy][qx] * b[qx + Q1D * dx];
        }
      }
    }
    for (int dy = 0; dy < D1D; ++dy) {
      for (int dx = 0; dx < D1D; ++dx) {
        real_t value = 0.0;
        for (int qy = 0; qy < Q1D; ++qy) {
          value += qd[qy][dx] * b[qy + Q1D * dy];
        }
        y[dx + D1D * (dy + D1D * e)] += value;
      }
    }
  }
}

int main(int argc, char **argv) {
  using namespace extracted_mfem;
  const int NE = argc > 1 ? std::atoi(argv[1]) : 1024;
  if (NE <= 0 || NE % NBZ != 0) {
    std::fprintf(stderr, "NE must be positive and divisible by %d\n", NBZ);
    return 1;
  }

  const size_t b_count = Q1D * D1D;
  const size_t d_count = static_cast<size_t>(Q1D) * Q1D * NE;
  const size_t xy_count = static_cast<size_t>(D1D) * D1D * NE;
  real_t *h_b = new real_t[b_count];
  real_t *h_D = new real_t[d_count];
  real_t *h_x = new real_t[xy_count];
  real_t *h_y_initial = new real_t[xy_count];
  real_t *h_y_reference = new real_t[xy_count];
  real_t *h_y_gpu = new real_t[xy_count];

  for (size_t i = 0; i < b_count; ++i) { h_b[i] = input_value(i, 1); }
  for (size_t i = 0; i < d_count; ++i) { h_D[i] = input_value(i, 2); }
  for (size_t i = 0; i < xy_count; ++i) {
    h_x[i] = input_value(i, 3);
    h_y_initial[i] = input_value(i, 4);
    h_y_reference[i] = h_y_initial[i];
  }
  reference_mass_apply_2d(NE, h_b, h_D, h_x, h_y_reference);

  real_t *d_b = nullptr;
  real_t *d_D = nullptr;
  real_t *d_x = nullptr;
  real_t *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_D, d_count * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_x, xy_count * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_y, xy_count * sizeof(real_t)));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, b_count * sizeof(real_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D, h_D, d_count * sizeof(real_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, xy_count * sizeof(real_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y_initial, xy_count * sizeof(real_t),
                        cudaMemcpyHostToDevice));

  const dim3 block(Q1D, Q1D, NBZ);
  const dim3 grid((NE + NBZ - 1) / NBZ);
  mass_apply_2d_kernel<<<grid, block>>>(NE, d_b, d_D, d_x, d_y);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, xy_count * sizeof(real_t),
                        cudaMemcpyDeviceToHost));

  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t mismatches = 0;
  size_t nonfinite = 0;
  double checksum_reference = 0.0;
  double checksum_gpu = 0.0;
  constexpr double atol = 2.0e-12;
  constexpr double rtol = 2.0e-12;
  for (size_t i = 0; i < xy_count; ++i) {
    const double expected = h_y_reference[i];
    const double actual = h_y_gpu[i];
    checksum_reference += expected;
    checksum_gpu += actual;
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      ++nonfinite;
      ++mismatches;
      continue;
    }
    const double abs_error = std::fabs(actual - expected);
    const double rel_error = abs_error / std::fmax(std::fabs(expected), 1.0e-30);
    max_abs = std::fmax(max_abs, abs_error);
    max_rel = std::fmax(max_rel, rel_error);
    if (abs_error > atol + rtol * std::fabs(expected)) { ++mismatches; }
  }

  // Time only resident kernel execution plus synchronization. Use a separate
  // output because the upstream operator intentionally accumulates into y.
  CUDA_CHECK(cudaMemset(d_y, 0, xy_count * sizeof(real_t)));
  for (int i = 0; i < 5; ++i) {
    mass_apply_2d_kernel<<<grid, block>>>(NE, d_b, d_D, d_x, d_y);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int sample = 0; sample < 5; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    mass_apply_2d_kernel<<<grid, block>>>(NE, d_b, d_D, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const double runtime_us =
        std::chrono::duration<double, std::micro>(end - begin).count();
    std::printf("sample=%d synchronized_wall_us=%.9f\n", sample, runtime_us);
  }

  std::printf("implementation=isolated_upstream_mfem_cuda "
              "kernel=mass_apply_2d ne=%d d1d=%d q1d=%d block=%dx%dx%d "
              "elements=%zu mismatches=%zu nonfinite=%zu "
              "max_abs=%.17g max_rel=%.17g checksum_reference=%.17g "
              "checksum_gpu=%.17g correctness=%s\n",
              NE, D1D, Q1D, Q1D, Q1D, NBZ, xy_count, mismatches, nonfinite,
              max_abs, max_rel, checksum_reference, checksum_gpu,
              mismatches == 0 ? "PASS" : "FAIL");

  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_D);
  cudaFree(d_b);
  delete[] h_y_gpu;
  delete[] h_y_reference;
  delete[] h_y_initial;
  delete[] h_x;
  delete[] h_D;
  delete[] h_b;
  return mismatches == 0 ? 0 : 3;
}
#endif
