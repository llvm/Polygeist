#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>

#include "mfem.hpp"

// MFEM's implementation headers reuse some internal aliases, just as normal
// MFEM translation units do.  Include only the selected kernel family.
#if defined(BENCH_MASS_2D) || defined(BENCH_MASS_3D)
#include "fem/integ/bilininteg_mass_kernels.hpp"
#elif defined(BENCH_DIFFUSION_2D) || defined(BENCH_DIFFUSION_3D)
#include "fem/integ/bilininteg_diffusion_kernels.hpp"
#elif defined(BENCH_CONVECTION_2D) || defined(BENCH_CONVECTION_3D)
#include "fem/integ/bilininteg_convection_kernels.hpp"
#elif defined(BENCH_CURLCURL_2D) || defined(BENCH_CURLCURL_3D) || \
    defined(BENCH_HCURL_MASS_3D)
#include "fem/integ/bilininteg_hcurl_kernels.hpp"
#elif defined(BENCH_DIVDIV_2D) || defined(BENCH_DIVDIV_3D) || \
    defined(BENCH_HDIV_MASS_3D)
#include "fem/integ/bilininteg_hdiv_kernels.hpp"
#else
#error "Select one BENCH_* kernel family"
#endif

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

using mfem::Array;
using mfem::Vector;
using mfem::real_t;

#if defined(BENCH_MASS_2D)
#define BENCH_NAME "mass_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &, const Array<real_t> &,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPAMassApply2D<4, 5>(MFEM_BENCH_NE, a0, a1, op, x, y);
}
#elif defined(BENCH_MASS_3D)
#define BENCH_NAME "mass_apply_3d"
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &, const Array<real_t> &,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPAMassApply3D<4, 5>(MFEM_BENCH_NE, a0, a1, op, x, y);
}
#elif defined(BENCH_DIFFUSION_2D)
#define BENCH_NAME "diffusion_apply_2d"
#define OP_SIZE (75 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPADiffusionApply2D<4, 5>(
      MFEM_BENCH_NE, true, a0, a1, a2, a3, op, x, y);
}
#elif defined(BENCH_DIFFUSION_3D)
#define BENCH_NAME "diffusion_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPADiffusionApply3D<4, 5>(
      MFEM_BENCH_NE, true, a0, a1, a2, a3, op, x, y);
}
#elif defined(BENCH_CONVECTION_2D)
#define BENCH_NAME "convection_apply_2d"
#define OP_SIZE (50 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::SmemPAConvectionApply2D<4, 5, mfem::convection::NBZ(4)>(
      MFEM_BENCH_NE, a0, a1, a2, a3, op, x, y);
}
#elif defined(BENCH_CONVECTION_3D)
#define BENCH_NAME "convection_apply_3d"
#define OP_SIZE (375 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::SmemPAConvectionApply3D<4, 5>(MFEM_BENCH_NE, a0, a1, a2, a3,
                                      op, x, y);
}
#elif defined(BENCH_CURLCURL_2D)
#define BENCH_NAME "curlcurl_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (24 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &a4, const Array<real_t> &a5,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::PACurlCurlApply2D(4, 5, true, MFEM_BENCH_NE, a0, a2,
                                    a1, a3, a4, a5, op, x, y);
}
#elif defined(BENCH_CURLCURL_3D)
#define BENCH_NAME "curlcurl_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (144 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &a4, const Array<real_t> &a5,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPACurlCurlApply3D<4, 5>(
      4, 5, true, MFEM_BENCH_NE, a0, a2, a1, a3, a4, a5, op, x, y);
}
#elif defined(BENCH_HCURL_MASS_3D)
#define BENCH_NAME "hcurl_mass_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (144 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPAHcurlMassApply3D<4, 5>(
      4, 5, MFEM_BENCH_NE, true, a0, a2, a1, a3, op, x, y);
}
#elif defined(BENCH_DIVDIV_2D)
#define BENCH_NAME "divdiv_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (24 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::PADivDivApply2D(4, 5, MFEM_BENCH_NE, a0, a2, a1, a3,
                                  op, x, y);
}
#elif defined(BENCH_DIVDIV_3D)
#define BENCH_NAME "divdiv_apply_3d"
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define X_SIZE (108 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::PADivDivApply3D(4, 5, MFEM_BENCH_NE, a0, a2, a1, a3,
                                  op, x, y);
}
#elif defined(BENCH_HDIV_MASS_3D)
#define BENCH_NAME "hdiv_mass_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (108 * MFEM_BENCH_NE)
#define Y_SIZE X_SIZE
static void launch(const Array<real_t> &a0, const Array<real_t> &a1,
                   const Array<real_t> &a2, const Array<real_t> &a3,
                   const Array<real_t> &, const Array<real_t> &,
                   const Vector &op, const Vector &x, Vector &y) {
  mfem::internal::SmemPAHdivMassApply3D<4, 5>(
      MFEM_BENCH_NE, true, a0, a2, a1, a3, op, x, y);
}
#else
#error "Select one BENCH_* operator"
#endif

static double value(int i, int salt) {
  return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0;
}

static void transpose(Array<real_t> &dst, const Array<real_t> &src,
                      int rows, int cols) {
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      dst[col * rows + row] = src[row * cols + col];
}

int main() {
  mfem::Device device("cuda");
  Array<real_t> a0(20), a1(20), a2(20), a3(20), a4(20), a5(20);
  Array<real_t> *arrays[] = {&a0, &a1, &a2, &a3, &a4, &a5};
  for (int a = 0; a < 6; ++a)
    for (int i = 0; i < arrays[a]->Size(); ++i)
      // The extracted C stores basis matrices row-major while MFEM's
      // DeviceMatrix view is column-major.  A per-matrix constant keeps the
      // same logical test input in both representations without conflating
      // layout conversion with operator correctness.
      (*arrays[a])[i] = value(0, a);
#if defined(BENCH_CURLCURL_2D) || defined(BENCH_CURLCURL_3D) || \
    defined(BENCH_HCURL_MASS_3D)
  transpose(a1, a0, 5, 3); // Bot = transpose(Bo)
  transpose(a3, a2, 5, 4); // Bct = transpose(Bc)
  transpose(a5, a4, 5, 4); // Gct = transpose(Gc)
#elif defined(BENCH_DIVDIV_2D) || defined(BENCH_DIVDIV_3D) || \
    defined(BENCH_HDIV_MASS_3D)
  transpose(a1, a0, 5, 3); // Bot = transpose(Bo)
  transpose(a3, a2, 5, 4); // Gct = transpose(Gc)
#elif defined(BENCH_MASS_2D) || defined(BENCH_MASS_3D)
  transpose(a1, a0, 5, 4); // Bt = transpose(B)
#else
  transpose(a2, a0, 5, 4); // Bt = transpose(B)
  transpose(a3, a1, 5, 4); // Gt = transpose(G)
#endif

  Vector op(OP_SIZE), x(X_SIZE), y(Y_SIZE);
  for (int i = 0; i < op.Size(); ++i) op[i] = value(i, 6);
  for (int i = 0; i < x.Size(); ++i) x[i] = value(i, 7);
  y = 0.0;

  launch(a0, a1, a2, a3, a4, a5, op, x, y);
  cudaDeviceSynchronize();
  const real_t *host_y = y.HostRead();
  double checksum = 0.0;
  double max_abs = 0.0;
  for (int i = 0; i < y.Size(); ++i) {
    checksum += host_y[i];
    max_abs = fmax(max_abs, fabs(host_y[i]));
  }

  y = 0.0;
  for (int i = 0; i < 3; ++i) launch(a0, a1, a2, a3, a4, a5, op, x, y);
  cudaDeviceSynchronize();
  double us = INFINITY;
  for (int i = 0; i < BENCH_ITERS; ++i) {
    const auto start = std::chrono::steady_clock::now();
    launch(a0, a1, a2, a3, a4, a5, op, x, y);
    cudaDeviceSynchronize();
    const auto stop = std::chrono::steady_clock::now();
    us = fmin(us, std::chrono::duration<double, std::micro>(stop - start).count());
  }
  std::printf("implementation=mfem_native_cuda kernel=%s ne=%d iterations=%d "
              "runtime_us=%.6f checksum=%.17g max_abs=%.17g "
              "timing_scope=resident_best_of_n\n",
              BENCH_NAME, MFEM_BENCH_NE, BENCH_ITERS, us, checksum, max_abs);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
