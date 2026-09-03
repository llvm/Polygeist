#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "mfem.hpp"
#include "fem/dfem/integrate.hpp"
#include "fem/dfem/interpolate.hpp"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

using mfem::Vector;
using mfem::DeviceTensor;
using mfem::real_t;
using namespace mfem::future;

#if defined(BENCH_INTERP_VALUE_2D)
#define BENCH_NAME "interp_value_2d"
#define DIMENSION 2
#define IS_INTERPOLATION 1
#define IS_GRADIENT 0
#elif defined(BENCH_INTERP_VALUE_3D)
#define BENCH_NAME "interp_value_3d"
#define DIMENSION 3
#define IS_INTERPOLATION 1
#define IS_GRADIENT 0
#elif defined(BENCH_INTEGRATE_VALUE_2D)
#define BENCH_NAME "integrate_value_2d"
#define DIMENSION 2
#define IS_INTERPOLATION 0
#define IS_GRADIENT 0
#elif defined(BENCH_INTEGRATE_VALUE_3D)
#define BENCH_NAME "integrate_value_3d"
#define DIMENSION 3
#define IS_INTERPOLATION 0
#define IS_GRADIENT 0
#elif defined(BENCH_INTERP_GRAD_2D)
#define BENCH_NAME "interp_grad_2d"
#define DIMENSION 2
#define IS_INTERPOLATION 1
#define IS_GRADIENT 1
#elif defined(BENCH_INTERP_GRAD_3D)
#define BENCH_NAME "interp_grad_3d"
#define DIMENSION 3
#define IS_INTERPOLATION 1
#define IS_GRADIENT 1
#elif defined(BENCH_INTEGRATE_GRAD_2D)
#define BENCH_NAME "integrate_grad_2d"
#define DIMENSION 2
#define IS_INTERPOLATION 0
#define IS_GRADIENT 1
#elif defined(BENCH_INTEGRATE_GRAD_3D)
#define BENCH_NAME "integrate_grad_3d"
#define DIMENSION 3
#define IS_INTERPOLATION 0
#define IS_GRADIENT 1
#else
#error "Select one BENCH_{INTERP,INTEGRATE}_{VALUE,GRAD}_{2D,3D} kernel"
#endif

static constexpr int D1D = 4;
static constexpr int Q1D = 5;
static constexpr int DOFS = DIMENSION == 2 ? D1D * D1D : D1D * D1D * D1D;
static constexpr int QPTS = DIMENSION == 2 ? Q1D * Q1D : Q1D * Q1D * Q1D;
static constexpr int TERMS = IS_GRADIENT ? DIMENSION : 1;
static constexpr int INPUT_SIZE =
    IS_INTERPOLATION ? DOFS * MFEM_BENCH_NE
                     : TERMS * QPTS * MFEM_BENCH_NE;
static constexpr int OUTPUT_SIZE =
    IS_INTERPOLATION ? TERMS * QPTS * MFEM_BENCH_NE
                     : DOFS * MFEM_BENCH_NE;
static constexpr int SCRATCH_SLICE = 100;
static constexpr int SCRATCH_SIZE = 6 * SCRATCH_SLICE;

static double value(int i, int salt) {
  return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0;
}

template <typename FieldOperatorType>
static void launch(const Vector &basis, const Vector &gradient,
                   const Vector &input, Vector &output,
                   FieldOperatorType field_operator) {
  const real_t *B = basis.Read();
  const real_t *G = gradient.Read();
  const real_t *X = input.Read();
  real_t *Y = output.ReadWrite();
  const DofToQuadMap dtq{
      DeviceTensor<3, const real_t>(B, Q1D, 1, D1D),
      DeviceTensor<3, const real_t>(G, Q1D, 1, D1D), -1};
  const DeviceTensor<1, const real_t> weights(nullptr, 0);
  const std::array<int, 6> scratch_sizes = {
      SCRATCH_SLICE, SCRATCH_SLICE, SCRATCH_SLICE,
      SCRATCH_SLICE, SCRATCH_SLICE, SCRATCH_SLICE};
  ThreadBlocks blocks{Q1D, Q1D, DIMENSION == 3 ? Q1D : 1};

  mfem::future::forall(
      [=] MFEM_HOST_DEVICE(int e, void *shared_memory) mutable {
        auto scratch = load_scratch_mem(
            shared_memory, 0, scratch_sizes);
#if IS_INTERPOLATION
        DeviceTensor<1> field_e(const_cast<real_t *>(X) + e * DOFS, DOFS);
        DeviceTensor<2> field_qp(Y + e * TERMS * QPTS, TERMS, QPTS);
#if DIMENSION == 2
        map_field_to_quadrature_data_tensor_product_2d(
            field_qp, dtq, field_e, field_operator, weights, scratch);
#else
        map_field_to_quadrature_data_tensor_product_3d(
            field_qp, dtq, field_e, field_operator, weights, scratch);
#endif
#else
        DeviceTensor<3> field_qp(
            const_cast<real_t *>(X) + e * TERMS * QPTS, 1, TERMS, QPTS);
        DeviceTensor<2> field_e(Y + e * DOFS, DOFS, 1);
#if DIMENSION == 2
        map_quadrature_data_to_fields_tensor_impl_2d(
            field_e, field_qp, field_operator, dtq, scratch);
#else
        map_quadrature_data_to_fields_tensor_impl_3d(
            field_e, field_qp, field_operator, dtq, scratch);
#endif
#endif
      },
      MFEM_BENCH_NE, blocks, SCRATCH_SIZE);
}

int main() {
  mfem::Device device("cuda");
  Vector basis(20), gradient(20), input(INPUT_SIZE), output(OUTPUT_SIZE);
  for (int i = 0; i < 20; ++i) {
    basis[i] = value(0, 0);
    gradient[i] = value(0, 1);
  }
  for (int i = 0; i < INPUT_SIZE; ++i) input[i] = value(i, 7);
  output = 0.0;

#if IS_GRADIENT
  Gradient<> field_operator;
#else
  Value<> field_operator;
#endif
  field_operator.vdim = 1;
  field_operator.dim = DIMENSION;
  field_operator.size_on_qp = TERMS;

  launch(basis, gradient, input, output, field_operator);
  cudaDeviceSynchronize();
  const real_t *host_output = output.HostRead();
  double checksum = 0.0;
  double max_abs = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    checksum += host_output[i];
    max_abs = fmax(max_abs, fabs(host_output[i]));
  }

  output = 0.0;
  for (int i = 0; i < 3; ++i)
    launch(basis, gradient, input, output, field_operator);
  cudaDeviceSynchronize();
  double runtime_us = INFINITY;
  for (int i = 0; i < BENCH_ITERS; ++i) {
    const auto start = std::chrono::steady_clock::now();
    launch(basis, gradient, input, output, field_operator);
    cudaDeviceSynchronize();
    const auto stop = std::chrono::steady_clock::now();
    runtime_us = fmin(
        runtime_us,
        std::chrono::duration<double, std::micro>(stop - start).count());
  }
  std::printf("implementation=mfem_native_cuda kernel=%s ne=%d iterations=%d "
              "runtime_us=%.6f checksum=%.17g max_abs=%.17g "
              "timing_scope=resident_best_of_n\n",
              BENCH_NAME, MFEM_BENCH_NE, BENCH_ITERS, runtime_us,
              checksum, max_abs);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
