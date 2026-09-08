#define main mfem_native_legacy_main
#ifdef BENCH_DFEM
#include "mfem_native_cuda_dfem_bench.cpp"
#else
#include "mfem_native_cuda_pa_bench.cpp"
#endif
#undef main

int main() {
  mfem::Device device("cuda");

#ifdef BENCH_DFEM
  Vector basis(20), gradient(20), input(INPUT_SIZE), output(OUTPUT_SIZE);
  // The normalized C fixtures store B(q,d) row-major. DeviceTensor stores its
  // first index contiguously, so explicitly preserve the same logical matrix.
  for (int q = 0; q < Q1D; ++q)
    for (int d = 0; d < D1D; ++d) {
      basis[q + Q1D * d] = value(q * D1D + d, 1);
      gradient[q + Q1D * d] = value(q * D1D + d, 2);
    }
  for (int i = 0; i < INPUT_SIZE; ++i) input[i] = value(i, 8);
#if !IS_INTERPOLATION && IS_GRADIENT
  // The normalized fixtures group quadrature values by component, whereas
  // DeviceTensor<3>(1, TERMS, QPTS) keeps the component index contiguous.
  for (int e = 0; e < MFEM_BENCH_NE; ++e)
    for (int c = 0; c < TERMS; ++c)
      for (int q_native = 0; q_native < QPTS; ++q_native) {
        int remaining = q_native;
        int q_normalized = 0;
        for (int axis = 0; axis < DIMENSION; ++axis) {
          q_normalized = q_normalized * Q1D + remaining % Q1D;
          remaining /= Q1D;
        }
        input[c + TERMS * (q_native + QPTS * e)] =
            value(q_normalized + QPTS * (c + TERMS * e), 8);
      }
#endif
  for (int i = 0; i < OUTPUT_SIZE; ++i) output[i] = value(i, 9);
#if IS_GRADIENT
  Gradient<> field_operator;
#else
  Value<> field_operator;
#endif
  field_operator.vdim = 1;
  field_operator.dim = DIMENSION;
  field_operator.size_on_qp = TERMS;
#define RUN_NATIVE() launch(basis, gradient, input, output, field_operator)
#define OUTPUT_VECTOR output
#else
  Array<real_t> a0(20), a1(20), a2(20), a3(20), a4(20), a5(20);
  Array<real_t> n0(20), n1(20), n2(20), n3(20), n4(20), n5(20);
  Array<real_t> *normalized[] = {&n0, &n1, &n2, &n3, &n4, &n5};
  for (int a = 0; a < 6; ++a)
    for (int i = 0; i < normalized[a]->Size(); ++i)
      (*normalized[a])[i] = value(i, a + 1);
#if defined(BENCH_CURLCURL_2D) || defined(BENCH_CURLCURL_3D) || \
    defined(BENCH_HCURL_MASS_3D)
  transpose(n1, n0, 5, 3);
  transpose(n3, n2, 5, 4);
  transpose(n5, n4, 5, 4);
  transpose(a0, n0, 5, 3); transpose(a1, n1, 3, 5);
  transpose(a2, n2, 5, 4); transpose(a3, n3, 4, 5);
  transpose(a4, n4, 5, 4); transpose(a5, n5, 4, 5);
#elif defined(BENCH_DIVDIV_2D) || defined(BENCH_DIVDIV_3D) || \
    defined(BENCH_HDIV_MASS_3D)
  transpose(n1, n0, 5, 3);
  transpose(n3, n2, 5, 4);
  transpose(a0, n0, 5, 3); transpose(a1, n1, 3, 5);
  transpose(a2, n2, 5, 4); transpose(a3, n3, 4, 5);
  transpose(a4, n4, 5, 4); transpose(a5, n5, 4, 5);
#elif defined(BENCH_MASS_2D) || defined(BENCH_MASS_3D)
  transpose(n1, n0, 5, 4);
  transpose(a0, n0, 5, 4); transpose(a1, n1, 4, 5);
  transpose(a2, n2, 5, 4); transpose(a3, n3, 5, 4);
  transpose(a4, n4, 5, 4); transpose(a5, n5, 5, 4);
#else
  transpose(n2, n0, 5, 4);
  transpose(n3, n1, 5, 4);
  transpose(a0, n0, 5, 4); transpose(a1, n1, 5, 4);
  transpose(a2, n2, 4, 5); transpose(a3, n3, 4, 5);
  transpose(a4, n4, 5, 4); transpose(a5, n5, 5, 4);
#endif
  Vector op(OP_SIZE), x(X_SIZE), y(Y_SIZE);
  for (int i = 0; i < op.Size(); ++i) op[i] = value(i, 7);
  for (int i = 0; i < x.Size(); ++i) x[i] = value(i, 8);
  for (int i = 0; i < y.Size(); ++i) y[i] = value(i, 9);
#define RUN_NATIVE() launch(a0, a1, a2, a3, a4, a5, op, x, y)
#define OUTPUT_VECTOR y
#endif

  for (int warmup = 0; warmup < 5; ++warmup) RUN_NATIVE();
  cudaDeviceSynchronize();
  for (int sample = 0; sample < 20; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    RUN_NATIVE();
    cudaDeviceSynchronize();
    const auto stop = std::chrono::steady_clock::now();
    const double runtime_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    std::printf("implementation=mfem_native_cuda kernel=%s ne=%d sample=%d "
                "runtime_us=%.9f\n",
                BENCH_NAME, MFEM_BENCH_NE, sample, runtime_us);
  }
  const real_t *host_output = OUTPUT_VECTOR.HostRead();
  double checksum = 0.0;
  for (int i = 0; i < OUTPUT_VECTOR.Size(); ++i) checksum += host_output[i];
  std::fprintf(stderr, "kernel=%s final_checksum=%.17g\n", BENCH_NAME, checksum);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
