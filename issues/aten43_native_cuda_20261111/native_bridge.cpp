#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Genuine libtorch dispatch only. This file defines no CUDA kernel or operation formula.
struct NativeContext {
  std::vector<at::Tensor> tensors;
  std::function<void(const float *)> operation;
  at::Tensor allocated_output;
  void *external_output = nullptr;
  size_t output_bytes = 0;
};

extern "C" void *native_prepare(const char *name, float **p, const size_t *sizes,
                                int count) {
  auto *c = new NativeContext;
  at::set_num_threads(1);
  for (int i = 0; i < count; ++i)
    c->tensors.push_back(at::from_blob(p[i], {(int64_t)sizes[i]},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA)));
  const std::string op(name);
#define UNARY(NAME) if (op == #NAME) c->operation = [c](const float *) { \
  at::NAME##_out(c->tensors[1], c->tensors[0]); }; else
  UNARY(abs) UNARY(neg) UNARY(exp) UNARY(log) UNARY(sqrt)
  UNARY(reciprocal) UNARY(sin) UNARY(cos) UNARY(tanh) UNARY(sigmoid)
  UNARY(floor) UNARY(ceil) UNARY(erf) UNARY(square) UNARY(rsqrt) UNARY(silu)
  UNARY(acos) UNARY(acosh) UNARY(asin) UNARY(asinh) UNARY(atan) UNARY(atanh)
  UNARY(cosh) UNARY(sinh) UNARY(tan)
#undef UNARY
#define BINARY(NAME) if (op == #NAME) c->operation = [c](const float *) { \
  at::NAME##_out(c->tensors[2], c->tensors[0], c->tensors[1]); }; else
  BINARY(pow) BINARY(atan2) BINARY(mul) BINARY(div)
  BINARY(sigmoid_backward) BINARY(tanh_backward) BINARY(silu_backward)
  BINARY(hardsigmoid_backward) BINARY(hardswish_backward)
#undef BINARY
  if (op == "add") c->operation = [c](const float *) {
    c->tensors[1].add_(c->tensors[0]);
  }; else if (op == "threshold_backward") c->operation = [c](const float *s) {
    at::threshold_backward_out(c->tensors[2], c->tensors[0], c->tensors[1], s[0]);
  }; else if (op == "hardshrink") c->operation = [c](const float *s) {
    at::hardshrink_out(c->tensors[1], c->tensors[0], s[0]);
  }; else if (op == "softshrink") c->operation = [c](const float *s) {
    at::softshrink_out(c->tensors[1], c->tensors[0], s[0]);
  }; else if (op == "shrink_backward") c->operation = [c](const float *s) {
    at::hardshrink_backward_out(c->tensors[2], c->tensors[0], c->tensors[1], s[0]);
  }; else if (op == "hardtanh_backward") c->operation = [c](const float *s) {
    at::hardtanh_backward_out(c->tensors[2], c->tensors[0], c->tensors[1], s[0], s[1]);
  }; else if (op == "addcmul") c->operation = [c](const float *s) {
    at::addcmul_out(c->tensors[3], c->tensors[0], c->tensors[1], c->tensors[2], s[0]);
  }; else if (op == "addcdiv") c->operation = [c](const float *s) {
    at::addcdiv_out(c->tensors[3], c->tensors[0], c->tensors[1], c->tensors[2], s[0]);
  }; else if (op == "rms_norm") {
    // Public ATen RMSNorm has no out variant. Keep allocation in the timed API;
    // export its result only outside timing. This boundary is reported explicitly.
    c->external_output = p[2]; c->output_bytes = sizes[2] * sizeof(float);
    c->operation = [c](const float *s) {
      c->allocated_output = at::rms_norm(c->tensors[0], {c->tensors[0].numel()}, c->tensors[1], s[0]);
    };
  } else throw std::runtime_error("No native ATen mapping: " + op);
  static bool printed = false;
  if (!printed) {
    cudaDeviceProp props{}; int runtime = 0, driver = 0;
    cudaGetDeviceProperties(&props, 0); cudaRuntimeGetVersion(&runtime); cudaDriverGetVersion(&driver);
    std::printf("NATIVE_BACKEND libtorch CUDA device=%s runtime=%d driver=%d tensor_cuda=%d\n",
                props.name, runtime, driver, c->tensors[0].is_cuda());
    printed = true;
  }
  return c;
}

extern "C" void native_run(void *context, const float *parameters) {
  static_cast<NativeContext *>(context)->operation(parameters);
}

extern "C" void native_export(void *context) {
  auto *c = static_cast<NativeContext *>(context);
  if (c->allocated_output.defined()) {
    auto e = cudaMemcpy(c->external_output, c->allocated_output.data_ptr<float>(),
                        c->output_bytes, cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  }
}
