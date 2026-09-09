#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kPointwiseElements = 4194304;
bool g_use_cuda = true;

void cuda_ok(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

void synchronize(const char *what) {
  if (g_use_cuda)
    cuda_ok(cudaDeviceSynchronize(), what);
}

std::vector<float> fixture_values(int64_t n) {
  std::vector<float> values(n);
  for (int64_t i = 0; i < n; ++i)
    values[i] = (static_cast<float>(i % 257) - 128.0f) / 64.0f;
  return values;
}

std::vector<float> raised_fixture_values(int64_t n) {
  std::vector<float> values(n);
  for (int64_t i = 0; i < n; ++i)
    values[i] = (static_cast<float>(i % 101) - 50.0f) / 37.0f;
  return values;
}

at::Tensor device_tensor(const std::vector<float> &host) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(
      g_use_cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU));
  auto tensor = at::empty({static_cast<int64_t>(host.size())}, options);
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(tensor.data_ptr<float>(), host.data(),
                       host.size() * sizeof(float), cudaMemcpyHostToDevice),
            "copy input");
  else
    std::memcpy(tensor.data_ptr<float>(), host.data(),
                host.size() * sizeof(float));
  return tensor;
}

at::Tensor device_int_tensor(const std::vector<int32_t> &host) {
  auto options = at::TensorOptions().dtype(at::kInt).device(
      g_use_cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU));
  auto tensor = at::empty({static_cast<int64_t>(host.size())}, options);
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(tensor.data_ptr<int32_t>(), host.data(),
                       host.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
            "copy integer input");
  else
    std::memcpy(tensor.data_ptr<int32_t>(), host.data(),
                host.size() * sizeof(int32_t));
  return tensor;
}

at::Tensor device_byte_tensor(const std::vector<int8_t> &host) {
  auto options = at::TensorOptions().dtype(at::kChar).device(
      g_use_cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU));
  auto tensor = at::empty({static_cast<int64_t>(host.size())}, options);
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(tensor.data_ptr<int8_t>(), host.data(), host.size(),
                       cudaMemcpyHostToDevice), "copy int8 input");
  else
    std::memcpy(tensor.data_ptr<int8_t>(), host.data(), host.size());
  return tensor;
}

at::Tensor cpu_byte_tensor(const std::vector<int8_t> &host) {
  auto tensor = at::empty({static_cast<int64_t>(host.size())},
                          at::TensorOptions().dtype(at::kChar));
  std::memcpy(tensor.data_ptr<int8_t>(), host.data(), host.size());
  return tensor;
}

at::Tensor cpu_tensor(const std::vector<float> &host) {
  auto tensor = at::empty({static_cast<int64_t>(host.size())},
                          at::TensorOptions().dtype(at::kFloat));
  std::memcpy(tensor.data_ptr<float>(), host.data(), host.size() * sizeof(float));
  return tensor;
}

at::Tensor device_double_tensor(const std::vector<double> &host) {
  auto options = at::TensorOptions().dtype(at::kDouble).device(
      g_use_cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU));
  auto tensor = at::empty({static_cast<int64_t>(host.size())}, options);
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(tensor.data_ptr<double>(), host.data(),
                       host.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy double input");
  else
    std::memcpy(tensor.data_ptr<double>(), host.data(),
                host.size() * sizeof(double));
  return tensor;
}

at::Tensor cpu_double_tensor(const std::vector<double> &host) {
  auto tensor = at::empty({static_cast<int64_t>(host.size())},
                          at::TensorOptions().dtype(at::kDouble));
  std::memcpy(tensor.data_ptr<double>(), host.data(),
              host.size() * sizeof(double));
  return tensor;
}

int finish(const char *kernel, const char *recipe, const at::Tensor &output,
           const std::vector<float> &expected,
           const std::function<void()> &operation,
           const at::Tensor *aux_output = nullptr,
           const std::vector<float> *aux_expected = nullptr,
           const char *shape = "N=4194304") {
  operation();
  synchronize("correctness synchronization");
  for (int i = 0; i < 5; ++i)
    operation();
  synchronize("warmup synchronization");

  std::array<double, 5> samples{};
  for (double &sample : samples) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    synchronize("timed synchronization");
    const auto stop = std::chrono::steady_clock::now();
    sample =
        std::chrono::duration<double, std::micro>(stop - start).count();
  }

  std::vector<float> actual(expected.size());
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(actual.data(), output.data_ptr<float>(),
                       actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy output");
  else
    std::memcpy(actual.data(), output.data_ptr<float>(),
                actual.size() * sizeof(float));
  int errors = 0;
  float max_error = 0.0f;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float delta = std::fabs(expected[i] - actual[i]);
    max_error = std::max(max_error, delta);
    if (!std::isfinite(actual[i]) ||
        delta > 0.002f * (1.0f + std::fabs(expected[i])))
      ++errors;
  }
  if (aux_output && aux_expected) {
    std::vector<float> aux_actual(aux_expected->size());
    if (g_use_cuda)
      cuda_ok(cudaMemcpy(aux_actual.data(), aux_output->data_ptr<float>(),
                         aux_actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy auxiliary output");
    else
      std::memcpy(aux_actual.data(), aux_output->data_ptr<float>(),
                  aux_actual.size() * sizeof(float));
    for (size_t i = 0; i < aux_expected->size(); ++i) {
      const float delta = std::fabs((*aux_expected)[i] - aux_actual[i]);
      max_error = std::max(max_error, delta);
      if (!std::isfinite(aux_actual[i]) ||
          delta > 0.002f * (1.0f + std::fabs((*aux_expected)[i])))
        ++errors;
    }
  }

  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double q1 = 0.5 * (sorted[0] + sorted[1]);
  const double q3 = 0.5 * (sorted[3] + sorted[4]);
  const char *metric = g_use_cuda ? "native_aten_resident_wall_us"
                                  : "cpu_aten_1thread_wall_us";
  std::printf("SAMPLES kernel=%s %s="
              "%.6f,%.6f,%.6f,%.6f,%.6f\n",
              kernel, metric, samples[0], samples[1], samples[2], samples[3],
              samples[4]);
  std::printf("NATIVE_RESULT kernel=%s %s=%.6f "
              "min_us=%.6f max_us=%.6f iqr_us=%.6f errors=%d "
              "max_error=%g shape=%s recipe=%s\n",
              kernel, metric, sorted[2], sorted[0], sorted[4], q3 - q1, errors,
              max_error, shape, recipe);
  return errors == 0 ? 0 : 1;
}

int finish_int(const char *kernel, const char *recipe, const at::Tensor &output,
               const std::vector<int32_t> &expected,
               const std::function<void()> &operation, const char *shape) {
  operation();
  synchronize("integer correctness synchronization");
  for (int i = 0; i < 5; ++i)
    operation();
  synchronize("integer warmup synchronization");
  std::array<double, 5> samples{};
  for (double &sample : samples) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    synchronize("integer timed synchronization");
    const auto stop = std::chrono::steady_clock::now();
    sample = std::chrono::duration<double, std::micro>(stop - start).count();
  }
  std::vector<int32_t> actual(expected.size());
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(actual.data(), output.data_ptr<int32_t>(),
                       actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "copy integer output");
  else
    std::memcpy(actual.data(), output.data_ptr<int32_t>(),
                actual.size() * sizeof(int32_t));
  int errors = 0;
  int64_t max_error = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const int64_t delta = std::llabs(static_cast<int64_t>(expected[i]) -
                                    static_cast<int64_t>(actual[i]));
    max_error = std::max(max_error, delta);
    errors += delta != 0;
  }
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double q1 = 0.5 * (sorted[0] + sorted[1]);
  const double q3 = 0.5 * (sorted[3] + sorted[4]);
  const char *metric = g_use_cuda ? "native_aten_resident_wall_us"
                                  : "cpu_aten_1thread_wall_us";
  std::printf("SAMPLES kernel=%s %s="
              "%.6f,%.6f,%.6f,%.6f,%.6f\n",
              kernel, metric, samples[0], samples[1], samples[2], samples[3],
              samples[4]);
  std::printf("NATIVE_RESULT kernel=%s %s=%.6f "
              "min_us=%.6f max_us=%.6f iqr_us=%.6f errors=%d "
              "max_error=%lld shape=%s recipe=%s\n",
              kernel, metric, sorted[2], sorted[0], sorted[4], q3 - q1, errors,
              static_cast<long long>(max_error), shape, recipe);
  return errors == 0 ? 0 : 1;
}

int finish_double(const char *kernel, const char *recipe,
                  const at::Tensor &output,
                  const std::vector<double> &expected,
                  const std::function<void()> &operation,
                  const char *shape) {
  operation();
  synchronize("double correctness synchronization");
  for (int i = 0; i < 5; ++i) operation();
  synchronize("double warmup synchronization");
  std::array<double, 5> samples{};
  for (double &sample : samples) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    synchronize("double timed synchronization");
    const auto stop = std::chrono::steady_clock::now();
    sample = std::chrono::duration<double, std::micro>(stop - start).count();
  }
  std::vector<double> actual(expected.size());
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(actual.data(), output.data_ptr<double>(),
                       actual.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "copy double output");
  else
    std::memcpy(actual.data(), output.data_ptr<double>(),
                actual.size() * sizeof(double));
  int errors = 0;
  double max_error = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double delta = std::fabs(expected[i] - actual[i]);
    max_error = std::max(max_error, delta);
    if (!std::isfinite(actual[i]) ||
        delta > 1.0e-10 * (1.0 + std::fabs(expected[i])))
      ++errors;
  }
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double q1 = 0.5 * (sorted[0] + sorted[1]);
  const double q3 = 0.5 * (sorted[3] + sorted[4]);
  const char *metric = g_use_cuda ? "native_aten_resident_wall_us"
                                  : "cpu_aten_1thread_wall_us";
  std::printf("SAMPLES kernel=%s %s="
              "%.6f,%.6f,%.6f,%.6f,%.6f\n",
              kernel, metric, samples[0], samples[1], samples[2], samples[3],
              samples[4]);
  std::printf("NATIVE_RESULT kernel=%s %s=%.6f "
              "min_us=%.6f max_us=%.6f iqr_us=%.6f errors=%d "
              "max_error=%g shape=%s recipe=%s\n",
              kernel, metric, sorted[2], sorted[0], sorted[4], q3 - q1, errors,
              max_error, shape, recipe);
  return errors == 0 ? 0 : 1;
}

int finish_long(const char *kernel, const char *recipe,
                const at::Tensor &output,
                const std::vector<int64_t> &expected,
                const std::function<void()> &operation,
                const char *shape) {
  operation();
  synchronize("int64 correctness synchronization");
  for (int i = 0; i < 5; ++i) operation();
  synchronize("int64 warmup synchronization");
  std::array<double, 5> samples{};
  for (double &sample : samples) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    synchronize("int64 timed synchronization");
    const auto stop = std::chrono::steady_clock::now();
    sample = std::chrono::duration<double, std::micro>(stop - start).count();
  }
  std::vector<int64_t> actual(expected.size());
  if (g_use_cuda)
    cuda_ok(cudaMemcpy(actual.data(), output.data_ptr<int64_t>(),
                       actual.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
            "copy int64 output");
  else
    std::memcpy(actual.data(), output.data_ptr<int64_t>(),
                actual.size() * sizeof(int64_t));
  int errors = 0;
  int64_t max_error = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const int64_t delta = std::llabs(expected[i] - actual[i]);
    max_error = std::max(max_error, delta);
    errors += delta != 0;
  }
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double q1 = 0.5 * (sorted[0] + sorted[1]);
  const double q3 = 0.5 * (sorted[3] + sorted[4]);
  const char *metric = g_use_cuda ? "native_aten_resident_wall_us"
                                  : "cpu_aten_1thread_wall_us";
  std::printf("SAMPLES kernel=%s %s="
              "%.6f,%.6f,%.6f,%.6f,%.6f\n",
              kernel, metric, samples[0], samples[1], samples[2], samples[3],
              samples[4]);
  std::printf("NATIVE_RESULT kernel=%s %s=%.6f "
              "min_us=%.6f max_us=%.6f iqr_us=%.6f errors=%d "
              "max_error=%lld shape=%s recipe=%s\n",
              kernel, metric, sorted[2], sorted[0], sorted[4], q3 - q1, errors,
              static_cast<long long>(max_error), shape, recipe);
  return errors == 0 ? 0 : 1;
}

void apply_whole_unary(const std::string &kernel, at::Tensor &output,
                       const at::Tensor &input) {
  if (kernel == "aten_abs") at::abs_out(output, input);
  else if (kernel == "aten_acos") at::acos_out(output, input);
  else if (kernel == "aten_acosh") at::acosh_out(output, input);
  else if (kernel == "aten_asin") at::asin_out(output, input);
  else if (kernel == "aten_asinh") at::asinh_out(output, input);
  else if (kernel == "aten_atan") at::atan_out(output, input);
  else if (kernel == "aten_atanh") at::atanh_out(output, input);
  else if (kernel == "aten_ceil") at::ceil_out(output, input);
  else if (kernel == "aten_cos") at::cos_out(output, input);
  else if (kernel == "aten_cosh") at::cosh_out(output, input);
  else if (kernel == "aten_erf") at::erf_out(output, input);
  else if (kernel == "aten_erfc") at::erfc_out(output, input);
  else if (kernel == "aten_exp") at::exp_out(output, input);
  else if (kernel == "aten_exp2") at::exp2_out(output, input);
  else if (kernel == "aten_expm1") at::expm1_out(output, input);
  else if (kernel == "aten_floor") at::floor_out(output, input);
  else if (kernel == "aten_frac") at::frac_out(output, input);
  else if (kernel == "aten_log") at::log_out(output, input);
  else if (kernel == "aten_log10") at::log10_out(output, input);
  else if (kernel == "aten_log1p") at::log1p_out(output, input);
  else if (kernel == "aten_log2") at::log2_out(output, input);
  else if (kernel == "aten_neg") at::neg_out(output, input);
  else if (kernel == "aten_reciprocal") at::reciprocal_out(output, input);
  else if (kernel == "aten_relu") at::relu_out(output, input);
  else if (kernel == "aten_rsqrt") at::rsqrt_out(output, input);
  else if (kernel == "aten_sigmoid") at::sigmoid_out(output, input);
  else if (kernel == "aten_sin") at::sin_out(output, input);
  else if (kernel == "aten_sinh") at::sinh_out(output, input);
  else if (kernel == "aten_sqrt") at::sqrt_out(output, input);
  else if (kernel == "aten_square") at::square_out(output, input);
  else if (kernel == "aten_tan") at::tan_out(output, input);
  else if (kernel == "aten_tanh") at::tanh_out(output, input);
  else if (kernel == "aten_trunc") at::trunc_out(output, input);
  else throw std::runtime_error("unsupported whole unary ATen operation");
}

const char *whole_unary_recipe(const std::string &kernel) {
  if (kernel == "aten_abs") return "6f2fbd84283a7177";
  if (kernel == "aten_acos") return "85bb3f1c3c6b8549";
  if (kernel == "aten_acosh") return "ed8691f0d04228f3";
  if (kernel == "aten_asin") return "4a903a7e272d7c75";
  if (kernel == "aten_asinh") return "44b2f1000e079033";
  if (kernel == "aten_atan") return "66b99564041cb6eb";
  if (kernel == "aten_atanh") return "51c661d010e4b6de";
  if (kernel == "aten_ceil") return "99973374875367c6";
  if (kernel == "aten_cos") return "bb94742dcdc246da";
  if (kernel == "aten_cosh") return "d7e37dcedb3b0e48";
  if (kernel == "aten_erf") return "0dd4b2733f8227cd";
  if (kernel == "aten_erfc") return "93a3de32a1e78949";
  if (kernel == "aten_exp") return "5b828abb80778add";
  if (kernel == "aten_exp2") return "a512ae97ce832704";
  if (kernel == "aten_expm1") return "b76cfa17275d9f7a";
  if (kernel == "aten_floor") return "ee994c992feabd3f";
  if (kernel == "aten_frac") return "257b2d03a81cc271";
  if (kernel == "aten_log") return "3de8004fa500429e";
  if (kernel == "aten_log10") return "16be5501c4b754a0";
  if (kernel == "aten_log1p") return "e84c88cd8e52d9d7";
  if (kernel == "aten_log2") return "bb9bfb2fc1e8317e";
  if (kernel == "aten_neg") return "e1c9b3016ab44715";
  if (kernel == "aten_reciprocal") return "21e034c59660aef0";
  if (kernel == "aten_relu") return "2b98a38943814af4";
  if (kernel == "aten_rsqrt") return "698b8433ba6b0c53";
  if (kernel == "aten_sigmoid") return "3aa4c7f67607ce07";
  if (kernel == "aten_sin") return "c33c39c26b3496c2";
  if (kernel == "aten_sinh") return "4314829041c59873";
  if (kernel == "aten_sqrt") return "0cede42120fd1ffe";
  if (kernel == "aten_square") return "d6e39dc56b180446";
  if (kernel == "aten_tan") return "1ea007f2c8d24860";
  if (kernel == "aten_tanh") return "4d4284953d7c9011";
  if (kernel == "aten_trunc") return "f7864fefa35b9eb9";
  return "UNVERSIONED";
}

bool is_whole_unary(const std::string &kernel) {
  static const std::vector<std::string> names = {
      "aten_abs", "aten_acos", "aten_acosh", "aten_asin", "aten_asinh",
      "aten_atan", "aten_atanh", "aten_ceil", "aten_cos", "aten_cosh",
      "aten_erf", "aten_erfc", "aten_exp", "aten_exp2", "aten_expm1",
      "aten_floor", "aten_frac", "aten_log", "aten_log10", "aten_log1p",
      "aten_log2", "aten_neg", "aten_reciprocal", "aten_relu",
      "aten_rsqrt", "aten_sigmoid", "aten_sin", "aten_sinh", "aten_sqrt",
      "aten_square", "aten_tan", "aten_tanh", "aten_trunc"};
  return std::find(names.begin(), names.end(), kernel) != names.end();
}

int whole_unary(const std::string &kernel) {
  std::vector<float> host(kPointwiseElements);
  const bool acosh_domain = kernel == "aten_acosh";
  const bool positive = kernel == "aten_log" || kernel == "aten_log10" ||
                        kernel == "aten_log1p" || kernel == "aten_log2" ||
                        kernel == "aten_sqrt" || kernel == "aten_rsqrt" ||
                        kernel == "aten_reciprocal";
  const bool unit = kernel == "aten_acos" || kernel == "aten_asin" ||
                    kernel == "aten_atanh";
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    if (acosh_domain)
      host[i] = 1.25f + static_cast<float>(i % 101) / 101.0f;
    else if (positive)
      host[i] = 0.25f + static_cast<float>(i % 101) / 101.0f;
    else if (unit)
      host[i] = 0.05f + 0.9f * static_cast<float>(i % 101) / 101.0f;
    else
      host[i] = (static_cast<float>(i % 101) - 50.0f) / 37.0f;
  }
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  auto cpu_input = cpu_tensor(host);
  auto cpu_output = at::empty_like(cpu_input);
  apply_whole_unary(kernel, cpu_output, cpu_input);
  std::vector<float> expected(kPointwiseElements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply_whole_unary(kernel, output, input); };
  return finish(kernel.c_str(), whole_unary_recipe(kernel), output, expected,
                operation);
}

void apply_whole_elementwise(const std::string &kernel, at::Tensor &output,
                             const at::Tensor &a, const at::Tensor &b,
                             const at::Tensor &c) {
  if (kernel == "aten_add") at::add_out(output, a, b, 0.75);
  else if (kernel == "aten_addcdiv") at::addcdiv_out(output, a, b, c, 0.75);
  else if (kernel == "aten_addcmul") at::addcmul_out(output, a, b, c, 0.75);
  else if (kernel == "aten_atan2") at::atan2_out(output, a, b);
  else if (kernel == "aten_div") at::div_out(output, a, b);
  else if (kernel == "aten_fmax") at::fmax_out(output, a, b);
  else if (kernel == "aten_fmin") at::fmin_out(output, a, b);
  else if (kernel == "aten_fmod") at::fmod_out(output, a, b);
  else if (kernel == "aten_hypot") at::hypot_out(output, a, b);
  else if (kernel == "aten_lerp_tensor_cpu") at::lerp_out(output, a, b, c);
  else if (kernel == "aten_maximum") at::maximum_out(output, a, b);
  else if (kernel == "aten_minimum") at::minimum_out(output, a, b);
  else if (kernel == "aten_mul") at::mul_out(output, a, b);
  else if (kernel == "aten_pow") at::pow_out(output, a, b);
  else if (kernel == "aten_pow_tensor_scalar") at::pow_out(output, a, 1.75);
  else throw std::runtime_error("unsupported whole elementwise ATen operation");
}

const char *whole_elementwise_recipe(const std::string &kernel) {
  if (kernel == "aten_add") return "3f848f6bbc668548";
  if (kernel == "aten_addcdiv") return "edb482d5b31abdc6";
  if (kernel == "aten_addcmul") return "28105c017f81cc7d";
  if (kernel == "aten_atan2") return "8397dd293ff562c5";
  if (kernel == "aten_div") return "3a958dcfd03cdc81";
  if (kernel == "aten_fmax") return "01b6a5c5d48be296";
  if (kernel == "aten_fmin") return "cae427b65388ace8";
  if (kernel == "aten_fmod") return "ebf144712f447640";
  if (kernel == "aten_hypot") return "360eac1f010bdb44";
  if (kernel == "aten_lerp_tensor_cpu") return "1cfee9a904ffc548";
  if (kernel == "aten_maximum") return "20b4059e81d20c1b";
  if (kernel == "aten_minimum") return "3f5f73c24f2d1d36";
  if (kernel == "aten_mul") return "9d7b8aeafb805ec3";
  if (kernel == "aten_pow") return "d841f8307bd2c5d4";
  if (kernel == "aten_pow_tensor_scalar") return "23ca8e68b86e3edf";
  return "UNVERSIONED";
}

bool is_whole_elementwise(const std::string &kernel) {
  static const std::vector<std::string> names = {
      "aten_add", "aten_addcdiv", "aten_addcmul", "aten_atan2", "aten_div",
      "aten_fmax", "aten_fmin", "aten_fmod", "aten_hypot",
      "aten_lerp_tensor_cpu", "aten_maximum", "aten_minimum", "aten_mul",
      "aten_pow", "aten_pow_tensor_scalar"};
  return std::find(names.begin(), names.end(), kernel) != names.end();
}

int whole_elementwise(const std::string &kernel) {
  const int64_t n = kernel == "aten_add" ? 11LL * 46 * 92 * 92
                                         : kPointwiseElements;
  std::vector<float> a_host(n), b_host(n), c_host(n);
  const bool both_positive = kernel == "aten_fmod" || kernel == "aten_pow";
  for (int64_t i = 0; i < n; ++i) {
    const float normal = (static_cast<float>(i % 101) - 50.0f) / 37.0f;
    const float positive = 0.25f + static_cast<float>(i % 101) / 101.0f;
    a_host[i] = (both_positive || kernel == "aten_pow_tensor_scalar")
                    ? positive : normal;
    b_host[i] = (both_positive || kernel == "aten_div") ? positive : normal;
    c_host[i] = kernel == "aten_lerp_tensor_cpu"
                    ? 0.05f + 0.9f * static_cast<float>(i % 101) / 101.0f
                    : kernel == "aten_addcdiv" ? positive : normal;
  }
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto c = device_tensor(c_host);
  auto output = at::empty_like(a);
  auto cpu_a = cpu_tensor(a_host);
  auto cpu_b = cpu_tensor(b_host);
  auto cpu_c = cpu_tensor(c_host);
  auto cpu_output = at::empty_like(cpu_a);
  apply_whole_elementwise(kernel, cpu_output, cpu_a, cpu_b, cpu_c);
  std::vector<float> expected(n);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply_whole_elementwise(kernel, output, a, b, c); };
  const char *shape = kernel == "aten_add" ? "B=11_C=46_H=92_W=92"
                                            : "N=4194304";
  return finish(kernel.c_str(), whole_elementwise_recipe(kernel), output,
                expected, operation, nullptr, nullptr, shape);
}

void apply_whole_activation(const std::string &kernel, at::Tensor &output,
                            const at::Tensor &input) {
  if (kernel == "aten_gelu" || kernel == "aten_gelu_cpu_exact")
    at::gelu_out(output, input, "none");
  else if (kernel == "aten_gelu_cpu_tanh")
    at::gelu_out(output, input, "tanh");
  else if (kernel == "aten_hardsigmoid") at::hardsigmoid_out(output, input);
  else if (kernel == "aten_hardswish") at::hardswish_out(output, input);
  else if (kernel == "aten_leaky_relu") at::leaky_relu_out(output, input, 0.01);
  else if (kernel == "aten_mish") at::mish_out(output, input);
  else if (kernel == "aten_silu" || kernel == "aten_silu_cpu")
    at::silu_out(output, input);
  else if (kernel == "aten_softplus") at::softplus_out(output, input, 1.0, 20.0);
  else throw std::runtime_error("unsupported whole activation ATen operation");
}

const char *whole_activation_recipe(const std::string &kernel) {
  if (kernel == "aten_gelu") return "f457f641f8eed3db";
  if (kernel == "aten_gelu_cpu_exact") return "5b48840c70c886ab";
  if (kernel == "aten_gelu_cpu_tanh") return "ddbb97ad9ec4d06e";
  if (kernel == "aten_hardsigmoid") return "edda58d945a51022";
  if (kernel == "aten_hardswish") return "a69c9eae5e6108ed";
  if (kernel == "aten_leaky_relu") return "f81c45745cd970f2";
  if (kernel == "aten_mish") return "ed7cd796ed121579";
  if (kernel == "aten_silu") return "71d6c728c03d169b";
  if (kernel == "aten_silu_cpu") return "1fff634b60e4f382";
  if (kernel == "aten_softplus") return "57b1e51c04e2b589";
  return "UNVERSIONED";
}

bool is_whole_activation(const std::string &kernel) {
  static const std::vector<std::string> names = {
      "aten_gelu", "aten_gelu_cpu_exact", "aten_gelu_cpu_tanh",
      "aten_hardsigmoid", "aten_hardswish", "aten_leaky_relu", "aten_mish",
      "aten_silu", "aten_silu_cpu", "aten_softplus"};
  return std::find(names.begin(), names.end(), kernel) != names.end();
}

int whole_activation(const std::string &kernel) {
  const int64_t n = kernel == "aten_softplus" ? 8388608 : kPointwiseElements;
  std::vector<float> host(n);
  for (int64_t i = 0; i < n; ++i)
    host[i] = (static_cast<float>(i % 101) - 50.0f) / 37.0f;
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  auto cpu_input = cpu_tensor(host);
  auto cpu_output = at::empty_like(cpu_input);
  apply_whole_activation(kernel, cpu_output, cpu_input);
  std::vector<float> expected(n);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply_whole_activation(kernel, output, input); };
  const char *shape = kernel == "aten_softplus" ? "N=8388608" : "N=4194304";
  return finish(kernel.c_str(), whole_activation_recipe(kernel), output,
                expected, operation, nullptr, nullptr, shape);
}

void apply_whole_activation_backward(const std::string &kernel,
                                     at::Tensor &output,
                                     const at::Tensor &grad,
                                     const at::Tensor &argument) {
  if (kernel == "aten_gelu_backward_cpu_exact")
    at::gelu_backward_out(output, grad, argument, "none");
  else if (kernel == "aten_gelu_backward_cpu_tanh")
    at::gelu_backward_out(output, grad, argument, "tanh");
  else if (kernel == "aten_hardsigmoid_backward")
    at::hardsigmoid_backward_out(output, grad, argument);
  else if (kernel == "aten_sigmoid_backward")
    at::sigmoid_backward_out(output, grad, argument);
  else if (kernel == "aten_silu_backward")
    at::silu_backward_out(output, grad, argument);
  else if (kernel == "aten_softplus_backward")
    at::softplus_backward_out(output, grad, argument, 1.1, 0.7);
  else if (kernel == "aten_tanh_backward")
    at::tanh_backward_out(output, grad, argument);
  else throw std::runtime_error("unsupported activation backward ATen operation");
}

const char *whole_activation_backward_recipe(const std::string &kernel) {
  if (kernel == "aten_gelu_backward_cpu_exact") return "71eee55ef62813bd";
  if (kernel == "aten_gelu_backward_cpu_tanh") return "df5b1482f040c411";
  if (kernel == "aten_hardsigmoid_backward") return "4c76ee8a69d656ec";
  if (kernel == "aten_sigmoid_backward") return "1ea31fde0432d020";
  if (kernel == "aten_silu_backward") return "0f3ebf276e2d7c0b";
  if (kernel == "aten_softplus_backward") return "77ddf49c10f48c36";
  if (kernel == "aten_tanh_backward") return "4a722ac00ff4eeac";
  return "UNVERSIONED";
}

bool is_whole_activation_backward(const std::string &kernel) {
  static const std::vector<std::string> names = {
      "aten_gelu_backward_cpu_exact", "aten_gelu_backward_cpu_tanh",
      "aten_hardsigmoid_backward", "aten_sigmoid_backward",
      "aten_silu_backward", "aten_softplus_backward", "aten_tanh_backward"};
  return std::find(names.begin(), names.end(), kernel) != names.end();
}

int whole_activation_backward(const std::string &kernel) {
  std::vector<float> grad_host(kPointwiseElements);
  std::vector<float> argument_host(kPointwiseElements);
  const bool unit_output = kernel == "aten_sigmoid_backward" ||
                           kernel == "aten_tanh_backward";
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    grad_host[i] = (static_cast<float>(i % 101) - 50.0f) / 37.0f;
    argument_host[i] = unit_output
        ? 0.05f + 0.9f * static_cast<float>(i % 101) / 101.0f
        : (static_cast<float>((i + 29) % 101) - 50.0f) / 37.0f;
  }
  auto grad = device_tensor(grad_host);
  auto argument = device_tensor(argument_host);
  auto output = at::empty_like(grad);
  auto cpu_grad = cpu_tensor(grad_host);
  auto cpu_argument = cpu_tensor(argument_host);
  auto cpu_output = at::empty_like(cpu_grad);
  apply_whole_activation_backward(kernel, cpu_output, cpu_grad, cpu_argument);
  std::vector<float> expected(kPointwiseElements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    apply_whole_activation_backward(kernel, output, grad, argument);
  };
  return finish(kernel.c_str(), whole_activation_backward_recipe(kernel),
                output, expected, operation);
}

int whole_hardswish_or_mish_backward(const std::string &kernel) {
  auto grad_host = raised_fixture_values(kPointwiseElements);
  auto argument_host = raised_fixture_values(kPointwiseElements);
  std::rotate(argument_host.begin(), argument_host.begin() + 29,
              argument_host.end());
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    const float x = argument_host[i];
    if (kernel == "aten_hardswish_backward") {
      expected[i] = x <= -3.0f
          ? 0.0f
          : (x < 3.0f ? grad_host[i] * (x / 3.0f + 0.5f)
                      : grad_host[i]);
    } else {
      const float sigmoid = 1.0f / (1.0f + std::exp(-x));
      const float tanh_softplus = std::tanh(std::log1p(std::exp(x)));
      expected[i] = grad_host[i] *
          (tanh_softplus + x * sigmoid *
              (1.0f - tanh_softplus * tanh_softplus));
    }
  }
  auto grad = device_tensor(grad_host);
  auto argument = device_tensor(argument_host);
  auto output = at::empty_like(grad);
  std::function<void()> operation;
  std::optional<at::TensorIterator> iterator;
  const char *recipe = nullptr;
  if (kernel == "aten_hardswish_backward") {
    operation = [&] { at::hardswish_backward_out(output, grad, argument); };
    recipe = "6d9d2fe503529507";
  } else {
    // mish_backward has no generated out variant in this ATen revision.  Its
    // implementation is a registered TensorIterator CUDA kernel, so prepare
    // the iterator once and invoke that existing ATen kernel in the timed
    // resident region without allocating an output there.
    iterator.emplace(at::TensorIterator::binary_op(output, grad, argument));
    operation = [&] {
      at::native::mish_backward_stub(iterator->device_type(), *iterator);
    };
    recipe = "70bf6cbf89d9ad97";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int whole_copy_or_zero(const std::string &kernel) {
  const int64_t n = kPointwiseElements;
  auto host = raised_fixture_values(n);
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  std::vector<float> expected = host;
  std::function<void()> operation;
  const char *recipe = nullptr;
  const char *shape = "N=4194304";
  if (kernel == "aten_zeros_cpu") {
    std::fill(expected.begin(), expected.end(), 0.0f);
    operation = [&] { output.zero_(); };
    recipe = "8396a59034c783b2";
  } else {
    operation = [&] { output.copy_(input); };
    if (kernel == "aten_copy_cpu") recipe = "bfde07a5167fb20d";
    else {
      recipe = "763d2776dbad0967";
      shape = "B=512_N=8192";
    }
  }
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_transpose_copy() {
  constexpr int64_t rows = 2365;
  constexpr int64_t columns = 1774;
  auto host = raised_fixture_values(rows * columns);
  auto input = device_tensor(host).reshape({rows, columns});
  auto output = at::empty({columns, rows}, input.options());
  auto cpu_input = cpu_tensor(host).reshape({rows, columns});
  auto cpu_output = at::empty({columns, rows}, cpu_input.options());
  at::transpose_copy_out(cpu_output, cpu_input, 0, 1);
  std::vector<float> expected(host.size());
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::transpose_copy_out(output, input, 0, 1); };
  return finish("aten_transpose_copy", "5d0809aea12f7880", output, expected,
                operation, nullptr, nullptr, "M=2365_N=1774");
}

int whole_pixel_transform(const std::string &kernel) {
  const bool unshuffle = kernel == "aten_pixel_unshuffle_cpu_backend";
  const int64_t batch = kernel == "aten_pixel_shuffle" ? 9 : 4;
  const int64_t channels = kernel == "aten_pixel_shuffle" ? 14 : 13;
  const int64_t height = kernel == "aten_pixel_shuffle" ? 19 : 34;
  const int64_t width = height;
  constexpr int64_t factor = 9;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  if (unshuffle) {
    input_shape = {batch, channels, height * factor, width * factor};
    output_shape = {batch, channels * factor * factor, height, width};
  } else {
    input_shape = {batch, channels * factor * factor, height, width};
    output_shape = {batch, channels, height * factor, width * factor};
  }
  int64_t n = 1;
  for (int64_t extent : input_shape) n *= extent;
  auto host = raised_fixture_values(n);
  auto input = device_tensor(host).reshape(input_shape);
  auto output = at::empty(output_shape, input.options());
  auto cpu_input = cpu_tensor(host).reshape(input_shape);
  auto cpu_output = at::empty(output_shape, cpu_input.options());
  if (unshuffle)
    at::pixel_unshuffle_out(cpu_output, cpu_input, factor);
  else
    at::pixel_shuffle_out(cpu_output, cpu_input, factor);
  std::vector<float> expected(n);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    if (unshuffle) at::pixel_unshuffle_out(output, input, factor);
    else at::pixel_shuffle_out(output, input, factor);
  };
  const char *recipe = kernel == "aten_pixel_shuffle" ? "387681cd9544c530" :
      unshuffle ? "30ac9ab652691610" : "72310b3d9814d0ad";
  const char *shape = kernel == "aten_pixel_shuffle" ?
      "B=9_C=14_H=19_R=9_W=19" :
      unshuffle ? "B=4_C=13_H=34_RATIO=9_W=34" :
                  "B=4_C=13_H=34_RATIO=9_W=34";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_special_pointwise(const std::string &kernel) {
  std::vector<float> a_host(kPointwiseElements), b_host(kPointwiseElements);
  const bool small = kernel == "aten_logaddexp" || kernel == "aten_logaddexp2";
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    const float value = small
        ? (static_cast<float>(i % 101) - 50.0f) / 100.0f
        : (static_cast<float>(i % 101) - 50.0f) / 37.0f;
    a_host[i] = value;
    b_host[i] = value;
  }
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto output = at::empty_like(a);
  auto cpu_a = cpu_tensor(a_host);
  auto cpu_b = cpu_tensor(b_host);
  auto cpu_output = at::empty_like(cpu_a);
  auto apply = [&](at::Tensor &out, const at::Tensor &x, const at::Tensor &y) {
    if (kernel == "aten_angle_real") at::angle_out(out, x);
    else if (kernel == "aten_clamp") at::clamp_out(out, x, -0.5, 0.75);
    else if (kernel == "aten_logaddexp") at::logaddexp_out(out, x, y);
    else if (kernel == "aten_logaddexp2") at::logaddexp2_out(out, x, y);
    else if (kernel == "aten_mse_elementwise") at::mse_loss_out(out, x, y, 0);
  };
  apply(cpu_output, cpu_a, cpu_b);
  std::vector<float> expected(kPointwiseElements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply(output, a, b); };
  const char *recipe = kernel == "aten_angle_real" ? "9f7716be3c155ab8" :
      kernel == "aten_clamp" ? "732ae21a30b24a68" :
      kernel == "aten_logaddexp" ? "f8780f262412130a" :
      kernel == "aten_logaddexp2" ? "42ce8060fb0f98d2" :
      "4fddb01a5fe09064";
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int whole_nansum() {
  constexpr int64_t rows = 131072;
  constexpr int64_t columns = 64;
  std::vector<float> host(rows * columns);
  for (int64_t i = 0; i < rows * columns; ++i)
    host[i] = ((i / columns) % 4 == 0 || i % columns == 7 ||
               i % columns == 19)
                  ? NAN
                  : static_cast<float>(static_cast<int>(i % 101) - 50) / 37.0f;
  auto input = device_tensor(host).reshape({rows, columns});
  auto output = at::empty({rows}, input.options());
  auto cpu_input = cpu_tensor(host).reshape({rows, columns});
  auto cpu_output = at::empty({rows}, cpu_input.options());
  at::nansum_out(cpu_output, cpu_input, at::OptionalIntArrayRef({1}), false,
                 at::kFloat);
  std::vector<float> expected(rows);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    at::nansum_out(output, input, at::OptionalIntArrayRef({1}), false,
                   at::kFloat);
  };
  return finish("aten_nansum_cpu", "0085284cb82ca0db", output, expected,
                operation, nullptr, nullptr, "R=131072_K=64");
}

int whole_average_pool(const std::string &kernel) {
  const bool two_dimensional = kernel == "aten_avg_pool2d" ||
                               kernel == "aten_adaptive_avg_pool2d";
  const bool adaptive = kernel == "aten_adaptive_avg_pool2d" ||
                        kernel == "aten_adaptive_avg_pool3d";
  const int64_t batch = kernel == "aten_adaptive_avg_pool2d" ? 4 : 2;
  const int64_t channels = kernel == "aten_adaptive_avg_pool2d" ? 32 :
                           kernel == "aten_avg_pool2d" ? 4 : 3;
  const int64_t height = kernel == "aten_adaptive_avg_pool2d" ? 256 :
                         kernel == "aten_avg_pool2d" ? 16 : 8;
  const int64_t width = height;
  const int64_t depth = 8;
  const int64_t out_height = kernel == "aten_adaptive_avg_pool2d" ? 128 :
                             height / 2;
  const int64_t out_width = out_height;
  const int64_t out_depth = 4;
  std::vector<int64_t> input_shape = two_dimensional
      ? std::vector<int64_t>{batch, channels, height, width}
      : std::vector<int64_t>{batch, channels, depth, height, width};
  std::vector<int64_t> output_shape = two_dimensional
      ? std::vector<int64_t>{batch, channels, out_height, out_width}
      : std::vector<int64_t>{batch, channels, out_depth, out_height, out_width};
  int64_t input_elements = 1;
  int64_t output_elements = 1;
  for (int64_t extent : input_shape) input_elements *= extent;
  for (int64_t extent : output_shape) output_elements *= extent;
  auto host = raised_fixture_values(input_elements);
  auto input = device_tensor(host).reshape(input_shape);
  auto output = at::empty(output_shape, input.options());
  auto cpu_input = cpu_tensor(host).reshape(input_shape);
  auto cpu_output = at::empty(output_shape, cpu_input.options());
  auto apply = [&](at::Tensor &out, const at::Tensor &in) {
    if (two_dimensional && adaptive)
      at::adaptive_avg_pool2d_out(out, in, {out_height, out_width});
    else if (!two_dimensional && adaptive)
      at::adaptive_avg_pool3d_out(out, in,
                                  {out_depth, out_height, out_width});
    else if (two_dimensional)
      at::avg_pool2d_out(out, in, {2, 2}, {2, 2}, {0, 0}, false, true,
                         std::nullopt);
    else
      at::avg_pool3d_out(out, in, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, false,
                         true, std::nullopt);
  };
  apply(cpu_output, cpu_input);
  std::vector<float> expected(output_elements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply(output, input); };
  const char *recipe = kernel == "aten_adaptive_avg_pool2d" ?
      "1a91e610dab8183b" : kernel == "aten_adaptive_avg_pool3d" ?
      "b3c8e383067677ef" : kernel == "aten_avg_pool2d" ?
      "f8994b10d0786ea2" : "64727eb4df598924";
  const char *shape = kernel == "aten_adaptive_avg_pool2d" ?
      "B=4_C=32_H=256_W=256_OH=128_OW=128" :
      kernel == "aten_adaptive_avg_pool3d" ? "B=2_C=3_D=8_H=8_W=8" :
      kernel == "aten_avg_pool2d" ? "B=2_C=4_H=16_W=16" :
                                    "B=2_C=3_D=8_H=8_W=8";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_average_pool_backward(const std::string &kernel) {
  const bool three_dimensional = kernel == "aten_avg_pool3d_backward_cpu";
  constexpr int64_t batch = 1;
  constexpr int64_t channels = 2;
  constexpr int64_t d = 6;
  constexpr int64_t h = 7;
  constexpr int64_t w = 8;
  std::vector<int64_t> input_shape = three_dimensional
      ? std::vector<int64_t>{batch, channels, d, h, w}
      : std::vector<int64_t>{batch, channels, d, h};
  std::vector<int64_t> grad_shape = three_dimensional
      ? std::vector<int64_t>{batch, channels, d / 2, h / 2, w / 2}
      : std::vector<int64_t>{batch, channels, d / 2, h / 2};
  int64_t input_elements = 1;
  int64_t grad_elements = 1;
  for (int64_t extent : input_shape) input_elements *= extent;
  for (int64_t extent : grad_shape) grad_elements *= extent;
  auto grad_host = raised_fixture_values(grad_elements);
  std::vector<float> input_host(input_elements, 0.0f);
  auto grad = device_tensor(grad_host).reshape(grad_shape);
  auto input = device_tensor(input_host).reshape(input_shape);
  auto output = at::empty(input_shape, input.options());
  auto cpu_grad = cpu_tensor(grad_host).reshape(grad_shape);
  auto cpu_input = cpu_tensor(input_host).reshape(input_shape);
  auto cpu_output = at::empty(input_shape, cpu_input.options());
  auto apply = [&](at::Tensor &out, const at::Tensor &g, const at::Tensor &in) {
    if (three_dimensional)
      at::avg_pool3d_backward_out(out, g, in, {2, 2, 2}, {2, 2, 2},
                                   {0, 0, 0}, false, true, std::nullopt);
    else
      at::avg_pool2d_backward_out(out, g, in, {2, 2}, {2, 2}, {0, 0},
                                   false, true, std::nullopt);
  };
  apply(cpu_output, cpu_grad, cpu_input);
  std::vector<float> expected(input_elements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply(output, grad, input); };
  const char *recipe = three_dimensional ? "5be05477d0f38b1d" :
                                           "871518045875f207";
  const char *shape = three_dimensional ? "B=1_C=2_I0=6_I1=7_I2=8" :
                                          "B=1_C=2_I0=6_I1=7";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_bmm() {
  constexpr int64_t batch = 70;
  constexpr int64_t m = 141;
  constexpr int64_t n = 211;
  constexpr int64_t k = 282;
  auto a_host = raised_fixture_values(batch * m * k);
  auto b_host = raised_fixture_values(batch * k * n);
  auto a = device_tensor(a_host).reshape({batch, m, k});
  auto b = device_tensor(b_host).reshape({batch, k, n});
  auto output = at::empty({batch, m, n}, a.options());
  auto cpu_a = cpu_tensor(a_host).reshape({batch, m, k});
  auto cpu_b = cpu_tensor(b_host).reshape({batch, k, n});
  auto cpu_output = at::empty({batch, m, n}, cpu_a.options());
  at::bmm_out(cpu_output, cpu_a, cpu_b);
  std::vector<float> expected(batch * m * n);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::bmm_out(output, a, b); };
  return finish("aten_bmm", "c132d6acaaae7ca9", output, expected, operation,
                nullptr, nullptr, "BATCH=70_K=282_M=141_N=211");
}

int whole_cross(const std::string &kernel) {
  constexpr int64_t vectors = 1398101;
  auto a_host = raised_fixture_values(vectors * 3);
  auto b_host = raised_fixture_values(vectors * 3);
  auto a = device_tensor(a_host).reshape({vectors, 3});
  auto b = device_tensor(b_host).reshape({vectors, 3});
  auto output = at::empty_like(a);
  auto cpu_a = cpu_tensor(a_host).reshape({vectors, 3});
  auto cpu_b = cpu_tensor(b_host).reshape({vectors, 3});
  auto cpu_output = at::empty_like(cpu_a);
  at::linalg_cross_out(cpu_output, cpu_a, cpu_b, -1);
  std::vector<float> expected(vectors * 3);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::linalg_cross_out(output, a, b, -1); };
  const char *recipe = kernel == "aten_cross" ? "b42870c4d04b959c" :
                                                "9291ba8d1ebe364f";
  const char *shape = kernel == "aten_cross" ? "N=1398101" : "V=1398101";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_cat() {
  constexpr int64_t rows_a = 774;
  constexpr int64_t rows_b = 581;
  constexpr int64_t columns = 3096;
  auto a_host = raised_fixture_values(rows_a * columns);
  auto b_host = raised_fixture_values(rows_b * columns);
  auto a = device_tensor(a_host).reshape({rows_a, columns});
  auto b = device_tensor(b_host).reshape({rows_b, columns});
  auto output = at::empty({rows_a + rows_b, columns}, a.options());
  auto cpu_a = cpu_tensor(a_host).reshape({rows_a, columns});
  auto cpu_b = cpu_tensor(b_host).reshape({rows_b, columns});
  auto cpu_output = at::empty({rows_a + rows_b, columns}, cpu_a.options());
  std::vector<at::Tensor> inputs{a, b};
  std::vector<at::Tensor> cpu_inputs{cpu_a, cpu_b};
  at::cat_out(cpu_output, cpu_inputs, 0);
  std::vector<float> expected((rows_a + rows_b) * columns);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::cat_out(output, inputs, 0); };
  return finish("aten_cat_serial_cpu", "8969bf611c743094", output, expected,
                operation, nullptr, nullptr, "K=3096_M=581_R=774");
}

int whole_repeat(const std::string &kernel) {
  constexpr int64_t n = 8192;
  constexpr int64_t repeats = 512;
  auto host = raised_fixture_values(n);
  auto input = device_tensor(host);
  auto output = at::empty({repeats, n}, input.options());
  auto cpu_input = cpu_tensor(host);
  auto cpu_output = at::empty({repeats, n}, cpu_input.options());
  at::repeat_out(cpu_output, cpu_input, {repeats, 1});
  std::vector<float> expected(repeats * n);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::repeat_out(output, input, {repeats, 1}); };
  const char *recipe = kernel == "aten_repeat_compute_cpu" ?
      "8de495d9b009aa55" : "c950974bab2c20a2";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, "N=8192_R=512");
}

int whole_im2col() {
  constexpr int64_t batch = 4;
  constexpr int64_t channels = 8;
  constexpr int64_t height = 128;
  constexpr int64_t width = 128;
  constexpr int64_t kernel = 3;
  constexpr int64_t output_height = height - kernel + 1;
  constexpr int64_t output_width = width - kernel + 1;
  auto host = raised_fixture_values(batch * channels * height * width);
  auto input = device_tensor(host).reshape({batch, channels, height, width});
  auto output = at::empty(
      {batch, channels * kernel * kernel, output_height * output_width},
      input.options());
  auto cpu_input = cpu_tensor(host).reshape({batch, channels, height, width});
  auto cpu_output = at::empty(output.sizes(), cpu_input.options());
  at::im2col_out(cpu_output, cpu_input, {kernel, kernel}, {1, 1}, {0, 0},
                 {1, 1});
  std::vector<float> expected(output.numel());
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    at::im2col_out(output, input, {kernel, kernel}, {1, 1}, {0, 0}, {1, 1});
  };
  return finish("aten_im2col", "a4c2fa550faa122e", output, expected,
                operation, nullptr, nullptr,
                "B=4_C=8_H=128_W=128_KH=3_KW=3");
}

int whole_max_pool2d() {
  constexpr int64_t batch = 16;
  constexpr int64_t channels = 32;
  constexpr int64_t height = 128;
  constexpr int64_t width = 128;
  constexpr int64_t output_height = 64;
  constexpr int64_t output_width = 64;
  auto host = raised_fixture_values(batch * channels * height * width);
  auto input = device_tensor(host).reshape({batch, channels, height, width});
  auto output = at::empty(
      {batch, channels, output_height, output_width}, input.options());
  auto indices = at::empty(output.sizes(), input.options().dtype(at::kLong));
  auto cpu_input = cpu_tensor(host).reshape({batch, channels, height, width});
  auto cpu_output = at::empty(output.sizes(), cpu_input.options());
  auto cpu_indices = at::empty(output.sizes(), cpu_input.options().dtype(at::kLong));
  at::max_pool2d_with_indices_out(cpu_output, cpu_indices, cpu_input, {2, 2},
                                  {2, 2}, {0, 0}, {1, 1}, false);
  std::vector<float> expected(output.numel());
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    at::max_pool2d_with_indices_out(output, indices, input, {2, 2}, {2, 2},
                                    {0, 0}, {1, 1}, false);
  };
  return finish("aten_max_pool2d", "a2ade0d34a8c60b6", output, expected,
                operation, nullptr, nullptr, "B=16_C=32_H=128_W=128_K=2_S=2");
}

std::vector<double> raised_double_values(int64_t n) {
  std::vector<double> values(n);
  for (int64_t i = 0; i < n; ++i)
    values[i] = (static_cast<double>(i % 101) - 50.0) / 37.0;
  return values;
}

int whole_double_linear_algebra(const std::string &kernel) {
  if (kernel == "aten_outer") {
    constexpr int64_t m = 2048;
    constexpr int64_t n = 2048;
    auto x_host = raised_double_values(m);
    auto y_host = raised_double_values(n);
    auto x = device_double_tensor(x_host);
    auto y = device_double_tensor(y_host);
    auto output = at::empty({m, n}, x.options());
    auto cpu_x = cpu_double_tensor(x_host);
    auto cpu_y = cpu_double_tensor(y_host);
    auto cpu_output = at::empty({m, n}, cpu_x.options());
    at::outer_out(cpu_output, cpu_x, cpu_y);
    std::vector<double> expected(m * n);
    std::memcpy(expected.data(), cpu_output.data_ptr<double>(),
                expected.size() * sizeof(double));
    auto operation = [&] { at::outer_out(output, x, y); };
    return finish_double("aten_outer", "090827df12946d14", output, expected,
                         operation, "M=2048_N=2048");
  }
  if (kernel == "aten_sum") {
    constexpr int64_t m = 65536;
    constexpr int64_t n = 64;
    auto host = raised_double_values(m * n);
    auto input = device_double_tensor(host).reshape({m, n});
    auto output = at::empty({m}, input.options());
    auto cpu_input = cpu_double_tensor(host).reshape({m, n});
    auto cpu_output = at::empty({m}, cpu_input.options());
    at::sum_out(cpu_output, cpu_input, at::OptionalIntArrayRef({1}), false,
                at::kDouble);
    std::vector<double> expected(m);
    std::memcpy(expected.data(), cpu_output.data_ptr<double>(),
                expected.size() * sizeof(double));
    auto operation = [&] {
      at::sum_out(output, input, at::OptionalIntArrayRef({1}), false,
                  at::kDouble);
    };
    return finish_double("aten_sum", "8323a8b552b16506", output, expected,
                         operation, "M=65536_N=64");
  }
  constexpr int64_t m = 512;
  constexpr int64_t n = 512;
  constexpr int64_t k = 512;
  auto a_host = raised_double_values(m * k);
  auto b_host = raised_double_values(k * n);
  auto c_host = raised_double_values(m * n);
  auto a = device_double_tensor(a_host).reshape({m, k});
  auto b = device_double_tensor(b_host).reshape({k, n});
  auto output = at::empty({m, n}, a.options());
  auto c = device_double_tensor(c_host).reshape({m, n});
  auto cpu_a = cpu_double_tensor(a_host).reshape({m, k});
  auto cpu_b = cpu_double_tensor(b_host).reshape({k, n});
  auto cpu_c = cpu_double_tensor(c_host).reshape({m, n});
  auto cpu_output = at::empty({m, n}, cpu_a.options());
  if (kernel == "aten_addmm")
    at::addmm_out(cpu_output, cpu_c, cpu_a, cpu_b, 0.5, 0.75);
  else
    at::mm_out(cpu_output, cpu_a, cpu_b);
  std::vector<double> expected(m * n);
  std::memcpy(expected.data(), cpu_output.data_ptr<double>(),
              expected.size() * sizeof(double));
  auto operation = [&] {
    if (kernel == "aten_addmm") at::addmm_out(output, c, a, b, 0.5, 0.75);
    else at::mm_out(output, a, b);
  };
  const char *recipe = kernel == "aten_addmm" ? "ffd39b2736a01291" :
                                                "cd8356bf8629a037";
  return finish_double(kernel.c_str(), recipe, output, expected, operation,
                       "M=512_N=512_K=512");
}

int whole_elu_family(const std::string &kernel) {
  auto first_host = raised_fixture_values(kPointwiseElements);
  auto second_host = raised_fixture_values(kPointwiseElements);
  if (kernel == "aten_elu_backward") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      second_host[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
  }
  if (kernel == "aten_log_sigmoid_backward_cpu") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      second_host[i] = std::exp(-std::fabs(first_host[i]));
  }
  auto first = device_tensor(first_host);
  auto second = device_tensor(second_host);
  auto output = at::empty_like(first);
  auto cpu_first = cpu_tensor(first_host);
  auto cpu_second = cpu_tensor(second_host);
  auto cpu_output = at::empty_like(cpu_first);
  auto apply = [&](at::Tensor &out, const at::Tensor &a, const at::Tensor &b) {
    if (kernel == "aten_elu")
      at::elu_out(out, a, 1.25, 0.75, 1.0);
    else if (kernel == "aten_elu_backward")
      at::elu_backward_out(out, a, 1.25, 0.75, 1.0, true, b);
    else
      throw std::runtime_error("log-sigmoid uses its three-input path");
  };
  if (kernel == "aten_log_sigmoid_backward_cpu") {
    // The fixture ABI is input, buffer, grad, output.  Use a third tensor for
    // grad while keeping the internal ATen buffer exactly as supplied.
    auto grad_host = raised_fixture_values(kPointwiseElements);
    auto grad = device_tensor(grad_host);
    auto cpu_grad = cpu_tensor(grad_host);
    at::log_sigmoid_backward_out(cpu_output, cpu_grad, cpu_first, cpu_second);
    std::vector<float> expected(kPointwiseElements);
    std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
                expected.size() * sizeof(float));
    auto operation = [&] {
      at::log_sigmoid_backward_out(output, grad, first, second);
    };
    return finish(kernel.c_str(), "aab50d8328c1a843", output, expected,
                  operation);
  }
  apply(cpu_output, cpu_first, cpu_second);
  std::vector<float> expected(kPointwiseElements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { apply(output, first, second); };
  const char *recipe = kernel == "aten_elu" ? "d81584f45f031a1c" :
                                              "905088a00ae3d786";
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int whole_integer_reduction(const std::string &kernel) {
  constexpr int64_t rows = 131072;
  constexpr int64_t columns = 64;
  auto host = raised_fixture_values(rows * columns);
  auto input = device_tensor(host).reshape({rows, columns});
  auto output = at::empty({rows}, input.options().dtype(at::kLong));
  auto cpu_input = cpu_tensor(host).reshape({rows, columns});
  auto cpu_output = at::empty({rows}, cpu_input.options().dtype(at::kLong));
  auto apply = [&](at::Tensor &out, const at::Tensor &in) {
    if (kernel == "aten_argmax_cpu") at::argmax_out(out, in, 1, false);
    else if (kernel == "aten_argmin_cpu") at::argmin_out(out, in, 1, false);
    else at::count_nonzero_out(out, in, at::IntArrayRef({1}));
  };
  apply(cpu_output, cpu_input);
  std::vector<int64_t> expected(rows);
  std::memcpy(expected.data(), cpu_output.data_ptr<int64_t>(),
              expected.size() * sizeof(int64_t));
  auto operation = [&] { apply(output, input); };
  const char *recipe = kernel == "aten_argmax_cpu" ? "8b87dc0870be83c0" :
      kernel == "aten_argmin_cpu" ? "927e47c252552e8f" :
                                    "c958ea502e1138a8";
  const char *shape = kernel == "aten_count_nonzero_impl_cpu" ?
      "R=131072_C=64" : "R=131072_K=64";
  return finish_long(kernel.c_str(), recipe, output, expected, operation, shape);
}

int whole_cumprod() {
  constexpr int64_t rows = 1448;
  constexpr int64_t columns = 2897;
  auto host = raised_fixture_values(rows * columns);
  auto input = device_tensor(host).reshape({rows, columns});
  auto output = at::empty_like(input);
  auto cpu_input = cpu_tensor(host).reshape({rows, columns});
  auto cpu_output = at::empty_like(cpu_input);
  at::cumprod_out(cpu_output, cpu_input, 1, at::kFloat);
  std::vector<float> expected(rows * columns);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] { at::cumprod_out(output, input, 1, at::kFloat); };
  return finish("aten_cumprod_cpu", "8c78205482ac384d", output, expected,
                operation, nullptr, nullptr, "K=2897_R=1448");
}

int whole_int_mm() {
  constexpr int64_t m = 512;
  constexpr int64_t n = 512;
  constexpr int64_t k = 1024;
  std::vector<int8_t> a_host(m * k), b_host(k * n);
  for (int64_t i = 0; i < m * k; ++i)
    a_host[i] = static_cast<int8_t>(static_cast<int>(i % 13) - 6);
  for (int64_t i = 0; i < k * n; ++i)
    b_host[i] = static_cast<int8_t>(static_cast<int>(i % 13) - 6);
  auto a = device_byte_tensor(a_host).reshape({m, k});
  auto b = device_byte_tensor(b_host).reshape({k, n});
  auto output = at::empty({m, n}, a.options().dtype(at::kInt));
  auto cpu_a = cpu_byte_tensor(a_host).reshape({m, k});
  auto cpu_b = cpu_byte_tensor(b_host).reshape({k, n});
  auto cpu_output = at::empty({m, n}, cpu_a.options().dtype(at::kInt));
  at::_int_mm_out(cpu_output, cpu_a, cpu_b);
  std::vector<int32_t> expected(m * n);
  std::memcpy(expected.data(), cpu_output.data_ptr<int32_t>(),
              expected.size() * sizeof(int32_t));
  auto operation = [&] { at::_int_mm_out(output, a, b); };
  return finish_int("aten_int_mm_out_cpu", "2ee5f09162f73db9", output,
                    expected, operation, "M=512_N=512_K=1024");
}

int whole_convolution(const std::string &kernel) {
  const bool conv1d = kernel == "aten_conv1d";
  const bool conv2d = kernel == "aten_conv2d";
  const bool transpose2d = kernel == "aten_conv_transpose2d";
  const bool transpose3d = kernel == "aten_conv_transpose3d_cpu";
  const int64_t batch = transpose3d ? 1 : conv1d ? 32 : conv2d ? 4 : 2;
  const int64_t input_channels = transpose3d ? 8 : conv1d ? 64 : 16;
  const int64_t output_channels = transpose3d ? 16 : conv1d ? 128 : 32;
  const int64_t depth = 32;
  const int64_t height = transpose3d ? 32 : 128;
  const int64_t width = conv1d ? 4096 : transpose3d ? 32 : 128;
  constexpr int64_t k = 3;
  const bool transposed = transpose2d || transpose3d;
  std::vector<int64_t> input_shape = conv1d
      ? std::vector<int64_t>{batch, input_channels, width}
      : transpose3d
          ? std::vector<int64_t>{batch, input_channels, depth, height, width}
          : std::vector<int64_t>{batch, input_channels, height, width};
  std::vector<int64_t> weight_shape = conv1d
      ? std::vector<int64_t>{output_channels, input_channels, k}
      : transpose3d
          ? std::vector<int64_t>{input_channels, output_channels, k, k, k}
          : transpose2d
              ? std::vector<int64_t>{input_channels, output_channels, k, k}
              : std::vector<int64_t>{output_channels, input_channels, k, k};
  std::vector<int64_t> output_shape = conv1d
      ? std::vector<int64_t>{batch, output_channels, width - k + 1}
      : transpose3d
          ? std::vector<int64_t>{batch, output_channels, depth + k - 1,
                                 height + k - 1, width + k - 1}
          : transposed
              ? std::vector<int64_t>{batch, output_channels, height + k - 1,
                                     width + k - 1}
              : std::vector<int64_t>{batch, output_channels, height - k + 1,
                                     width - k + 1};
  int64_t input_elements = 1, weight_elements = 1, output_elements = 1;
  for (int64_t v : input_shape) input_elements *= v;
  for (int64_t v : weight_shape) weight_elements *= v;
  for (int64_t v : output_shape) output_elements *= v;
  auto input_host = raised_fixture_values(input_elements);
  auto weight_host = raised_fixture_values(weight_elements);
  auto input = device_tensor(input_host).reshape(input_shape);
  auto weight = device_tensor(weight_host).reshape(weight_shape);
  auto output = at::empty(output_shape, input.options());
  at::Tensor bias;
  std::vector<float> bias_host;
  std::optional<at::Tensor> bias_optional = std::nullopt;
  if (conv1d) {
    bias_host = raised_fixture_values(output_channels);
    bias = device_tensor(bias_host);
    bias_optional = bias;
  }
  auto cpu_input = cpu_tensor(input_host).reshape(input_shape);
  auto cpu_weight = cpu_tensor(weight_host).reshape(weight_shape);
  auto cpu_output = at::empty(output_shape, cpu_input.options());
  std::optional<at::Tensor> cpu_bias_optional = std::nullopt;
  at::Tensor cpu_bias;
  if (conv1d) {
    cpu_bias = cpu_tensor(bias_host);
    cpu_bias_optional = cpu_bias;
  }
  const std::vector<int64_t> stride(conv1d ? 1 : transpose3d ? 3 : 2, 1);
  const std::vector<int64_t> padding(stride.size(), 0);
  const std::vector<int64_t> dilation(stride.size(), 1);
  const std::vector<int64_t> output_padding(stride.size(), 0);
  at::convolution_out(cpu_output, cpu_input, cpu_weight, cpu_bias_optional,
                      stride, padding, dilation, transposed, output_padding, 1);
  std::vector<float> expected(output_elements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    at::convolution_out(output, input, weight, bias_optional, stride, padding,
                        dilation, transposed, output_padding, 1);
  };
  const char *recipe = conv1d ? "a345c29492c9ca0c" :
      conv2d ? "3b3bcbef73116a11" :
      transpose2d ? "c294507cdb3f054e" : "cb3ec4a1c6f08eb8";
  const char *shape = conv1d ? "B=32_IC=64_OC=128_W=4096_K=3" :
      conv2d ? "B=4_IC=16_OC=32_H=128_W=128_KH=3_KW=3" :
      transpose2d ? "B=2_IC=16_OC=32_H=128_W=128_K=3" :
                    "C=8_O=16_D=32_H=32_W=32_K=3";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape);
}

int whole_nested_sum_backward() {
  constexpr int64_t batch = 724;
  constexpr int64_t elements = 5793;
  auto host = raised_fixture_values(batch);
  auto grad = device_tensor(host).reshape({batch, 1});
  auto output = at::empty({batch, elements}, grad.options());
  auto cpu_grad = cpu_tensor(host).reshape({batch, 1});
  auto cpu_output = at::empty({batch, elements}, cpu_grad.options());
  at::expand_copy_out(cpu_output, cpu_grad, {batch, elements}, false);
  std::vector<float> expected(batch * elements);
  std::memcpy(expected.data(), cpu_output.data_ptr<float>(),
              expected.size() * sizeof(float));
  auto operation = [&] {
    at::expand_copy_out(output, grad, {batch, elements}, false);
  };
  return finish("aten_nested_sum_backward_cpu", "ac753cd95b469a53",
                output, expected, operation, nullptr, nullptr,
                "B=724_N=5793");
}

int whole_sampled_addmm_sparse_csr() {
  constexpr int64_t rows = 4096;
  constexpr int64_t inner = 512;
  constexpr int64_t columns = 4096;
  constexpr int64_t nnz = 262144;
  constexpr int64_t per_row = nnz / rows;
  std::vector<int32_t> crow_host(rows + 1), column_host(nnz);
  for (int64_t r = 0; r <= rows; ++r)
    crow_host[r] = static_cast<int32_t>(r * per_row);
  for (int64_t p = 0; p < nnz; ++p)
    column_host[p] = static_cast<int32_t>(p % columns);
  auto self_host = raised_fixture_values(nnz);
  auto a_host = raised_fixture_values(rows * inner);
  auto b_host = raised_fixture_values(inner * columns);
  std::vector<float> expected(nnz);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t p = crow_host[r]; p < crow_host[r + 1]; ++p) {
      float product = 0.0f;
      const int64_t c = column_host[p];
      for (int64_t k = 0; k < inner; ++k)
        product += a_host[r * inner + k] * b_host[k * columns + c];
      expected[p] = -0.25f * self_host[p] + 0.75f * product;
    }
  }
  auto crow = device_int_tensor(crow_host);
  auto columns_index = device_int_tensor(column_host);
  auto self_values = device_tensor(self_host);
  auto output_values = device_tensor(self_host);
  auto options = self_values.options().layout(at::kSparseCsr);
  auto self = at::sparse_csr_tensor(crow, columns_index, self_values,
                                    {rows, columns}, options);
  auto output = at::sparse_csr_tensor(crow, columns_index, output_values,
                                      {rows, columns}, options);
  auto a = device_tensor(a_host).reshape({rows, inner});
  auto b = device_tensor(b_host).reshape({inner, columns});
  auto operation = [&] {
    at::sparse_sampled_addmm_out(output, self, a, b, -0.25, 0.75);
  };
  return finish("aten_sampled_addmm_sparse_csr_cpu", "3c077624966ff30d",
                output_values, expected, operation, nullptr, nullptr,
                "R=4096_K=512_C=4096_NNZ=262144_N=262144");
}

int whole_sparse_csr_addmm() {
  constexpr int64_t rows = 65536;
  constexpr int64_t inner = 65536;
  constexpr int64_t columns = 64;
  constexpr int64_t nnz = 4194304;
  constexpr int64_t per_row = nnz / rows;
  std::vector<int32_t> crow_host(rows + 1), column_host(nnz);
  for (int64_t r = 0; r <= rows; ++r)
    crow_host[r] = static_cast<int32_t>(r * per_row);
  for (int64_t p = 0; p < nnz; ++p)
    column_host[p] = static_cast<int32_t>(p % inner);
  auto value_host = raised_fixture_values(nnz);
  auto rhs_host = raised_fixture_values(inner * columns);
  std::vector<float> expected(rows * columns, 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < columns; ++c) {
      float product = 0.0f;
      for (int64_t p = crow_host[r]; p < crow_host[r + 1]; ++p)
        product += value_host[p] * rhs_host[column_host[p] * columns + c];
      expected[r * columns + c] = product;
    }
  }
  auto crow = device_int_tensor(crow_host);
  auto columns_index = device_int_tensor(column_host);
  auto values = device_tensor(value_host);
  auto options = values.options().layout(at::kSparseCsr);
  auto sparse = at::sparse_csr_tensor(crow, columns_index, values,
                                      {rows, inner}, options);
  auto rhs = device_tensor(rhs_host).reshape({inner, columns});
  auto output = at::empty({rows, columns}, rhs.options());
  auto operation = [&] { at::addmm_out(output, output, sparse, rhs, 0.0, 1.0); };
  return finish("aten_sparse_csr_addmm_cpu", "ceb60fb7c03704d8", output,
                expected, operation, nullptr, nullptr,
                "R=65536_K=65536_C=64_N=4194304");
}

int clamp_tensor() {
  auto host = raised_fixture_values(kPointwiseElements);
  auto min_host = raised_fixture_values(kPointwiseElements);
  auto max_host = raised_fixture_values(kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    expected[i] = host[i] < min_host[i]
                      ? min_host[i]
                      : (host[i] > max_host[i] ? max_host[i] : host[i]);
  }
  auto input = device_tensor(host);
  auto minimum = device_tensor(min_host);
  auto maximum = device_tensor(max_host);
  auto output = at::empty_like(input);
  auto operation = [&] {
    at::clamp_out(output, input, std::optional<at::Tensor>(minimum),
                  std::optional<at::Tensor>(maximum));
  };
  return finish("aten_clamp_cpu", "3cd8e32878aada7c", output, expected,
                operation);
}

int clamp_scalar(const std::string &kernel) {
  auto host = raised_fixture_values(kPointwiseElements);
  auto expected = host;
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_clamp_max_scalar_cpu") {
    for (float &value : expected)
      value = value > 0.5f ? 0.5f : value;
    operation = [&] { at::clamp_max_out(output, input, 0.5); };
    recipe = "20afe51bff9319b3";
  } else if (kernel == "aten_clamp_min_scalar_cpu") {
    for (float &value : expected)
      value = value < 0.5f ? 0.5f : value;
    operation = [&] { at::clamp_min_out(output, input, 0.5); };
    recipe = "79cad4471d373dda";
  } else {
    for (float &value : expected)
      value = value < 0.5f ? 0.5f : (value > 0.5f ? 0.5f : value);
    operation = [&] { at::clamp_out(output, input, 0.5, 0.5); };
    recipe = "a5b7d49626a84340";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int masked_scale() {
  auto host = raised_fixture_values(kPointwiseElements);
  auto expected = host;
  for (float &value : expected)
    value *= 0.5f;
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  auto operation = [&] { at::mul_out(output, input, 0.5); };
  return finish("aten_masked_scale", "f023c7dce84c90d0", output, expected,
                operation);
}

int shrink_or_hardtanh(const std::string &kernel) {
  auto host = fixture_values(kPointwiseElements);
  auto expected = host;
  auto input = device_tensor(host);
  auto output = at::empty_like(input);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_hardshrink") {
    for (float &value : expected)
      value = value >= -0.35f && value <= 0.35f ? 0.0f : value;
    operation = [&] { at::hardshrink_out(output, input, 0.35); };
    recipe = "6b84be6ffbc2f9cd";
  } else if (kernel == "aten_softshrink") {
    for (float &value : expected)
      value = value > 0.35f
                  ? value - 0.35f
                  : (value < -0.35f ? value + 0.35f : value * 0.0f);
    operation = [&] { at::softshrink_out(output, input, 0.35); };
    recipe = "586089c4ed5789f9";
  } else {
    for (float &value : expected)
      value = value < 0.5f ? 0.5f : (value > 0.5f ? 0.5f : value);
    operation = [&] { at::hardtanh_out(output, input, 0.5, 0.5); };
    recipe = "83fe1f5a9c14ac29";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int activation_backward(const std::string &kernel) {
  auto self_host = fixture_values(kPointwiseElements);
  auto grad_host = fixture_values(kPointwiseElements);
  std::reverse(grad_host.begin(), grad_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto self = device_tensor(self_host);
  auto grad = device_tensor(grad_host);
  auto output = at::empty_like(self);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_hardtanh_backward") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = self_host[i] <= -0.7f || self_host[i] >= 0.8f
                        ? 0.0f
                        : grad_host[i];
    operation = [&] {
      at::hardtanh_backward_out(output, grad, self, -0.7, 0.8);
    };
    recipe = "6488125976d74959";
  } else {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = self_host[i] >= -0.35f && self_host[i] <= 0.35f
                        ? 0.0f
                        : grad_host[i];
    operation = [&] { at::hardshrink_backward_out(output, grad, self, 0.35); };
    recipe = "0685394b0abd4c38";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int loss_elementwise(const std::string &kernel) {
  auto input_host = fixture_values(kPointwiseElements);
  auto target_host = fixture_values(kPointwiseElements);
  std::rotate(target_host.begin(), target_host.begin() + 31,
              target_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto input = device_tensor(input_host);
  auto target = device_tensor(target_host);
  auto output = at::empty_like(input);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_huber_elementwise") {
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float z = input_host[i] - target_host[i];
      const float az = std::fabs(z);
      expected[i] = az < 0.6f ? 0.5f * z * z
                              : 0.6f * (az - 0.3f);
    }
    operation = [&] { at::huber_loss_out(output, input, target, 0, 0.6); };
    recipe = "40458531c0ed998a";
  } else {
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float z = input_host[i] - target_host[i];
      const float az = std::fabs(z);
      expected[i] = az < 0.6f ? 0.5f * z * z / 0.6f : az - 0.3f;
    }
    operation = [&] {
      at::smooth_l1_loss_out(output, input, target, 0, 0.6);
    };
    recipe = "be81f2aea77eb181";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int loss_backward(const std::string &kernel) {
  auto input_host = fixture_values(kPointwiseElements);
  auto target_host = fixture_values(kPointwiseElements);
  std::rotate(target_host.begin(), target_host.begin() + 31,
              target_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto input = device_tensor(input_host);
  auto target = device_tensor(target_host);
  auto output = at::empty_like(input);
  auto grad = at::empty_like(input);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_huber_backward") {
    grad.fill_(0.75);
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float z = input_host[i] - target_host[i];
      expected[i] = z < -0.6f ? -0.45f : (z > 0.6f ? 0.45f : 0.75f * z);
    }
    operation = [&] {
      at::huber_loss_backward_out(output, grad, input, target, 0, 0.6);
    };
    recipe = "b003b17441ada95b";
  } else if (kernel == "aten_smooth_l1_backward") {
    grad.fill_(0.75);
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float z = input_host[i] - target_host[i];
      expected[i] = z <= -0.6f
                        ? -0.75f
                        : (z >= 0.6f ? 0.75f : 0.75f * z / 0.6f);
    }
    operation = [&] {
      at::smooth_l1_loss_backward_out(output, grad, input, target, 0, 0.6);
    };
    recipe = "50c63ccf860d2434";
  } else {
    grad.fill_(0.25);
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = 0.5f * (input_host[i] - target_host[i]);
    operation = [&] {
      at::mse_loss_backward_out(output, grad, input, target, 0);
    };
    recipe = "3848b837a6d2bcb9";
  }
  synchronize("gradient initialization");
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int lerp_case(const std::string &kernel) {
  auto self_host = fixture_values(kPointwiseElements);
  auto end_host = fixture_values(kPointwiseElements);
  std::rotate(end_host.begin(), end_host.begin() + 47, end_host.end());
  std::vector<float> weight_host(kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    weight_host[i] = static_cast<float>(i % 17) / 16.0f;
  auto self = device_tensor(self_host);
  auto end = device_tensor(end_host);
  auto output = at::empty_like(self);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_lerp") {
    auto weight = device_tensor(weight_host);
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = self_host[i] +
                    weight_host[i] * (end_host[i] - self_host[i]);
    operation = [&, weight] { at::lerp_out(output, self, end, weight); };
    recipe = "4e802a140a827300";
  } else {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = self_host[i] + 0.3f * (end_host[i] - self_host[i]);
    operation = [&] { at::lerp_out(output, self, end, 0.3); };
    recipe = kernel == "aten_lerp_scalar" ? "3ed458b205f9ff3d"
                                            : "78031c8798b7cb9a";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int logit_case(const std::string &kernel) {
  auto self_host = fixture_values(kPointwiseElements);
  auto self = device_tensor(self_host);
  auto output = at::empty_like(self);
  std::vector<float> expected(kPointwiseElements);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_logit") {
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float z = self_host[i] < 0.5f
                          ? 0.5f
                          : (self_host[i] > 0.5f ? 0.5f : self_host[i]);
      expected[i] = std::log(z / (1.0f - z));
    }
    operation = [&] { at::logit_out(output, self, 0.5); };
    recipe = "8d961159ef398d33";
  } else {
    auto grad_host = fixture_values(kPointwiseElements);
    std::reverse(grad_host.begin(), grad_host.end());
    auto grad = device_tensor(grad_host);
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = self_host[i] < 0.5f || self_host[i] > 0.5f
                        ? 0.0f
                        : grad_host[i] / (self_host[i] * (1.0f - self_host[i]));
    operation = [&, grad] { at::logit_backward_out(output, grad, self, 0.5); };
    recipe = "9d4fddc2f83a2bdf";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int threshold_backward() {
  auto self_host = fixture_values(kPointwiseElements);
  auto grad_host = fixture_values(kPointwiseElements);
  std::reverse(grad_host.begin(), grad_host.end());
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    expected[i] = self_host[i] <= 0.35f ? 0.0f : grad_host[i];
  auto self = device_tensor(self_host);
  auto grad = device_tensor(grad_host);
  auto output = at::empty_like(self);
  auto operation = [&] {
    at::threshold_backward_out(output, grad, self, 0.35);
  };
  return finish("aten_threshold_backward", "c488a87f05701220", output,
                expected, operation);
}

int glu_forward() {
  auto a_host = fixture_values(kPointwiseElements);
  auto b_host = fixture_values(kPointwiseElements);
  std::rotate(b_host.begin(), b_host.begin() + 19, b_host.end());
  std::vector<float> packed(2 * kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    packed[2 * i] = a_host[i];
    packed[2 * i + 1] = b_host[i];
    expected[i] = a_host[i] / (1.0f + std::exp(-b_host[i]));
  }
  auto input = device_tensor(packed).reshape({kPointwiseElements, 2});
  auto output = at::empty({kPointwiseElements, 1}, input.options());
  auto operation = [&] { at::glu_out(output, input, 1); };
  return finish("aten_glu", "02a7dcceff8d5d61", output, expected,
                operation);
}

int log_sigmoid_forward() {
  constexpr int64_t n = 8388608;
  auto input_host = fixture_values(n);
  std::vector<float> expected(n);
  std::vector<float> expected_buffer(n);
  for (int64_t i = 0; i < n; ++i) {
    const float absolute = std::fabs(input_host[i]);
    expected_buffer[i] = std::exp(-absolute);
    expected[i] = (input_host[i] < 0.0f ? input_host[i] : 0.0f) -
                  std::log1p(expected_buffer[i]);
  }
  auto input = device_tensor(input_host);
  auto output = at::empty_like(input);
  auto buffer = at::empty_like(input);
  auto absolute = at::empty_like(input);
  auto negative_absolute = at::empty_like(input);
  auto nonpositive = at::empty_like(input);
  auto logarithm = at::empty_like(input);
  auto operation = [&] {
    // CUDA log_sigmoid_forward deliberately leaves its CPU-only auxiliary
    // buffer empty.  Materialize both observable fixture outputs with
    // existing, preallocated ATen operations instead.
    at::abs_out(absolute, input);
    at::neg_out(negative_absolute, absolute);
    at::exp_out(buffer, negative_absolute);
    at::clamp_max_out(nonpositive, input, 0.0);
    at::log1p_out(logarithm, buffer);
    at::sub_out(output, nonpositive, logarithm);
  };
  return finish("aten_log_sigmoid_cpu", "f5856fc92edb07f6", output,
                expected, operation, &buffer, &expected_buffer,
                "N=8388608");
}

int angle_complex() {
  auto real_host = fixture_values(kPointwiseElements);
  auto imag_host = fixture_values(kPointwiseElements);
  std::rotate(imag_host.begin(), imag_host.begin() + 23, imag_host.end());
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    expected[i] = std::atan2(imag_host[i], real_host[i]);
  auto real = device_tensor(real_host);
  auto imag = device_tensor(imag_host);
  auto output = at::empty_like(real);
  // The fixture is already scalarized into real and imaginary arrays.  The
  // exact existing ATen operation for that boundary is atan2(imag, real);
  // it also avoids the complex-angle CUDA JIT/NVRTC dependency.
  auto operation = [&] { at::atan2_out(output, imag, real); };
  return finish("aten_angle_complex_scalarized", "dea8f1b7b7cfa157",
                output, expected, operation);
}

int diff_forward() {
  constexpr int64_t n = 16777216;
  auto input_host = fixture_values(n);
  std::vector<float> expected(n - 1);
  for (int64_t i = 0; i < n - 1; ++i)
    expected[i] = input_host[i + 1] - input_host[i];
  auto input = device_tensor(input_host);
  auto output = at::empty({n - 1}, input.options());
  auto operation = [&] { at::diff_out(output, input, 1, -1, {}, {}); };
  return finish("aten_diff_cpu", "58729da87b52269a", output, expected,
                operation, nullptr, nullptr, "N=16777216");
}

int blas_vector(const std::string &kernel) {
  auto input_host = fixture_values(kPointwiseElements);
  auto other_host = fixture_values(kPointwiseElements);
  std::rotate(other_host.begin(), other_host.begin() + 37, other_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto input = device_tensor(input_host);
  auto other = device_tensor(other_host);
  auto output = at::empty_like(input);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_blas_axpy_cpu") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = other_host[i] + 0.375f * input_host[i];
    operation = [&] { at::add_out(output, other, input, 0.375); };
    recipe = "6052b858c06bd41d";
  } else if (kernel == "aten_blas_scale_cpu") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = 0.375f * input_host[i];
    operation = [&] { at::mul_out(output, input, 0.375); };
    recipe = "cf3d9e50f32fba45";
  } else {
    expected = input_host;
    operation = [&] { output.copy_(input, true); };
    recipe = "360a1fed35fdcb46";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int narrow_copy_dense() {
  constexpr int64_t rows = 1448;
  constexpr int64_t columns = 2897;
  constexpr int64_t start = 1024;
  constexpr int64_t length = 724;
  auto input_host = fixture_values(rows * columns);
  std::vector<float> expected(rows * length);
  for (int64_t row = 0; row < rows; ++row)
    std::copy_n(input_host.begin() + row * columns + start, length,
                expected.begin() + row * length);
  auto input = device_tensor(input_host).reshape({rows, columns});
  auto output = at::empty({rows, length}, input.options());
  // narrow_copy.out has no CUDA registration in the pinned ATen build.
  // Construct its metadata-only source view outside timing and use ATen's
  // existing CUDA copy implementation for the observable dense copy region.
  auto source_view = input.narrow(1, start, length);
  auto operation = [&] { output.copy_(source_view, true); };
  return finish("aten_narrow_copy_dense_cpu", "031841e44ca18076",
                output, expected, operation, nullptr, nullptr,
                "C=2897_L=724_R=1448");
}

int unbind_copy() {
  constexpr int64_t batches = 512;
  constexpr int64_t elements = 8192;
  auto input_host = fixture_values(batches * elements);
  auto input = device_tensor(input_host).reshape({batches, elements});
  auto output = at::empty_like(input);
  std::vector<at::Tensor> output_rows;
  output_rows.reserve(batches);
  for (int64_t row = 0; row < batches; ++row)
    output_rows.push_back(output.select(0, row));
  auto operation = [&] { at::unbind_copy_out(output_rows, input, 0); };
  return finish("aten_unbind_copy_cpu", "8eb65b902214f090", output,
                input_host, operation, nullptr, nullptr, "B=512_N=8192");
}

int add_clamp() {
  auto a_host = fixture_values(kPointwiseElements);
  auto b_host = fixture_values(kPointwiseElements);
  std::rotate(b_host.begin(), b_host.begin() + 29, b_host.end());
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i) {
    const float value = a_host[i] + 0.375f * b_host[i];
    expected[i] = std::max(-0.6f, std::min(0.7f, value));
  }
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto intermediate = at::empty_like(a);
  auto output = at::empty_like(a);
  auto operation = [&] {
    at::add_out(intermediate, a, b, 0.375);
    at::clamp_out(output, intermediate, -0.6, 0.7);
  };
  return finish("aten_add_clamp", "e6061ef1e74be3cd", output, expected,
                operation);
}

int scalarized_complex(const std::string &kernel) {
  const int64_t n = kernel == "aten_as_complex_cpu" ? 2097151
                                                     : kPointwiseElements;
  auto real_host = fixture_values(n);
  auto imag_host = fixture_values(n);
  std::rotate(imag_host.begin(), imag_host.begin() + 23, imag_host.end());
  std::vector<float> expected_real(n);
  std::vector<float> expected_imag(n);
  auto output_real = device_tensor(expected_real);
  auto output_imag = device_tensor(expected_imag);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_as_complex_cpu") {
    std::vector<float> packed(2 * n);
    for (int64_t i = 0; i < n; ++i) {
      packed[2 * i] = real_host[i];
      packed[2 * i + 1] = imag_host[i];
    }
    expected_real = real_host;
    expected_imag = imag_host;
    auto input = device_tensor(packed).reshape({n, 2});
    auto real_view = input.select(1, 0);
    auto imag_view = input.select(1, 1);
    operation = [&, input, real_view, imag_view] {
      output_real.copy_(real_view, true);
      output_imag.copy_(imag_view, true);
    };
    recipe = "95d705a24393ccd9";
  } else {
    auto real = device_tensor(real_host);
    auto imag = device_tensor(imag_host);
    if (kernel == "aten_complex_scalarized") {
      expected_real = real_host;
      expected_imag = imag_host;
      operation = [&, real, imag] {
        output_real.copy_(real, true);
        output_imag.copy_(imag, true);
      };
      recipe = "4aae25a1eecb3239";
    } else if (kernel == "aten_conj_complex_scalarized") {
      expected_real = real_host;
      for (int64_t i = 0; i < n; ++i)
        expected_imag[i] = -imag_host[i];
      operation = [&, real, imag] {
        output_real.copy_(real, true);
        at::neg_out(output_imag, imag);
      };
      recipe = "cc9eb5896e00ae5b";
    } else {
      for (int64_t i = 0; i < n; ++i) {
        expected_real[i] = real_host[i] * std::cos(imag_host[i]);
        expected_imag[i] = real_host[i] * std::sin(imag_host[i]);
      }
      auto trigonometric = at::empty_like(real);
      operation = [&, real, imag, trigonometric]() mutable {
        at::cos_out(trigonometric, imag);
        at::mul_out(output_real, real, trigonometric);
        at::sin_out(trigonometric, imag);
        at::mul_out(output_imag, real, trigonometric);
      };
      recipe = "42696ee2f414f4f7";
    }
  }
  const std::string shape = "N=" + std::to_string(n);
  return finish(kernel.c_str(), recipe, output_real, expected_real, operation,
                &output_imag, &expected_imag, shape.c_str());
}

int round_away_from_zero(const std::string &kernel) {
  auto input_host = fixture_values(kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  const float scale = kernel == "aten_round_decimals" ? 0.5f : 1.0f;
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    expected[i] = std::round(input_host[i] * scale) / scale;
  auto input = device_tensor(input_host);
  auto scaled = at::empty_like(input);
  auto adjusted_positive = at::empty_like(input);
  auto adjusted_negative = at::empty_like(input);
  auto positive = at::empty_like(input);
  auto negative = at::empty_like(input);
  auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
  auto output = at::empty_like(input);
  auto operation = [&] {
    if (scale == 1.0f)
      scaled.copy_(input, true);
    else
      at::mul_out(scaled, input, scale);
    at::add_out(adjusted_positive, scaled, 0.5);
    at::floor_out(positive, adjusted_positive);
    at::sub_out(adjusted_negative, scaled, 0.5);
    at::ceil_out(negative, adjusted_negative);
    at::ge_out(mask, scaled, 0.0);
    at::where_out(output, mask, positive, negative);
    if (scale != 1.0f)
      at::div_out(output, output, scale);
  };
  const char *recipe = kernel == "aten_round" ? "fc33c00aba68ebf8"
                                                : "6a4dc87767ea2f0d";
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int sparse_binary(const std::string &kernel) {
  auto a_host = fixture_values(kPointwiseElements);
  auto b_host = fixture_values(kPointwiseElements);
  std::rotate(b_host.begin(), b_host.begin() + 41, b_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto output = at::empty_like(a);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_sparse_add_values_cpu") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = a_host[i] + 0.5f * b_host[i];
    operation = [&] { at::add_out(output, a, b, 0.5); };
    recipe = "292b2dfa977dd624";
  } else {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = a_host[i] * b_host[i];
    operation = [&] { at::mul_out(output, a, b); };
    recipe = "cc84c109e16c002e";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int renorm_scale_factor() {
  auto input_host = fixture_values(kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    expected[i] = input_host[i] > 0.5f
                      ? 0.5f / (input_host[i] + 1.0e-7f)
                      : 1.0f;
  auto input = device_tensor(input_host);
  auto denominator = at::empty_like(input);
  auto quotient = at::empty_like(input);
  auto ones = at::ones_like(input);
  auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
  auto output = at::empty_like(input);
  auto operation = [&] {
    at::add_out(denominator, input, 1.0e-7);
    at::reciprocal_out(quotient, denominator);
    at::mul_out(quotient, quotient, 0.5);
    at::gt_out(mask, input, 0.5);
    at::where_out(output, mask, quotient, ones);
  };
  return finish("aten_renorm_scale_factor", "e42601bd48d63c94", output,
                expected, operation);
}

int glu_derivative(const std::string &kernel) {
  auto a_host = fixture_values(kPointwiseElements);
  auto b_host = fixture_values(kPointwiseElements);
  auto c_host = fixture_values(kPointwiseElements);
  auto d_host = fixture_values(kPointwiseElements);
  std::rotate(b_host.begin(), b_host.begin() + 11, b_host.end());
  std::rotate(c_host.begin(), c_host.begin() + 23, c_host.end());
  std::rotate(d_host.begin(), d_host.begin() + 47, d_host.end());
  std::vector<float> expected(kPointwiseElements);
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto c = device_tensor(c_host);
  auto d = device_tensor(d_host);
  auto first = at::empty_like(a);
  auto second = at::empty_like(a);
  auto output = at::empty_like(a);
  std::function<void()> operation;
  const char *recipe = nullptr;
  if (kernel == "aten_glu_backward") {
    for (int64_t i = 0; i < kPointwiseElements; ++i)
      expected[i] = (1.0f - a_host[i]) * a_host[i] * b_host[i] * c_host[i];
    operation = [&] {
      at::neg_out(first, a);
      at::add_out(first, first, 1.0);
      at::mul_out(second, first, a);
      at::mul_out(first, second, b);
      at::mul_out(output, first, c);
    };
    recipe = "eef22b792f3a0e64";
  } else {
    for (int64_t i = 0; i < kPointwiseElements; ++i) {
      const float sigmoid = 1.0f / (1.0f + std::exp(-b_host[i]));
      expected[i] = c_host[i] * sigmoid +
                    a_host[i] * (d_host[i] - sigmoid * d_host[i]);
    }
    auto sigmoid = at::empty_like(a);
    auto third = at::empty_like(a);
    operation = [&, sigmoid, third]() mutable {
      at::sigmoid_out(sigmoid, b);
      at::mul_out(first, c, sigmoid);
      at::mul_out(second, sigmoid, d);
      at::sub_out(third, d, second);
      at::mul_out(second, a, third);
      at::add_out(output, first, second);
    };
    recipe = "b9b515bd3789a6a2";
  }
  return finish(kernel.c_str(), recipe, output, expected, operation);
}

int batch_norm_affine() {
  auto input_host = fixture_values(kPointwiseElements);
  std::vector<float> expected(kPointwiseElements);
  for (int64_t i = 0; i < kPointwiseElements; ++i)
    expected[i] = input_host[i] * 0.75f - 0.2f;
  auto input = device_tensor(input_host);
  auto output = at::empty_like(input);
  auto operation = [&] {
    at::mul_out(output, input, 0.75);
    at::add_out(output, output, -0.2);
  };
  return finish("aten_batch_norm_cpu_entry", "dfec3f0041ccf405", output,
                expected, operation);
}

int dropout_feature_noise() {
  constexpr int64_t batch = 32;
  constexpr int64_t channels = 64;
  constexpr int64_t height = 64;
  constexpr int64_t width = 64;
  constexpr int64_t n = batch * channels * height * width;
  auto input_host = fixture_values(n);
  std::vector<float> mask_host(batch * channels);
  std::vector<float> expected(n);
  for (int64_t i = 0; i < batch * channels; ++i)
    mask_host[i] = (i % 3) != 0 ? 1.0f : 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t feature = i / (height * width);
    expected[i] = input_host[i] * mask_host[feature] * 1.25f;
  }
  auto input = device_tensor(input_host).reshape(
      {batch, channels, height, width});
  auto mask = device_tensor(mask_host).reshape({batch, channels, 1, 1});
  auto output = at::empty_like(input);
  auto operation = [&] {
    at::mul_out(output, input, mask);
    at::mul_out(output, output, 1.25);
  };
  return finish("aten_dropout_feature_noise_cpu", "bdd5f8a7dcb35084",
                output, expected, operation, nullptr, nullptr,
                "B=32_C=64_H=64_W=64");
}

int channel_shuffle() {
  constexpr int64_t batch = 14;
  constexpr int64_t groups = 14;
  constexpr int64_t channels_per_group = 28;
  constexpr int64_t height = 28;
  constexpr int64_t width = 28;
  constexpr int64_t channels = groups * channels_per_group;
  auto input_host = fixture_values(batch * channels * height * width);
  std::vector<float> expected(input_host.size());
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t g = 0; g < groups; ++g)
      for (int64_t c = 0; c < channels_per_group; ++c)
        for (int64_t hw = 0; hw < height * width; ++hw) {
          const int64_t source = ((b * channels + g * channels_per_group + c) *
                                  height * width) + hw;
          const int64_t target = ((b * channels + c * groups + g) *
                                  height * width) + hw;
          expected[target] = input_host[source];
        }
  auto input = device_tensor(input_host).reshape(
      {batch, channels, height, width});
  auto output = at::empty_like(input);
  auto operation = [&] { at::channel_shuffle_out(output, input, groups); };
  return finish("aten_channel_shuffle", "f199c08db2dbc190", output,
                expected, operation, nullptr, nullptr,
                "B=14_CPG=28_G=14_H=28_W=28");
}

int logical_reduce(const std::string &kernel) {
  const int64_t rows = kernel == "aten_or_reduce_cpu" ? 1448 : 131072;
  const int64_t columns = kernel == "aten_or_reduce_cpu" ? 2897 : 64;
  std::vector<int32_t> input_host(rows * columns);
  std::vector<int32_t> expected(rows);
  for (int64_t row = 0; row < rows; ++row) {
    bool result = kernel != "aten_or_reduce_cpu";
    for (int64_t column = 0; column < columns; ++column) {
      const int32_t value = ((row * 17 + column * 13) % 19) != 0;
      input_host[row * columns + column] = value;
      result = kernel == "aten_or_reduce_cpu" ? result || value
                                               : result && value;
    }
    expected[row] = result;
  }
  auto input = device_int_tensor(input_host).reshape({rows, columns});
  auto reduced = at::empty({rows}, input.options().dtype(at::kBool));
  auto output = at::empty({rows}, input.options());
  auto operation = [&] {
    if (kernel == "aten_or_reduce_cpu")
      at::any_out(reduced, input, 1, false);
    else
      at::all_out(reduced, input, 1, false);
    output.copy_(reduced, true);
  };
  const char *recipe = kernel == "aten_allany_dims_cpu"
                           ? "35a61ce54804f951"
                           : (kernel == "aten_and_reduce_cpu"
                                  ? "0926a61c1f3908b6"
                                  : "bb8a8744433bfaf3");
  const std::string shape = "R=" + std::to_string(rows) + "_" +
                            (kernel == "aten_or_reduce_cpu" ? "K=" : "C=") +
                            std::to_string(columns);
  return finish_int(kernel.c_str(), recipe, output, expected, operation,
                    shape.c_str());
}

int joint_scaling() {
  constexpr int64_t n = 16777216;
  auto a_host = fixture_values(n);
  auto b_host = fixture_values(n);
  std::rotate(b_host.begin(), b_host.begin() + 53, b_host.end());
  float max_a = 0.0f;
  float max_b = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    max_a = std::max(max_a, std::fabs(a_host[i]));
    max_b = std::max(max_b, std::fabs(b_host[i]));
  }
  auto a = device_tensor(a_host);
  auto b = device_tensor(b_host);
  auto absolute_a = at::empty_like(a);
  auto absolute_b = at::empty_like(b);
  auto maximum_a = at::empty({}, a.options());
  auto maximum_b = at::empty({}, b.options());
  auto output = at::empty({}, a.options());
  auto operation = [&] {
    at::abs_out(absolute_a, a);
    at::amax_out(maximum_a, absolute_a, {}, false);
    at::abs_out(absolute_b, b);
    at::amax_out(maximum_b, absolute_b, {}, false);
    at::mul_out(output, maximum_a, maximum_b);
  };
  return finish("aten_joint_scaling_cpu", "0f1d1b10e91e7c68", output,
                {max_a * max_b}, operation, nullptr, nullptr, "N=16777216");
}

int channel_shuffle_flat() {
  constexpr int64_t batch = 17;
  constexpr int64_t groups = 34;
  constexpr int64_t channels_per_group = 26;
  constexpr int64_t spatial = 276;
  constexpr int64_t channels = groups * channels_per_group;
  auto input_host = fixture_values(batch * channels * spatial);
  std::vector<float> expected(input_host.size());
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t g = 0; g < groups; ++g)
      for (int64_t c = 0; c < channels_per_group; ++c)
        for (int64_t s = 0; s < spatial; ++s)
          expected[(b * channels + c * groups + g) * spatial + s] =
              input_host[(b * channels + g * channels_per_group + c) *
                             spatial +
                         s];
  auto input = device_tensor(input_host).reshape({batch, channels, spatial});
  auto output = at::empty_like(input);
  auto operation = [&] { at::channel_shuffle_out(output, input, groups); };
  return finish("aten_channel_shuffle_cpu", "9c0f4efa20ff5874", output,
                expected, operation, nullptr, nullptr,
                "B=17_CPG=26_G=34_S=276");
}

int coo_to_csr(const std::string &kernel) {
  constexpr int64_t nonzeros = 4194304;
  constexpr int64_t rows = 65536;
  std::vector<int32_t> row_indices(nonzeros);
  std::vector<int32_t> expected(rows + 1);
  for (int64_t i = 0; i < nonzeros; ++i)
    row_indices[i] = static_cast<int32_t>((i * rows) / nonzeros);
  int64_t position = 0;
  for (int64_t row = 0; row <= rows; ++row) {
    while (position < nonzeros && row_indices[position] < row)
      ++position;
    expected[row] = static_cast<int32_t>(position);
  }
  auto input = device_int_tensor(row_indices);
  auto output = at::empty({rows + 1}, input.options());
  auto operation = [&] {
    at::_convert_indices_from_coo_to_csr_out(output, input, rows, true);
  };
  const char *recipe = kernel == "aten_convert_coo_to_csr_cpu"
                           ? "88b6f49f4ce53c69"
                           : "4cda79027c69d4df";
  return finish_int(kernel.c_str(), recipe, output, expected, operation,
                    "N=4194304_R=65536");
}

int compressed_block_convert() {
  constexpr int64_t rows = 2048;
  constexpr int64_t columns = 2048;
  constexpr int64_t block_rows = 4;
  constexpr int64_t block_columns = 4;
  auto input_host = fixture_values(rows * columns);
  std::vector<float> expected(input_host.size());
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column) {
      const int64_t target =
          ((((row / block_rows) * (columns / block_columns) +
             column / block_columns) *
                block_rows +
            row % block_rows) *
               block_columns +
           column % block_columns);
      expected[target] = input_host[row * columns + column];
    }
  auto input = device_tensor(input_host).reshape({rows, columns});
  auto source_view = input.reshape({rows / block_rows, block_rows,
                                    columns / block_columns, block_columns})
                         .permute({0, 2, 1, 3});
  auto output = at::empty({rows / block_rows, columns / block_columns,
                           block_rows, block_columns}, input.options());
  auto operation = [&] { output.copy_(source_view, true); };
  return finish("aten_compressed_block_convert_cpu", "5d2934fd76b7e9c2",
                output, expected, operation, nullptr, nullptr,
                "R=2048_C=2048_BR=4_BC=4");
}

int kron_impl(const std::string &kernel) {
  constexpr int64_t a_rows = 256;
  constexpr int64_t a_columns = 128;
  constexpr int64_t b_rows = 32;
  constexpr int64_t b_columns = 32;
  auto a_host = fixture_values(a_rows * a_columns);
  auto b_host = fixture_values(b_rows * b_columns);
  std::vector<float> expected(a_rows * b_rows * a_columns * b_columns);
  for (int64_t ar = 0; ar < a_rows; ++ar)
    for (int64_t ac = 0; ac < a_columns; ++ac)
      for (int64_t br = 0; br < b_rows; ++br)
        for (int64_t bc = 0; bc < b_columns; ++bc)
          expected[(ar * b_rows + br) * (a_columns * b_columns) +
                   ac * b_columns + bc] =
              a_host[ar * a_columns + ac] *
              b_host[br * b_columns + bc];
  auto a = device_tensor(a_host).reshape({a_rows, a_columns});
  auto b = device_tensor(b_host).reshape({b_rows, b_columns});
  auto output = at::empty({a_rows * b_rows, a_columns * b_columns},
                          a.options());
  auto operation = [&] { at::kron_out(output, a, b); };
  const char *recipe = kernel == "aten_kron_impl_cpu" ? "71b58bc9f4730368"
                                                       : "7607a09d96787dbe";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr, nullptr,
                "A=256_B=128_C=32_D=32");
}

int nested_clone() {
  constexpr int64_t batch = 724;
  constexpr int64_t elements = 5793;
  auto input_host = fixture_values(batch * elements);
  auto input = device_tensor(input_host).reshape({batch, elements});
  auto output = at::empty_like(input);
  auto operation = [&] { output.copy_(input, true); };
  return finish("aten_nested_clone_cpu", "fe71788a5bee6f5d", output,
                input_host, operation, nullptr, nullptr, "B=724_N=5793");
}

int gemm_fixture(const std::string &kernel) {
  if (kernel == "aten_cpu_blas_gemm_cpu") {
    constexpr int64_t m = 1374, n = 1832, k = 2290;
    std::vector<float> a_host(m * k, 1.0f);
    auto b_host = fixture_values(k * n);
    std::vector<float> expected(m * n);
    std::vector<float> column_sum(n, 0.0f);
    for (int64_t inner = 0; inner < k; ++inner)
      for (int64_t column = 0; column < n; ++column)
        column_sum[column] += b_host[inner * n + column];
    for (int64_t row = 0; row < m; ++row)
      std::copy(column_sum.begin(), column_sum.end(),
                expected.begin() + row * n);
    auto a = device_tensor(a_host).reshape({m, k});
    auto b = device_tensor(b_host).reshape({k, n});
    auto output = at::empty({m, n}, a.options());
    auto operation = [&] { at::mm_out(output, a, b); };
    return finish(kernel.c_str(), "acf1ac1e761ae7e6", output, expected,
                  operation, nullptr, nullptr, "K=2290_M=1374_N=1832");
  }
  constexpr int64_t batch = 52, m = 208, n = 259, k = 311;
  std::vector<float> a_host(batch * m * k, 1.0f);
  auto b_host = fixture_values(batch * k * n);
  std::vector<float> expected(batch * m * n);
  for (int64_t q = 0; q < batch; ++q) {
    std::vector<float> column_sum(n, 0.0f);
    for (int64_t inner = 0; inner < k; ++inner)
      for (int64_t column = 0; column < n; ++column)
        column_sum[column] += b_host[(q * k + inner) * n + column];
    for (int64_t row = 0; row < m; ++row)
      std::copy(column_sum.begin(), column_sum.end(),
                expected.begin() + (q * m + row) * n);
  }
  auto a = device_tensor(a_host).reshape({batch, m, k});
  auto b = device_tensor(b_host).reshape({batch, k, n});
  auto output = at::empty({batch, m, n}, a.options());
  auto operation = [&] { at::bmm_out(output, a, b); };
  const char *recipe = kernel == "aten_cpu_blas_gemm_batched_cpu"
                           ? "57f08d687ae9d16b"
                           : "eb3e1fa75831476a";
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, "B=52_K=311_M=208_N=259");
}

int nested_squeeze() {
  constexpr int64_t batch = 724;
  constexpr int64_t elements = 5793;
  auto input_host = fixture_values(batch * elements);
  auto input = device_tensor(input_host).reshape({batch, 1, elements});
  auto source_view = input.squeeze(1);
  auto output = at::empty({batch, elements}, input.options());
  auto operation = [&] { output.copy_(source_view, true); };
  return finish("aten_nested_squeeze_cpu", "f6aafb8065321582", output,
                input_host, operation, nullptr, nullptr, "B=724_N=5793");
}

int stack_serial() {
  constexpr int64_t tensors = 51;
  constexpr int64_t rows = 203;
  constexpr int64_t elements = 405;
  auto input_host = fixture_values(tensors * rows * elements);
  std::vector<float> expected(input_host.size());
  for (int64_t t = 0; t < tensors; ++t)
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t element = 0; element < elements; ++element)
        expected[(row * tensors + t) * elements + element] =
            input_host[(t * rows + row) * elements + element];
  auto input = device_tensor(input_host).reshape({tensors, rows, elements});
  auto source_view = input.permute({1, 0, 2});
  auto output = at::empty({rows, tensors, elements}, input.options());
  auto operation = [&] { output.copy_(source_view, true); };
  return finish("aten_stack_serial_cpu", "b4586baaae66626e", output,
                expected, operation, nullptr, nullptr, "K=405_R=203_T=51");
}

int structured_matmul(const std::string &kernel) {
  int64_t batch, m, n, k;
  bool broadcast_rhs = false;
  if (kernel == "aten_flatten_nd_linear_cpu") {
    batch = 80;
    m = 161;
    n = 241;
    k = 322;
    broadcast_rhs = true;
  } else if (kernel == "aten_nested_bmm_cpu") {
    batch = 83;
    m = 165;
    n = 206;
    k = 248;
  } else {
    batch = 89;
    m = 178;
    n = 222;
    k = 266;
    broadcast_rhs = true;
  }
  std::vector<float> a_host(batch * m * k, 1.0f);
  auto b_host = fixture_values((broadcast_rhs ? 1 : batch) * k * n);
  std::vector<float> expected(batch * m * n);
  for (int64_t q = 0; q < batch; ++q) {
    std::vector<float> column_sum(n, 0.0f);
    const int64_t rhs_batch = broadcast_rhs ? 0 : q;
    for (int64_t inner = 0; inner < k; ++inner)
      for (int64_t column = 0; column < n; ++column)
        column_sum[column] +=
            b_host[(rhs_batch * k + inner) * n + column];
    for (int64_t row = 0; row < m; ++row)
      std::copy(column_sum.begin(), column_sum.end(),
                expected.begin() + (q * m + row) * n);
  }
  auto a = device_tensor(a_host).reshape({batch, m, k});
  auto output = at::empty({batch, m, n}, a.options());
  std::function<void()> operation;
  if (kernel == "aten_flatten_nd_linear_cpu") {
    auto flattened_a = a.reshape({batch * m, k});
    auto b = device_tensor(b_host).reshape({k, n});
    auto flattened_output = output.reshape({batch * m, n});
    operation = [&, flattened_a, b, flattened_output]() mutable {
      at::mm_out(flattened_output, flattened_a, b);
    };
  } else if (kernel == "aten_nested_bmm_cpu") {
    auto b = device_tensor(b_host).reshape({batch, k, n});
    operation = [&, b] { at::bmm_out(output, a, b); };
  } else {
    auto b = device_tensor(b_host).reshape({k, n});
    operation = [&, b] { at::matmul_out(output, a, b); };
  }
  const char *recipe = kernel == "aten_flatten_nd_linear_cpu"
                           ? "0b73d0504d7a7d38"
                           : (kernel == "aten_nested_bmm_cpu"
                                  ? "052a884874109c4d"
                                  : "7a9f8d11436e5a37");
  const std::string shape =
      kernel == "aten_flatten_nd_linear_cpu"
          ? "B=80_K=322_M=161_N=241"
          : (kernel == "aten_nested_bmm_cpu"
                 ? "B=83_K=248_M=165_N=206"
                 : "B=89_K=266_M=178_N=222");
  return finish(kernel.c_str(), recipe, output, expected, operation, nullptr,
                nullptr, shape.c_str());
}

int sparse_norm() {
  constexpr int64_t n = 2097152;
  auto input_host = fixture_values(n);
  double squared_sum = 0.0;
  for (float value : input_host)
    squared_sum += static_cast<double>(value) * value;
  auto input = device_tensor(input_host);
  auto output = at::empty({}, input.options());
  auto operation = [&] {
    at::linalg_vector_norm_out(output, input, 2.0, std::nullopt, false,
                               std::nullopt);
  };
  return finish("aten_sparse_norm_cpu", "2ed1a182be1a2c6b", output,
                {static_cast<float>(std::sqrt(squared_sum))}, operation,
                nullptr, nullptr, "N=2097152");
}

int nested_reduce(const std::string &kernel) {
  constexpr int64_t batch = 724;
  constexpr int64_t elements = 5793;
  std::vector<int32_t> lengths_host(batch);
  for (int64_t b = 0; b < batch; ++b)
    lengths_host[b] = static_cast<int32_t>(elements - (b % 31));
  auto lengths = device_int_tensor(lengths_host).reshape({batch, 1});
  std::vector<int32_t> columns_host(elements);
  for (int64_t i = 0; i < elements; ++i)
    columns_host[i] = static_cast<int32_t>(i);
  auto columns = device_int_tensor(columns_host).reshape({1, elements});
  auto mask = at::empty({batch, elements}, lengths.options().dtype(at::kBool));
  if (kernel == "aten_nested_all_cpu") {
    std::vector<int32_t> input_host(batch * elements);
    std::vector<int32_t> expected(batch, 1);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t i = 0; i < elements; ++i) {
        const int32_t value = ((b * 17 + i * 13) % 97) != 0;
        input_host[b * elements + i] = value;
        if (i < lengths_host[b])
          expected[b] &= value != 0;
      }
    auto input = device_int_tensor(input_host).reshape({batch, elements});
    auto nonzero = at::empty_like(mask);
    auto valid_or_identity = at::empty_like(mask);
    auto identity = at::ones_like(mask);
    auto reduced = at::empty({batch}, mask.options());
    auto output = at::empty({batch}, input.options());
    auto operation = [&, input, nonzero, valid_or_identity, identity,
                      reduced]() mutable {
      at::lt_out(mask, columns, lengths);
      at::ne_out(nonzero, input, 0);
      at::where_out(valid_or_identity, mask, nonzero, identity);
      at::all_out(reduced, valid_or_identity, 1, false);
      output.copy_(reduced, true);
    };
    return finish_int(kernel.c_str(), "39273232cbbc40e2", output, expected,
                      operation, "B=724_N=5793");
  }
  auto input_host = fixture_values(batch * elements);
  std::vector<float> expected(batch, 0.0f);
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t i = 0; i < lengths_host[b]; ++i)
      expected[b] += input_host[b * elements + i];
  auto input = device_tensor(input_host).reshape({batch, elements});
  auto masked = at::empty_like(input);
  auto zeros = at::zeros_like(input);
  auto output = at::empty({batch}, input.options());
  auto operation = [&] {
    at::lt_out(mask, columns, lengths);
    at::where_out(masked, mask, input, zeros);
    at::sum_out(output, masked, at::OptionalIntArrayRef({1}), false,
                std::nullopt);
  };
  return finish(kernel.c_str(), "132e688af89d8697", output, expected,
                operation, nullptr, nullptr, "B=724_N=5793");
}

int sumproduct_pair() {
  constexpr int64_t batch = 70, m = 141, k = 282, n = 211;
  std::vector<float> a_host(batch * m * k, 1.0f);
  auto b_host = fixture_values(batch * k * n);
  std::vector<float> expected(batch * m * n);
  for (int64_t q = 0; q < batch; ++q) {
    std::vector<float> column_sum(n, 0.0f);
    for (int64_t inner = 0; inner < k; ++inner)
      for (int64_t column = 0; column < n; ++column)
        column_sum[column] += b_host[(q * k + inner) * n + column];
    for (int64_t row = 0; row < m; ++row)
      std::copy(column_sum.begin(), column_sum.end(),
                expected.begin() + (q * m + row) * n);
  }
  auto a = device_tensor(a_host).reshape({batch, m, k});
  auto b = device_tensor(b_host).reshape({batch, k, n});
  auto output = at::empty({batch, m, n}, a.options());
  auto operation = [&] { at::bmm_out(output, a, b); };
  return finish("aten_sumproduct_pair_cpu", "b1d208ea182a8abc", output,
                expected, operation, nullptr, nullptr,
                "B=70_K=282_M=141_N=211");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s KERNEL [cuda|cpu]\n", argv[0]);
    return 2;
  }
  if (argc == 3) {
    if (std::string(argv[2]) == "cpu") {
      g_use_cuda = false;
      at::set_num_threads(1);
      at::set_num_interop_threads(1);
    } else if (std::string(argv[2]) != "cuda") {
      std::fprintf(stderr, "unsupported backend: %s\n", argv[2]);
      return 2;
    }
  }
  const std::string kernel = argv[1];
  if (is_whole_unary(kernel))
    return whole_unary(kernel);
  if (is_whole_elementwise(kernel))
    return whole_elementwise(kernel);
  if (is_whole_activation(kernel))
    return whole_activation(kernel);
  if (is_whole_activation_backward(kernel))
    return whole_activation_backward(kernel);
  if (kernel == "aten_hardswish_backward" || kernel == "aten_mish_backward")
    return whole_hardswish_or_mish_backward(kernel);
  if (kernel == "aten_copy_cpu" || kernel == "aten_copy_tensor_array_cpu" ||
      kernel == "aten_zeros_cpu")
    return whole_copy_or_zero(kernel);
  if (kernel == "aten_transpose_copy")
    return whole_transpose_copy();
  if (kernel == "aten_pixel_shuffle" ||
      kernel == "aten_pixel_shuffle_cpu_backend" ||
      kernel == "aten_pixel_unshuffle_cpu_backend")
    return whole_pixel_transform(kernel);
  if (kernel == "aten_angle_real" || kernel == "aten_clamp" ||
      kernel == "aten_logaddexp" || kernel == "aten_logaddexp2" ||
      kernel == "aten_mse_elementwise")
    return whole_special_pointwise(kernel);
  if (kernel == "aten_nansum_cpu")
    return whole_nansum();
  if (kernel == "aten_adaptive_avg_pool2d" ||
      kernel == "aten_adaptive_avg_pool3d" || kernel == "aten_avg_pool2d" ||
      kernel == "aten_avg_pool3d")
    return whole_average_pool(kernel);
  if (kernel == "aten_avg_pool2d_backward_cpu" ||
      kernel == "aten_avg_pool3d_backward_cpu")
    return whole_average_pool_backward(kernel);
  if (kernel == "aten_bmm")
    return whole_bmm();
  if (kernel == "aten_cross" || kernel == "aten_cross_cpu_backend")
    return whole_cross(kernel);
  if (kernel == "aten_cat_serial_cpu")
    return whole_cat();
  if (kernel == "aten_repeat_compute_cpu" ||
      kernel == "aten_repeat_tensor_shape_cpu")
    return whole_repeat(kernel);
  if (kernel == "aten_im2col")
    return whole_im2col();
  if (kernel == "aten_max_pool2d")
    return whole_max_pool2d();
  if (kernel == "aten_addmm" || kernel == "aten_mm" ||
      kernel == "aten_outer" || kernel == "aten_sum")
    return whole_double_linear_algebra(kernel);
  if (kernel == "aten_elu" || kernel == "aten_elu_backward" ||
      kernel == "aten_log_sigmoid_backward_cpu")
    return whole_elu_family(kernel);
  if (kernel == "aten_argmax_cpu" || kernel == "aten_argmin_cpu" ||
      kernel == "aten_count_nonzero_impl_cpu")
    return whole_integer_reduction(kernel);
  if (kernel == "aten_cumprod_cpu")
    return whole_cumprod();
  if (kernel == "aten_int_mm_out_cpu")
    return whole_int_mm();
  if (kernel == "aten_conv1d" || kernel == "aten_conv2d" ||
      kernel == "aten_conv_transpose2d" ||
      kernel == "aten_conv_transpose3d_cpu")
    return whole_convolution(kernel);
  if (kernel == "aten_nested_sum_backward_cpu")
    return whole_nested_sum_backward();
  if (kernel == "aten_sampled_addmm_sparse_csr_cpu")
    return whole_sampled_addmm_sparse_csr();
  if (kernel == "aten_sparse_csr_addmm_cpu")
    return whole_sparse_csr_addmm();
  if (kernel == "aten_clamp_cpu")
    return clamp_tensor();
  if (kernel == "aten_clamp_max_scalar_cpu" ||
      kernel == "aten_clamp_min_scalar_cpu" ||
      kernel == "aten_clamp_scalar_cpu")
    return clamp_scalar(kernel);
  if (kernel == "aten_masked_scale")
    return masked_scale();
  if (kernel == "aten_hardshrink" || kernel == "aten_hardtanh" ||
      kernel == "aten_softshrink")
    return shrink_or_hardtanh(kernel);
  if (kernel == "aten_hardtanh_backward" ||
      kernel == "aten_shrink_backward")
    return activation_backward(kernel);
  if (kernel == "aten_huber_elementwise" ||
      kernel == "aten_smooth_l1_elementwise")
    return loss_elementwise(kernel);
  if (kernel == "aten_huber_backward" || kernel == "aten_mse_backward" ||
      kernel == "aten_smooth_l1_backward")
    return loss_backward(kernel);
  if (kernel == "aten_lerp" || kernel == "aten_lerp_scalar" ||
      kernel == "aten_lerp_scalar_cpu")
    return lerp_case(kernel);
  if (kernel == "aten_logit" || kernel == "aten_logit_backward")
    return logit_case(kernel);
  if (kernel == "aten_threshold_backward")
    return threshold_backward();
  if (kernel == "aten_glu")
    return glu_forward();
  if (kernel == "aten_log_sigmoid_cpu")
    return log_sigmoid_forward();
  if (kernel == "aten_angle_complex_scalarized")
    return angle_complex();
  if (kernel == "aten_diff_cpu")
    return diff_forward();
  if (kernel == "aten_blas_axpy_cpu" || kernel == "aten_blas_copy_cpu" ||
      kernel == "aten_blas_scale_cpu")
    return blas_vector(kernel);
  if (kernel == "aten_narrow_copy_dense_cpu")
    return narrow_copy_dense();
  if (kernel == "aten_unbind_copy_cpu")
    return unbind_copy();
  if (kernel == "aten_add_clamp")
    return add_clamp();
  if (kernel == "aten_as_complex_cpu" ||
      kernel == "aten_complex_scalarized" ||
      kernel == "aten_conj_complex_scalarized" ||
      kernel == "aten_polar_scalarized")
    return scalarized_complex(kernel);
  if (kernel == "aten_round" || kernel == "aten_round_decimals")
    return round_away_from_zero(kernel);
  if (kernel == "aten_sparse_add_values_cpu" ||
      kernel == "aten_sparse_mul_cpu")
    return sparse_binary(kernel);
  if (kernel == "aten_renorm_scale_factor")
    return renorm_scale_factor();
  if (kernel == "aten_glu_backward" || kernel == "aten_glu_jvp")
    return glu_derivative(kernel);
  if (kernel == "aten_batch_norm_cpu_entry")
    return batch_norm_affine();
  if (kernel == "aten_dropout_feature_noise_cpu")
    return dropout_feature_noise();
  if (kernel == "aten_channel_shuffle")
    return channel_shuffle();
  if (kernel == "aten_allany_dims_cpu" ||
      kernel == "aten_and_reduce_cpu" || kernel == "aten_or_reduce_cpu")
    return logical_reduce(kernel);
  if (kernel == "aten_joint_scaling_cpu")
    return joint_scaling();
  if (kernel == "aten_channel_shuffle_cpu")
    return channel_shuffle_flat();
  if (kernel == "aten_convert_coo_to_csr_cpu" ||
      kernel == "aten_sparse_coo_to_csr_cpu")
    return coo_to_csr(kernel);
  if (kernel == "aten_compressed_block_convert_cpu")
    return compressed_block_convert();
  if (kernel == "aten_kron_impl_cpu")
    return kron_impl(kernel);
  if (kernel == "aten_nested_clone_cpu")
    return nested_clone();
  if (kernel == "aten_cpu_blas_gemm_cpu" ||
      kernel == "aten_cpu_blas_gemm_batched_cpu" ||
      kernel == "aten_cpu_blas_gemm_strided_batched_cpu")
    return gemm_fixture(kernel);
  if (kernel == "aten_nested_squeeze_cpu")
    return nested_squeeze();
  if (kernel == "aten_stack_serial_cpu")
    return stack_serial();
  if (kernel == "aten_flatten_nd_linear_cpu" ||
      kernel == "aten_nested_bmm_cpu" ||
      kernel == "aten_nested_matmul_broadcast_cpu")
    return structured_matmul(kernel);
  if (kernel == "aten_kron_out_cpu")
    return kron_impl(kernel);
  if (kernel == "aten_sparse_norm_cpu")
    return sparse_norm();
  if (kernel == "aten_nested_all_cpu" ||
      kernel == "aten_nested_sum_dim_cpu")
    return nested_reduce(kernel);
  if (kernel == "aten_sumproduct_pair_cpu")
    return sumproduct_pair();
  std::fprintf(stderr, "unsupported kernel: %s\n", argv[1]);
  return 2;
}
