/*
 * FP32 scalar-row specialization of ATen _vec_softmax's scalar tail.
 * Source: third_party/pytorch/aten/src/ATen/native/cpu/SoftMaxKernel.cpp,
 * lines 801-816; upstream file SHA256:
 * 8a49d0f097857b201af82c01c0d3da5edbc12fec9eb3bd8c496638069f67dcef.
 * PyTorch's BSD-style license is in third_party/pytorch/LICENSE.
 *
 * Adaptations: scalar_t=float; dim_stride=1; one contiguous row; c10::irange
 * becomes a C for-loop; std::max(a,b) becomes a < b ? b : a; the float
 * overload of std::exp becomes expf. dim_size is an int (positive and <=
 * INT_MAX). No arithmetic stages or accumulation types are changed.
 * The upstream C++ row is compiled independently by the reproduction driver.
 */
#include <math.h>

#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif

void NETLIB_SPEC(int n, float input[], float output[]) {
  float max_input = input[0];
  for (int d = 1; d < n; ++d)
    max_input = max_input < input[d] ? input[d] : max_input;

  float sum_data = 0;
  for (int d = 0; d < n; ++d) {
    output[d] = expf(input[d] - max_input);
    sum_data += output[d];
  }
  for (int d = 0; d < n; ++d)
    output[d] = output[d] / sum_data;
}
