// RUN: %python %S/../../scripts/correctness/gen_wrapper.py --cuda-timing %s kernel_probe | FileCheck %s

void kernel_probe(int n, double A[n]) {}

// CHECK: polygeist_cublas_time_begin();
// CHECK: kernel_probe_impl(
// CHECK: double polygeist_device_ms = polygeist_cublas_time_end_ms();
// CHECK: POLYGEIST_DEVICE_TIMING kernel=kernel_probe device_ms=%.6f
