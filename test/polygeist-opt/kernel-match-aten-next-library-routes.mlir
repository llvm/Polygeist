// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_bf16_gemv_trans_cpu/debuf.mlir | FileCheck %s --check-prefix=GEMV
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sum/debuf.mlir | FileCheck %s --check-prefix=SUM64
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sum_cpu_backend/debuf.mlir | FileCheck %s --check-prefix=SUM32
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_min_values_cpu/debuf.mlir | FileCheck %s --check-prefix=MIN
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_max_values_cpu/debuf.mlir | FileCheck %s --check-prefix=MAX
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_bilinear_cpu/debuf.mlir | FileCheck %s --check-prefix=BILINEAR
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_trilinear_cpu/debuf.mlir | FileCheck %s --check-prefix=TRILINEAR

// GEMV: kernel.launch @cublasSgemvTZero_memref
// GEMV-NOT: linalg.generic
// SUM64: kernel.launch @cubSegmentedSum_f64_memref
// SUM64-NOT: linalg.generic
// SUM32: kernel.launch @cubSegmentedSum_f32_memref
// SUM32-NOT: linalg.generic
// MIN: kernel.launch @cubSegmentedMin_f32_memref
// MIN-NOT: linalg.generic
// MAX: kernel.launch @cubSegmentedMax_f32_memref
// MAX-NOT: linalg.generic
// BILINEAR: kernel.launch @cutensornetNetwork_f32_n3_aten
// BILINEAR-SAME: network_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// BILINEAR-NOT: linalg.generic
// TRILINEAR: kernel.launch @cutensornetNetwork_f32_n3_aten
// TRILINEAR-SAME: network_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// TRILINEAR-NOT: linalg.generic
