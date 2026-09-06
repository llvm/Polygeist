// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_bf16_gemv_trans_cpu/debuf.mlir | FileCheck %s --check-prefix=GEMV
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sum/debuf.mlir | FileCheck %s --check-prefix=SUM64
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sum_cpu_backend/debuf.mlir | FileCheck %s --check-prefix=SUM32
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_min_values_cpu/debuf.mlir | FileCheck %s --check-prefix=MIN
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_max_values_cpu/debuf.mlir | FileCheck %s --check-prefix=MAX
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_bilinear_cpu/debuf.mlir | FileCheck %s --check-prefix=BILINEAR
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_trilinear_cpu/debuf.mlir | FileCheck %s --check-prefix=TRILINEAR
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_and_reduce_cpu/debuf.mlir | FileCheck %s --check-prefix=AND
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_count_nonzero_impl_cpu/debuf.mlir | FileCheck %s --check-prefix=COUNT
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_cumprod_cpu/debuf.mlir | FileCheck %s --check-prefix=CUMPROD
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_max_pool2d/debuf.mlir | FileCheck %s --check-prefix=MAXPOOL
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_avg_pool2d_backward_cpu/debuf.mlir | FileCheck %s --check-prefix=AVGPOOL2DBWD
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_avg_pool3d_backward_cpu/debuf.mlir | FileCheck %s --check-prefix=AVGPOOL3DBWD

// GEMV: kernel.launch @cublasSgemvTZero_memref
// GEMV-SAME: polygeist.fixed_extents = array<i64: 64, 128>
// GEMV-NOT: linalg.generic
// SUM64: kernel.launch @cubSegmentedSum_f64_memref
// SUM64-SAME: polygeist.fixed_extents = array<i64: 16, 64>
// SUM64-NOT: linalg.generic
// SUM32: kernel.launch @cubSegmentedSum_f32_memref
// SUM32-NOT: linalg.generic
// MIN: kernel.launch @cubSegmentedMin_f32_memref
// MIN-SAME: polygeist.fixed_extents = array<i64: 32, 64>
// MIN-NOT: linalg.generic
// MAX: kernel.launch @cubSegmentedMax_f32_memref
// MAX-NOT: linalg.generic
// BILINEAR: kernel.launch @cutensornetNetwork_f32_n3_aten
// BILINEAR-SAME: network_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// BILINEAR-SAME: polygeist.fixed_operand_extents = array<i64: 8, 16, 24, 16, 20, 8, 20, 8, 24>
// BILINEAR-NOT: linalg.generic
// TRILINEAR: kernel.launch @cutensornetNetwork_f32_n3_aten
// TRILINEAR-SAME: network_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// TRILINEAR-NOT: linalg.generic
// AND: kernel.launch @cubSegmentedLogicalAnd_i32_memref
// AND-NOT: linalg.generic
// COUNT: kernel.launch @cubSegmentedCountNonzero2D_f32_tensor
// COUNT-NOT: linalg.generic
// CUMPROD: kernel.launch @cubSegmentedInclusiveProduct2D_f32_tensor
// CUMPROD-NOT: linalg.generic
// MAXPOOL: kernel.launch @cudnnMaxPoolFwd_batched
// MAXPOOL-NOT: linalg.generic
// AVGPOOL2DBWD: arith.constant 5 : i32
// AVGPOOL2DBWD: arith.constant 2 : i32
// AVGPOOL2DBWD: arith.constant 6 : i32
// AVGPOOL2DBWD: arith.constant 7 : i32
// AVGPOOL2DBWD: kernel.launch @cudnnAveragePool_f32_flat2
// AVGPOOL2DBWD-NOT: linalg.generic
// AVGPOOL2DBWD-NOT: memref.copy
// AVGPOOL3DBWD: arith.constant 5 : i32
// AVGPOOL3DBWD: arith.constant 3 : i32
// AVGPOOL3DBWD: arith.constant 6 : i32
// AVGPOOL3DBWD: arith.constant 7 : i32
// AVGPOOL3DBWD: arith.constant 8 : i32
// AVGPOOL3DBWD: arith.constant 4 : i32
// AVGPOOL3DBWD: kernel.launch @cudnnAveragePool_f32_flat2
// AVGPOOL3DBWD-NOT: linalg.generic
// AVGPOOL3DBWD-NOT: memref.copy
