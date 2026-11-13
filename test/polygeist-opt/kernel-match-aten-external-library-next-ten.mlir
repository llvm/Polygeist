// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_transpose3d_backward_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=TRANSPOSE-BACKWARD
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_dilated_convolution_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=DILATED
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_nested_bmm_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=NESTED-BMM
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_nested_matmul_broadcast_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=NESTED-BROADCAST

// TRANSPOSE-BACKWARD-LABEL: func.func @aten_conv_transpose3d_backward_cpu
// TRANSPOSE-BACKWARD: kernel.launch @cudnnConvolution3D_f32(
// TRANSPOSE-BACKWARD-NOT: linalg.generic
// TRANSPOSE-BACKWARD-NOT: affine.for
// TRANSPOSE-BACKWARD-NOT: scf.for

// DILATED-LABEL: func.func @aten_dilated_convolution_cpu
// DILATED: kernel.launch @cudnnConvolution2D_f32_dilated(
// DILATED-NOT: linalg.generic
// DILATED-NOT: affine.for
// DILATED-NOT: scf.for

// NESTED-BMM-LABEL: func.func @aten_nested_bmm_cpu
// NESTED-BMM: kernel.launch @cublasSgemm_strided_batched_nn_zero(
// NESTED-BMM-NOT: linalg.generic
// NESTED-BMM-NOT: affine.for
// NESTED-BMM-NOT: scf.for

// NESTED-BROADCAST-LABEL: func.func @aten_nested_matmul_broadcast_cpu
// NESTED-BROADCAST: kernel.launch @cublasSgemm_strided_batched_broadcast_rhs(
// NESTED-BROADCAST-NOT: linalg.generic
// NESTED-BROADCAST-NOT: affine.for
// NESTED-BROADCAST-NOT: scf.for
