// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_tbc_backward_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=TBC
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_transpose3d_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_transpose3d_grad_weight_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=TRANSPOSE-WEIGHT
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_slow_conv3d_backward_input_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=SLOW-INPUT
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_slow_conv3d_backward_weight_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=SLOW-WEIGHT

// TBC: memref.cast %arg0
// TBC: kernel.launch @cudnnConvolutionTBCBackward_f32_memref(%fixed_conv_0_0, %fixed_conv_0_1, %fixed_conv_0_2)
// TRANSPOSE: memref.cast %arg0
// TRANSPOSE: kernel.launch @cudnnConvolutionTranspose3D_f32_memref(%fixed_conv_0_0, %fixed_conv_0_1, %fixed_conv_0_2)
// TRANSPOSE-WEIGHT: %fixed_conv_0_0 = memref.cast %arg1
// TRANSPOSE-WEIGHT: %fixed_conv_0_1 = memref.cast %arg0
// TRANSPOSE-WEIGHT: kernel.launch @cudnnConvolutionBackwardFilter3D_f32_memref(%fixed_conv_0_0, %fixed_conv_0_1, %fixed_conv_0_2)
// SLOW-INPUT-NOT: linalg.fill
// SLOW-INPUT: kernel.launch @cudnnConvolutionTranspose3D_f32_memref(%fixed_conv_0_0, %fixed_conv_0_1, %fixed_conv_0_2)
// SLOW-WEIGHT: kernel.launch @cudnnConvolutionBackwardFilter3D_f32_memref(%fixed_conv_0_0, %fixed_conv_0_1, %fixed_conv_0_2)
