// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_transpose2d/debuf.mlir | FileCheck %s --check-prefix=TRANSPOSE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_conv_tbc_cpu/debuf.mlir | FileCheck %s --check-prefix=TBC
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_depthwise_conv3x3_cpu/debuf.mlir | FileCheck %s --check-prefix=DEPTHWISE

// TRANSPOSE-LABEL: func.func @aten_conv_transpose2d
// TRANSPOSE: kernel.launch @cudnnConvolutionTranspose2D_f32_memref(
// TRANSPOSE-NOT: linalg.generic
// TRANSPOSE-NOT: polygeist.submapInverse
// TRANSPOSE-NOT: memref.copy

// TBC-LABEL: func.func @aten_conv_tbc_cpu
// TBC: kernel.launch @cudnnConvolutionTBC_f32_memref(
// TBC-NOT: linalg.generic
// TBC-NOT: tensor.insert_slice
// TBC-NOT: memref.copy

// DEPTHWISE-LABEL: func.func @aten_depthwise_conv3x3_cpu
// DEPTHWISE: kernel.launch @cudnnDepthwiseConvolution2D_f32_memref(
// DEPTHWISE-NOT: linalg.generic
// DEPTHWISE-NOT: affine.apply
// DEPTHWISE-NOT: memref.copy
