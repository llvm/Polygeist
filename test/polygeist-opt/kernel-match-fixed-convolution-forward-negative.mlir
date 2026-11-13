// Each mutation preserves the multiply-accumulate computation but violates a
// physical-layout or padding condition required by the fixed cuDNN wrapper.
// RUN: sed 's/d3 + d0, d1, d4/d3 + d0 * 2, d1, d4/' %S/../../issues/aten_c_kernels/results/aten_conv_tbc_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=TBC
// RUN: sed 's/d3 + d0, d4 + d1/d3 + d0 * 2, d4 + d1/' %S/../../issues/aten_c_kernels/results/aten_conv_transpose2d/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=TRANSPOSE
// RUN: sed 's/-d0 - d1 + 16/-d0 - d1 + 15/' %S/../../issues/aten_c_kernels/results/aten_depthwise_conv3x3_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=DEPTHWISE

// TBC-LABEL: func.func @aten_conv_tbc_cpu
// TBC-NOT: kernel.launch @cudnnConvolutionTBC_f32_memref
// TBC: linalg.generic
// TBC: linalg.generic

// TRANSPOSE-LABEL: func.func @aten_conv_transpose2d
// TRANSPOSE-NOT: kernel.launch @cudnnConvolutionTranspose2D_f32_memref
// TRANSPOSE: linalg.generic
// TRANSPOSE: linalg.generic

// DEPTHWISE-LABEL: func.func @aten_depthwise_conv3x3_cpu
// DEPTHWISE-NOT: kernel.launch @cudnnDepthwiseConvolution2D_f32_memref
// DEPTHWISE: linalg.generic
// DEPTHWISE: linalg.generic
