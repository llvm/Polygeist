// RUN: sed 's/arith.cmpf oeq/arith.cmpf olt/' %S/../../issues/aten_c_kernels/results/aten_addr_elementwise/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=ADDR
// RUN: sed 's/0.000000e+00/1.000000e+00/' %S/../../issues/aten_c_kernels/results/aten_bf16_dot_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=DOT
// RUN: sed 's/2.560000e+02/2.550000e+02/' %S/../../issues/aten_c_kernels/results/aten_binary_cross_entropy/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BCE
// RUN: sed 's/%3#1/%3#0/' %S/../../issues/aten_c_kernels/results/aten_log_sigmoid_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=LOGSIGMOID
// RUN: sed 's/%inserted\[1\]/%inserted[2]/g' %S/../../issues/aten_c_kernels/results/aten_nested_batch_offsets_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=OFFSETS
// RUN: sed 's/(d0, d2, d1, d3)/(d0, d1, d2, d3)/' %S/../../issues/aten_c_kernels/results/aten_transform_bias_rescale_qkv_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=QKV

// ADDR-NOT: kernel.launch @cudnnAddrElementwise_f32_memref
// ADDR: linalg.generic
// DOT-NOT: kernel.launch @cublasSdot_memref
// DOT: linalg.generic
// BCE-NOT: kernel.launch @cudnnBinaryCrossEntropyMean_f32_memref
// BCE: linalg.generic
// LOGSIGMOID-NOT: kernel.launch @cudnnLogSigmoid_f32_memref
// LOGSIGMOID: linalg.generic
// OFFSETS-NOT: kernel.launch @cubExclusiveSum1D_i32_memref
// OFFSETS: linalg.generic
// QKV-NOT: kernel.launch @cudnnTransformBiasRescaleQKV_f32_memref
// QKV: linalg.generic
