// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_addr_elementwise/debuf.mlir | FileCheck %s --check-prefix=ADDR
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_bf16_dot_cpu/debuf.mlir | FileCheck %s --check-prefix=DOT
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_binary_cross_entropy/debuf.mlir | FileCheck %s --check-prefix=BCE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_log_sigmoid_cpu/debuf.mlir | FileCheck %s --check-prefix=LOGSIGMOID
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_nested_batch_offsets_cpu/debuf.mlir | FileCheck %s --check-prefix=OFFSETS
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_transform_bias_rescale_qkv_cpu/debuf.mlir | FileCheck %s --check-prefix=QKV

// ADDR: kernel.launch @cudnnAddrElementwise_f32_memref(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, f32, f32, memref<?xf32>) -> ()
// ADDR-NOT: linalg.generic

// DOT: kernel.launch @cublasSdot_memref(%arg0, %arg1, %arg2) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
// DOT-NOT: linalg.generic

// BCE: kernel.launch @cudnnBinaryCrossEntropyMean_f32_memref(%arg0, %arg1, %arg2) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
// BCE-NOT: arith.divf
// BCE-NOT: linalg.generic

// LOGSIGMOID: kernel.launch @cudnnLogSigmoid_f32_memref(%arg0, %arg1, %arg2) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
// LOGSIGMOID-NOT: memref.copy
// LOGSIGMOID-NOT: linalg.generic

// OFFSETS: kernel.launch @cubExclusiveSum1D_i32_memref(%arg0, %arg1) : (memref<?xi32>, memref<?xi32>) -> ()
// OFFSETS-NOT: tensor.insert_slice
// OFFSETS-NOT: linalg.generic

// QKV: memref.cast %arg0 : memref<?x16x3x4x8xf32> to memref<?x?x?x?x?xf32>
// QKV: kernel.launch @cudnnTransformBiasRescaleQKV_f32_memref(%aten_qkv_5_0, %aten_qkv_5_1, %arg2, %aten_qkv_5_2, %aten_qkv_5_3, %aten_qkv_5_4)
// QKV-NOT: memref.copy
// QKV-NOT: linalg.generic
