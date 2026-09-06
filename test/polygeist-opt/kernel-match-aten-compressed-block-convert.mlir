// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_compressed_block_convert_cpu/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.remsi/arith.divsi/' %S/../../issues/aten_c_kernels/results/aten_compressed_block_convert_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-INDEX
// RUN: sed 's/arith.constant 4 : index/arith.constant 8 : index/' %S/../../issues/aten_c_kernels/results/aten_compressed_block_convert_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-BLOCK

// MATCH-LABEL: func.func @aten_compressed_block_convert_cpu
// MATCH: memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 4, 16, 4], strides: [256, 64, 4, 1]
// MATCH: kernel.launch @cutensorPermute_f32_r4_tensor
// MATCH-SAME: cutensor_input_modes = array<i64: 0, 2, 1, 3>
// MATCH-SAME: cutensor_output_modes = array<i64: 0, 1, 2, 3>
// MATCH-NOT: affine.for

// BAD-INDEX-LABEL: func.func @aten_compressed_block_convert_cpu
// BAD-INDEX-NOT: kernel.launch @cutensorPermute_f32_r4_tensor
// BAD-INDEX: affine.for

// BAD-BLOCK-LABEL: func.func @aten_compressed_block_convert_cpu
// BAD-BLOCK-NOT: kernel.launch @cutensorPermute_f32_r4_tensor
// BAD-BLOCK: affine.for
