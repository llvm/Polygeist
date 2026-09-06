// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_quant_col_offsets_cpu/debuf.mlir --only-kernel cubQuantColOffsets_i8_i32_memref | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.extsi %in/arith.extui %in/' %S/../../issues/aten_c_kernels/results/aten_quant_col_offsets_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --only-kernel cubQuantColOffsets_i8_i32_memref | FileCheck %s --check-prefix=BAD-EXTEND
// RUN: sed 's/(d1, d0)/(d0, d1)/' %S/../../issues/aten_c_kernels/results/aten_quant_col_offsets_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --only-kernel cubQuantColOffsets_i8_i32_memref | FileCheck %s --check-prefix=BAD-MAP
// RUN: sed 's/arith.subi %in, %2/arith.addi %in, %2/' %S/../../issues/aten_c_kernels/results/aten_quant_col_offsets_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --only-kernel cubQuantColOffsets_i8_i32_memref | FileCheck %s --check-prefix=BAD-OFFSET

// MATCH: kernel.launch @cubQuantColOffsets_i8_i32_memref(%arg0, %2, %arg2)
// MATCH-SAME: polygeist.fixed_extents = array<i64: 64, 48>
// MATCH-NOT: linalg.generic
// MATCH-NOT: memref.copy

// BAD-EXTEND-NOT: kernel.launch @cubQuantColOffsets_i8_i32_memref
// BAD-EXTEND: linalg.generic
// BAD-MAP-NOT: kernel.launch @cubQuantColOffsets_i8_i32_memref
// BAD-MAP: linalg.generic
// BAD-OFFSET-NOT: kernel.launch @cubQuantColOffsets_i8_i32_memref
// BAD-OFFSET: linalg.generic
