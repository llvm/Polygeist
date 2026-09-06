// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_diff_cpu/debuf.mlir --only-kernel cubAdjacentDifference_f32_memref | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.subf %in, %in_2/arith.subf %in_2, %in/' %S/../../issues/aten_c_kernels/results/aten_diff_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --only-kernel cubAdjacentDifference_f32_memref --disable-pointwise-matching | FileCheck %s --check-prefix=BAD-ORDER
// RUN: sed 's/%0\[1\]/%0[0]/' %S/../../issues/aten_c_kernels/results/aten_diff_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --only-kernel cubAdjacentDifference_f32_memref --disable-pointwise-matching | FileCheck %s --check-prefix=BAD-OFFSET

// MATCH: kernel.launch @cubAdjacentDifference_f32_memref(%arg0, %arg1)
// MATCH-SAME: polygeist.fixed_extents = array<i64: 128>
// MATCH-NOT: linalg.generic
// MATCH-NOT: memref.copy

// BAD-ORDER-NOT: kernel.launch @cubAdjacentDifference_f32_memref
// BAD-ORDER: linalg.generic
// BAD-OFFSET-NOT: kernel.launch @cubAdjacentDifference_f32_memref
// BAD-OFFSET: linalg.generic
