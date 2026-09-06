// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_embedding_bag_counts_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.constant 1 : i32/arith.constant 2 : i32/' %S/../../issues/aten_c_kernels/results/aten_embedding_bag_counts_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-INCREMENT
// RUN: sed 's/tensor.extract %1\[%arg2\]/tensor.extract %arg3[%arg2]/' %S/../../issues/aten_c_kernels/results/aten_embedding_bag_counts_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-INPUT
// RUN: sed 's/arith.constant 0 : i32/arith.constant 1 : i32/' %S/../../issues/aten_c_kernels/results/aten_embedding_bag_counts_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-ZERO

// MATCH: kernel.launch @cubHistogramEvenI32ShiftZero_memref(%arg0, %arg1,
// MATCH-SAME: polygeist.fixed_extents = array<i64: 512>
// MATCH-NOT: linalg.generic
// MATCH-NOT: affine.for
// MATCH-NOT: memref.copy

// BAD-INCREMENT-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// BAD-INCREMENT: affine.for
// BAD-INPUT-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// BAD-INPUT: affine.for
// BAD-ZERO-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// BAD-ZERO: affine.for
