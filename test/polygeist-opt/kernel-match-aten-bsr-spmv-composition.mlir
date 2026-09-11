// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_bsr_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/d0 + d1 \* 4/d0 + d1 * 3/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_bsr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-LAYOUT
// RUN: sed 's/to_tensor %arg4/to_tensor %arg3/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_bsr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=ALIAS

// CHECK: %[[ROWS:.*]] = arith.constant 16 : index
// CHECK: %[[DIM:.*]] = arith.constant 4 : index
// CHECK: kernel.launch @cusparseSpMM_BSR_f32_memref(%[[ROWS]], %[[DIM]], %arg0, %arg1,
// CHECK-NOT: linalg.generic
// CHECK-NOT: affine.for
// CHECK-NOT: scf.for
// CHECK-NOT: memref.copy

// BAD-LAYOUT-NOT: kernel.launch @cusparseSpMM_BSR_f32_memref
// BAD-LAYOUT: affine.for

// ALIAS-NOT: kernel.launch @cusparseSpMM_BSR_f32_memref
// ALIAS: affine.for
