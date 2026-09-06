// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_csr_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/d0 + 1/d0 + 2/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: sed 's/to_tensor %arg4/to_tensor %arg3/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmv_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=ALIAS

// The complete tensorized affine/scf region is one CSR SpMV.  The negative
// case proves that merely looking like an indirect reduction is insufficient:
// the second row pointer must be exactly rowptr[row + 1].

// CHECK: %[[ROWS:.*]] = arith.constant 64 : index
// CHECK: kernel.launch @cusparseSpMV_CSR_f32_memref(%[[ROWS]], %arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK-NOT: scf.for
// CHECK-NOT: memref.copy

// BAD-NOT: kernel.launch @cusparseSpMV_CSR_f32_memref
// BAD: scf.for

// ALIAS-NOT: kernel.launch @cusparseSpMV_CSR_f32_memref
// ALIAS: scf.for
