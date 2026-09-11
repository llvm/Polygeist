// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_csr_addmm_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/d0 + 1/d0 + 2/' %S/../../issues/aten_c_kernels/results/aten_sparse_csr_addmm_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: sed 's/to_tensor %arg4/to_tensor %arg3/' %S/../../issues/aten_c_kernels/results/aten_sparse_csr_addmm_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=ALIAS

// CHECK: %[[ROWS:.*]] = arith.constant 64 : index
// CHECK: kernel.launch @cusparseSpMM_CSR_f32_memref(%[[ROWS]], %arg0, %arg1, %arg2,
// CHECK-NOT: scf.for
// CHECK-NOT: affine.for
// CHECK-NOT: memref.copy

// BAD-NOT: kernel.launch @cusparseSpMM_CSR_f32_memref
// BAD: scf.for

// ALIAS-NOT: kernel.launch @cusparseSpMM_CSR_f32_memref
// ALIAS: scf.for
