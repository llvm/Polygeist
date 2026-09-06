// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_addmm_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=ADDMM
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_hspmm_cpu/debuf.mlir --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=HSPMM
// RUN: sed 's/linalg.yield %cst/linalg.yield %out/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmm_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=NONZERO
// RUN: sed 's/to_tensor %arg4/to_tensor %arg3/' %S/../../issues/aten_c_kernels/results/aten_sparse_addmm_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite 2>&1 | FileCheck %s --check-prefix=ALIAS

// ADDMM: kernel.launch @cusparseSpMM_COO_f32_memref
// ADDMM-NOT: linalg.generic
// ADDMM-NOT: affine.for
// ADDMM-NOT: memref.copy

// HSPMM: kernel.launch @cusparseSpMM_COO_f32_memref
// HSPMM-NOT: linalg.generic
// HSPMM-NOT: affine.for

// NONZERO-NOT: kernel.launch @cusparseSpMM_COO_f32_memref
// NONZERO: affine.for

// ALIAS-NOT: kernel.launch @cusparseSpMM_COO_f32_memref
// ALIAS: affine.for
