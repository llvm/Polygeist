// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sampled_addmm_sparse_csr_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/%13 = arith.mulf %arg6, %extracted_2/%13 = arith.mulf %arg5, %extracted_2/' %S/../../issues/aten_c_kernels/results/aten_sampled_addmm_sparse_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-SCALE
// RUN: sed 's/memref.load %arg3\[%arg8, %17\]/memref.load %arg3[%17, %arg8]/' %S/../../issues/aten_c_kernels/results/aten_sampled_addmm_sparse_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-LAYOUT
// RUN: sed 's/to_tensor %arg7/to_tensor %arg2/' %S/../../issues/aten_c_kernels/results/aten_sampled_addmm_sparse_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=ALIAS

// CHECK: %[[ROWS:.*]] = arith.constant 16 : index
// CHECK: %[[A:.*]] = memref.cast %arg3 : memref<?x32xf32> to memref<?x?xf32>
// CHECK: %[[B:.*]] = memref.cast %arg4 : memref<?x24xf32> to memref<?x?xf32>
// CHECK: kernel.launch @cusparseSDDMM_CSR_f32_memref(%[[ROWS]], %arg0, %arg1, %arg2, %[[A]], %[[B]], %arg5, %arg6, %arg7)
// CHECK-NOT: linalg.generic
// CHECK-NOT: affine.for
// CHECK-NOT: scf.while
// CHECK-NOT: memref.copy

// BAD-SCALE-NOT: kernel.launch @cusparseSDDMM_CSR_f32_memref
// BAD-SCALE: scf.while

// BAD-LAYOUT-NOT: kernel.launch @cusparseSDDMM_CSR_f32_memref
// BAD-LAYOUT: linalg.generic

// ALIAS-NOT: kernel.launch @cusparseSDDMM_CSR_f32_memref
// ALIAS: affine.for
