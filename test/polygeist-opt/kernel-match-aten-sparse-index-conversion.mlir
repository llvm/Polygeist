// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_convert_coo_to_csr_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=COO
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_coo_to_csr_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=COO
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_convert_csr_to_coo_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=CSR
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_matmul_csr_to_coo_cpu/debuf.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=CSR
// RUN: sed 's/arith.cmpi slt, %extracted_2, %5/arith.cmpi sgt, %extracted_2, %5/' %S/../../issues/aten_c_kernels/results/aten_convert_coo_to_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-ORDER
// RUN: sed 's/%10 = arith.addi/%10 = arith.subi/' %S/../../issues/aten_c_kernels/results/aten_convert_coo_to_csr_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-INCREMENT
// RUN: sed 's/into %arg6\[%6\]/into %arg6[%arg3]/' %S/../../issues/aten_c_kernels/results/aten_convert_csr_to_coo_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=BAD-INDEX
// RUN: sed 's/to_tensor %arg2/to_tensor %arg0/' %S/../../issues/aten_c_kernels/results/aten_convert_csr_to_coo_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin --enable-structured-rewrite | FileCheck %s --check-prefix=ALIAS

// COO: kernel.launch @cusparseXcoo2csr_i32_memref
// COO-NOT: affine.for
// COO-NOT: scf.while
// COO-NOT: memref.copy

// CSR: kernel.launch @cusparseXcsr2coo_i32_memref
// CSR-NOT: affine.for
// CSR-NOT: scf.while
// CSR-NOT: memref.copy

// BAD-ORDER-NOT: kernel.launch @cusparseXcoo2csr_i32_memref
// BAD-ORDER: scf.while

// BAD-INCREMENT-NOT: kernel.launch @cusparseXcoo2csr_i32_memref
// BAD-INCREMENT: scf.while

// BAD-INDEX-NOT: kernel.launch @cusparseXcsr2coo_i32_memref
// BAD-INDEX: tensor.insert

// ALIAS-NOT: kernel.launch @cusparseXcsr2coo_i32_memref
// ALIAS: affine.for
