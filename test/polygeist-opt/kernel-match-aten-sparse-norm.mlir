// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_sparse_norm_cpu/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/math.sqrt/math.rsqrt/' %S/../../issues/aten_c_kernels/results/aten_sparse_norm_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=REJECT

// MATCH-LABEL: func.func @aten_sparse_norm_cpu
// MATCH: kernel.launch @cublasSnrm2_f32_memref(%arg0, %arg1)
// MATCH-NOT: linalg.generic
// MATCH-NOT: math.sqrt
// MATCH-NOT: memref.copy

// REJECT-LABEL: func.func @aten_sparse_norm_cpu
// REJECT-NOT: kernel.launch @cublasSnrm2_f32_memref
// REJECT: linalg.generic
