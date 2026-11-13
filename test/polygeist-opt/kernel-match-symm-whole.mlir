// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py --enable-structured-rewrite %S/../../issues/polybench_section42/ir/symm/raised_debufferized.mlir | FileCheck %s

// CHECK-LABEL: func.func @kernel_symm
// CHECK: kernel.launch @cublasDsymmLeftLowerRowMajor_memref
// CHECK-NOT: affine.for
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref.copy
// CHECK: return
