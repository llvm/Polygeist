// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py --enable-structured-rewrite %S/../../issues/polybench_section42/ir/trisolv/raised_debufferized.mlir | FileCheck %s

// Verify whole-recurrence recognition on the tensor-carried form emitted by
// the canonical Section 4.2 raising pipeline. This is structural matching,
// not a benchmark-name rewrite.

// CHECK-LABEL: func.func @kernel_trisolv
// CHECK: kernel.launch @cublasDtrsvLowerRowMajor_memref
// CHECK-NOT: affine.for
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref.copy
// CHECK: return
