// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/polybench_section42/ir/syrk/raised_debufferized.mlir > %t.syrk
// RUN: FileCheck %s --check-prefix=SYRK < %t.syrk
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/polybench_section42/ir/syr2k/raised_debufferized.mlir > %t.syr2k
// RUN: FileCheck %s --check-prefix=SYR2K < %t.syr2k

// This checks the canonical raised structure, not a benchmark-name rule: two
// masked lower-triangle steps, with the scale followed by a rank-k update.

// SYRK-LABEL: func.func @kernel_syrk
// SYRK: kernel.launch @cublasDsyrk
// SYRK-NOT: linalg.generic

// SYR2K-LABEL: func.func @kernel_syr2k
// SYR2K: kernel.launch @cublasDsyr2k
// SYR2K-NOT: linalg.generic
