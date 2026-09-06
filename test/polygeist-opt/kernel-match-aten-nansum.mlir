// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_nansum_cpu/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.cmpf oeq/arith.cmpf one/' %S/../../issues/aten_c_kernels/results/aten_nansum_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=REJECT

// MATCH-LABEL: func.func @aten_nansum_cpu
// MATCH: kernel.launch @cubSegmentedNanSum_f32_memref(%arg0, %arg1)
// MATCH-SAME: polygeist.fixed_extents = array<i64: 16, 64>
// MATCH-NOT: linalg.generic
// MATCH-NOT: memref.copy

// REJECT-LABEL: func.func @aten_nansum_cpu
// REJECT-NOT: kernel.launch @cubSegmentedNanSum_f32_memref
// REJECT: linalg.generic
