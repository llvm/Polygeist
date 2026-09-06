// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_allany_dims_cpu/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/arith.cmpi ne, %arg1/arith.cmpi eq, %arg1/' %S/../../issues/aten_c_kernels/results/aten_allany_dims_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-FLAG
// RUN: sed '0,/arith.constant true/s//arith.constant false/' %S/../../issues/aten_c_kernels/results/aten_allany_dims_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-ANY

// MATCH-LABEL: func.func @aten_allany_dims_cpu
// MATCH: kernel.launch @cubSegmentedLogicalSelect_i32_memref(%arg0, %arg0, %arg1, %arg2)
// MATCH-NOT: linalg.generic

// BAD-FLAG-LABEL: func.func @aten_allany_dims_cpu
// BAD-FLAG-NOT: kernel.launch @cubSegmentedLogicalSelect_i32_memref
// BAD-FLAG: linalg.generic

// BAD-ANY-LABEL: func.func @aten_allany_dims_cpu
// BAD-ANY-NOT: kernel.launch @cubSegmentedLogicalSelect_i32_memref
// BAD-ANY: linalg.generic
