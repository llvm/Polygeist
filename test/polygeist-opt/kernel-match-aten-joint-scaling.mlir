// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_joint_scaling_cpu/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed '0,/arith.cmpf ogt/s//arith.cmpf olt/' %S/../../issues/aten_c_kernels/results/aten_joint_scaling_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=REJECT

// MATCH-LABEL: func.func @aten_joint_scaling_cpu
// MATCH: kernel.launch @cublasJointMaxAbsProduct_f32_memref(%arg0, %arg1, %arg2)
// MATCH-NOT: linalg.generic
// MATCH-NOT: memref.copy

// REJECT-LABEL: func.func @aten_joint_scaling_cpu
// REJECT-NOT: kernel.launch @cublasJointMaxAbsProduct_f32_memref
// REJECT: linalg.generic
