// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_argmax_cpu/debuf.mlir | FileCheck %s --check-prefix=MAX
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_argmin_cpu/debuf.mlir | FileCheck %s --check-prefix=MIN
// RUN: sed 's/arith.addi %0, %c1/arith.addi %0, %c32/' %S/../../issues/aten_c_kernels/results/aten_argmax_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-INDEX
// RUN: sed 's/arith.cmpf ogt/arith.cmpf oge/' %S/../../issues/aten_c_kernels/results/aten_argmax_cpu/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-TIE

// MAX: kernel.launch @cubSegmentedArgMax_f32_i32_memref(%arg0, %arg1)
// MAX-NOT: linalg.generic
// MIN: kernel.launch @cubSegmentedArgMin_f32_i32_memref(%arg0, %arg1)
// MIN-NOT: linalg.generic
// BAD-INDEX-NOT: kernel.launch @cubSegmentedArgMax_f32_i32_memref
// BAD-INDEX: linalg.generic
// BAD-TIE-NOT: kernel.launch @cubSegmentedArgMax_f32_i32_memref
// BAD-TIE: linalg.generic
