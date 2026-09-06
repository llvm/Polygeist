// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_avg_pool3d/debuf.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_adaptive_avg_pool3d/debuf.mlir | FileCheck %s --check-prefix=ADAPTIVE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_avg_pool2d_cpu/debuf.mlir | FileCheck %s --check-prefix=BLOCK2D
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_avg_pool3d_cpu/debuf.mlir | FileCheck %s --check-prefix=BLOCK3D

// CHECK-LABEL: func.func @aten_avg_pool3d
// CHECK-NOT: linalg.generic
// CHECK: arith.constant 4 : i32
// CHECK: arith.constant 3 : i32
// CHECK: arith.constant 2 : i32
// CHECK: arith.constant 8 : i32
// CHECK: kernel.launch @cudnnAveragePool_f32_r5(
// CHECK-SAME: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: memref.copy

// ADAPTIVE-LABEL: func.func @aten_adaptive_avg_pool3d
// ADAPTIVE-NOT: linalg.generic
// ADAPTIVE: kernel.launch @cudnnAveragePool_f32_r5(
// ADAPTIVE-NOT: tensor.insert_slice
// ADAPTIVE-NOT: memref.copy

// The flattened CPU fixtures currently reach matching after their window
// accumulation has disappeared into uninitialized stack temporaries.  Do not
// manufacture a library mapping from the function name alone.
// BLOCK2D-LABEL: func.func @aten_avg_pool2d_cpu
// BLOCK2D-NOT: kernel.launch @cudnnAveragePool
// BLOCK3D-LABEL: func.func @aten_avg_pool3d_cpu
// BLOCK3D-NOT: kernel.launch @cudnnAveragePool
