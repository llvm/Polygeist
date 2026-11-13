// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_kron_impl_cpu/debuf.mlir | FileCheck %s --check-prefix=IMPL
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_kron_out_cpu/debuf.mlir | FileCheck %s --check-prefix=OUT

// IMPL-LABEL: func.func @aten_kron_impl_cpu
// IMPL: %[[X:.*]] = memref.cast %arg0 : memref<?x12xf32> to memref<?x?xf32>
// IMPL: %[[Y:.*]] = memref.cast %arg1 : memref<?x10xf32> to memref<?x?xf32>
// IMPL: %[[OUTPUT:.*]] = memref.cast %arg2 : memref<?x120xf32> to memref<?x?xf32>
// IMPL: kernel.launch @cutensorKroneckerProduct2D_f32_memref(%[[X]], %[[Y]], %[[OUTPUT]])
// IMPL-NOT: linalg.generic
// IMPL-NOT: polygeist.submapInverse
// IMPL-NOT: memref.copy

// OUT-LABEL: func.func @aten_kron_out_cpu
// OUT: %[[X:.*]] = memref.cast %arg0 : memref<?x12xf32> to memref<?x?xf32>
// OUT: %[[Y:.*]] = memref.cast %arg1 : memref<?x10xf32> to memref<?x?xf32>
// OUT: %[[OUTPUT:.*]] = memref.cast %arg2 : memref<?x120xf32> to memref<?x?xf32>
// OUT: kernel.launch @cutensorKroneckerProduct2D_f32_memref(%[[X]], %[[Y]], %[[OUTPUT]])
// OUT-NOT: linalg.generic
// OUT-NOT: polygeist.submapInverse
// OUT-NOT: memref.copy
