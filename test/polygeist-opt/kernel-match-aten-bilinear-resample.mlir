// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %S/../../issues/aten_c_kernels/results/aten_upsample_bilinear2d/debuf.mlir | FileCheck %s --check-prefix=MATCH
// RUN: sed 's/5.000000e-01/2.500000e-01/' %S/../../issues/aten_c_kernels/results/aten_upsample_bilinear2d/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-SCALE
// RUN: sed 's/memref<?x3x8x8xf32>/memref<?x3x8x7xf32>/g' %S/../../issues/aten_c_kernels/results/aten_upsample_bilinear2d/debuf.mlir | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD-SHAPE

// MATCH: kernel.launch @cudnnBilinearUpsample2x_f32_r4
// MATCH-NOT: linalg.generic

// BAD-SCALE-NOT: kernel.launch @cudnnBilinearUpsample2x_f32_r4
// BAD-SCALE: linalg.generic

// BAD-SHAPE-NOT: kernel.launch @cudnnBilinearUpsample2x_f32_r4
// BAD-SHAPE: linalg.generic
