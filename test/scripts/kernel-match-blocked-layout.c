// RUN: cgeist %s --function=split_layout --raise-scf-to-affine -S -o %t.source.mlir
// RUN: polygeist-opt %t.source.mlir --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --linalg-debufferize=use-multi-root=true -o %t.raised.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.raised.mlir --enable-structured-rewrite > %t.matched.mlir
// RUN: FileCheck %s < %t.matched.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/r % 3/r % 2/' %s > %t.invalid.c
// RUN: cgeist %t.invalid.c --function=split_layout --raise-scf-to-affine -S -o %t.invalid.mlir
// RUN: polygeist-opt %t.invalid.mlir --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --linalg-debufferize=use-multi-root=true -o %t.invalid.raised.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.invalid.raised.mlir --enable-structured-rewrite | FileCheck %s --check-prefix=CHECK-NEGATIVE

void split_layout(float input[24][40], float output[8][8][3][5]) {
  for (int r = 0; r < 24; ++r)
    for (int c = 0; c < 40; ++c)
      output[r / 3][c / 5][r % 3][c % 5] = input[r][c];
}
// CHECK: kernel.launch @cutensorPermute_f32_r4_tensor
// CHECK-NOT: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cutensorPermute_f32_r4_tensor
// CHECK-NEGATIVE: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cutensorPermute_f32_r4_tensor
