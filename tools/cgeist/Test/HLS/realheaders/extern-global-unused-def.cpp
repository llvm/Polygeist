// A global defined in a translation unit that does not use it, and used
// through an `extern` declaration in another, gets the definition and its
// initializer in either order; a definition nothing uses stays out of the
// module, as with one unit. Before, the merged module kept only the
// declaration (the values were lost and the lowered code did not link: the
// "globals declared but undefined" semantics class; W9r7 review defect 2).
// One unit with the `extern` declaration before the definition keeps the
// definition too.
// RUN: cgeist %s %S/Inputs/extern-global-init.cpp %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %S/Inputs/extern-global-init.cpp %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s -DONE_UNIT %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=ONE

// CHECK: memref.global @W : memref<16xi32> = dense<[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "W"}]}
// CHECK-NOT: @Unused
// CHECK-NOT: cgeist.unused_definition
// ONE: memref.global @W : memref<16xi32> = dense<[5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
extern int W[16];
#ifdef ONE_UNIT
int W[16] = {5, 6};
#endif
void g2(int out[16]) {
#pragma HLS array_partition variable=W complete
  for (int i = 0; i < 16; i++) {
    W[i] += i;
    out[i] = W[(i * 3) & 15];
  }
}
