// --hls-keep-invariant-loops (default off): LICM does not hoist a loop out of
// an enclosing loop, so an inner loop whose bounds and body do not depend on
// the enclosing loop stays where the source put it, as Vitis HLS sees it when
// it decides pipelining / flattening (OPEN_ISSUES O5 in the hls-est tree).
// Without the flag cgeist hoists it (unchanged); other invariant ops are
// hoisted with the flag too. (The diffs skip the module attribute line, whose
// dlti entries come out in varying order.)
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | grep -v '^module attributes' > %t.default.mlir
// RUN: FileCheck %s --check-prefix=HOIST < %t.default.mlir
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank --hls-keep-invariant-loops=0 -o - | grep -v '^module attributes' > %t.off.mlir
// RUN: diff %t.default.mlir %t.off.mlir
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank --hls-keep-invariant-loops -o - | FileCheck %s --check-prefix=KEEP
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank --hls-keep-invariant-loops -o - | FileCheck %s --check-prefix=KEEP
// Nothing to keep: the flag changes nothing.
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -DVARIANT -o - | grep -v '^module attributes' > %t.v.default.mlir
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -DVARIANT --hls-keep-invariant-loops -o - | grep -v '^module attributes' > %t.v.keep.mlir
// RUN: diff %t.v.default.mlir %t.v.keep.mlir
// RUN: not cgeist %s --function='*' -S --hls-keep-invariant-loops --parallel-licm=0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: error: --hls-keep-invariant-loops needs --parallel-licm (the default)

#ifndef VARIANT
// The O5 kernel: the reduction over r does not depend on j.
void o5(float A[64][8], float out[64][64], float alpha) {
  float t[64][64];
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      float a3 = alpha * 3.0f;
      float acc = 0;
      for (int r = 0; r < 8; r++)
        acc += a3 * A[i][r];
      t[i][j] = acc;
    }
  }
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++)
      out[i][j] = t[i][j];
}

// No memory access in the invariant loop (MLIR's own LICM part hoists it).
void pure(int out[16][16], int c) {
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++) {
      int s = 0;
      for (int r = 0; r < 8; r++)
        s = s * c + r;
      out[i][j] = s + j;
    }
}
#else
// The reduction depends on j: nothing is hoisted with or without the flag.
void variant(float A[64][8], float out[64][64], float alpha) {
  float bA[64][8];
  for (int i = 0; i < 64; i++)
    for (int r = 0; r < 8; r++)
      bA[i][r] = A[i][r];
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) {
      float acc = 0;
      for (int r = 0; r < 8; r++)
        acc += alpha * bA[i][r] * j;
      out[i][j] = acc;
    }
}
#endif

// HOIST-LABEL: func.func @o5(
// HOIST: %[[A3:.*]] = arith.mulf %arg2, %{{.*}} : f32
// HOIST: affine.for %[[I:.*]] = 0 to 64 {
// HOIST-NEXT: %[[ACC:.*]] = affine.for %{{.*}} = 0 to 8 iter_args
// HOIST: arith.mulf %[[A3]]
// HOIST: affine.for %[[J:.*]] = 0 to 64 {
// HOIST-NEXT: affine.store %[[ACC]], %{{.*}}[%[[I]], %[[J]]]

// HOIST-LABEL: func.func @pure(
// HOIST: %[[S:.*]] = affine.for %{{.*}} = 0 to 8 iter_args
// HOIST: affine.for %{{.*}} = 0 to 16 {
// HOIST-NEXT: affine.for %{{.*}} = 0 to 16 {
// HOIST-NOT: affine.for
// HOIST: arith.addi %[[S]]

// KEEP-LABEL: func.func @o5(
// KEEP: %[[A3:.*]] = arith.mulf %arg2, %{{.*}} : f32
// KEEP: affine.for %[[I:.*]] = 0 to 64 {
// KEEP-NEXT: affine.for %[[J:.*]] = 0 to 64 {
// KEEP-NEXT: %[[ACC:.*]] = affine.for %{{.*}} = 0 to 8 iter_args
// KEEP: arith.mulf %[[A3]]
// KEEP: affine.store %[[ACC]], %{{.*}}[%[[I]], %[[J]]]

// KEEP-LABEL: func.func @pure(
// KEEP: affine.for %{{.*}} = 0 to 16 {
// KEEP-NEXT: affine.for %{{.*}} = 0 to 16 {
// KEEP: %[[S:.*]] = affine.for %{{.*}} = 0 to 8 iter_args
// KEEP: arith.addi %[[S]]
