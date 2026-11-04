// VITIS_NOTES N153 (hls-est tree): Vitis treats `u < C1 && k + u < C2` as a
// counted loop of min(C1, C2 - k) trips, unrolled and partitioned like a plain
// bound (kernels w9r21p_d1 / d3; class_ece8893_spring_2025 pranay_kamineny
// lab2). WhileConjunctiveBound rewrites the test to `u < min(C1, C2 - k)` when
// C2 is a constant and k a constant or the counter of a loop with constant
// bounds (so C2 - k cannot overflow); RaiseToAffine raises the min bound.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --while-conjunctive-bounds=false -o - | FileCheck %s --check-prefix=OFF

// CHECK-LABEL: func.func @tile(
// CHECK: affine.for %{{.*}} = 0 to 16 {
// CHECK: affine.for %{{.*}} = 0 to 128 step 32 {
// CHECK: affine.for %{{.*}} = 0 to 32 {
// CHECK-NOT: scf.while
// CHECK: return
// OFF-LABEL: func.func @tile(
// OFF: scf.while
void tile(short K[16][128], short out[16][128]) {
  for (int j = 0; j < 16; ++j)
    for (int k = 0; k < 128; k += 32)
      for (int u = 0; u < 32 && (k + u) < 128; ++u)
        out[j][k + u] = K[j][k + u];
}

// min(40, 128 - k) is no affine bound: a counted scf.for.
// CHECK-LABEL: func.func @tile40(
// CHECK: scf.for
// CHECK-NOT: scf.while
// CHECK: return
// OFF-LABEL: func.func @tile40(
// OFF: scf.while
void tile40(short K[16][128], short out[16][128]) {
  for (int j = 0; j < 16; ++j)
    for (int k = 0; k < 128; k += 32)
      for (int u = 0; u < 40 && (k + u) < 128; ++u)
        out[j][k + u] = K[j][k + u];
}

// CHECK-LABEL: func.func @flipped(
// CHECK: affine.for %{{.*}} = 0 to 57 {
// OFF-LABEL: func.func @flipped(
// OFF: scf.while
void flipped(short K[128], short out[128]) {
  for (int u = 0; u < 100 && 7 + u < 64; ++u)
    out[u] = K[u];
}

// A run-time C2: `B - k` may overflow where `k + u < B` never does; the loop
// stays a while.
// CHECK-LABEL: func.func @runtime(
// CHECK: scf.while
void runtime(short K[16][128], short out[16][128], int B) {
  for (int j = 0; j < 16; ++j)
    for (int k = 0; k < 128; k += 32)
      for (int u = 0; u < 32 && (k + u) < B; ++u)
        out[j][k + u] = K[j][k + u];
}

// Not an upper bound on the counter plus an offset: unchanged.
// CHECK-LABEL: func.func @scaled(
// CHECK: scf.while
void scaled(short K[128], short out[128], int B) {
  for (int u = 0; u < 100 && u * 2 < B; ++u)
    out[u] = K[u];
}
