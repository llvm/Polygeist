// A global whose initializer is not a constant (C++ dynamic initialization,
// run before main) was left uninitialized, so the design read 0: davendra
// lab2's `static const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk)` made
// every attention score 0 (OPEN_ISSUES O44 in the hls-est tree), Ray-tracing's
// kInfinity / kEpsilon and auto_ntt's WORD_SIZE_PLUS2_MASK were 0 as well.
// cgeist now emits the initialization as a function marked hls.global_init;
// one that folds to a single constant store becomes the global's initial
// value (as LLVM's GlobalOpt does before Vitis schedules), any other stays a
// function that check_semantics runs once before the top.
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-dynamic-global-init=false -o - 2>/dev/null | FileCheck %s --check-prefix=OFF

// CHECK-DAG: memref.global "private" @_ZL5scale : memref<1xf32> = dense<0.0883883461>
// CHECK-DAG: memref.global "private" @_ZL1S : memref<1xi32> = uninitialized
// CHECK-LABEL: func.func @__hls_global_init._ZL1S()
// CHECK-SAME: hls.global_init =
// CHECK: memref.get_global @tbl
// CHECK: memref.get_global @_ZL1S
// CHECK-NOT: func.func @__hls_global_init._ZL5scale
// CHECK-NOT: func.func @__hls_global_init.pes
// OFF-DAG: memref.global "private" @_ZL5scale : memref<1xf32> = uninitialized
// OFF-NOT: hls.global_init

#include <math.h>
static const float scale = 1.0f / sqrtf(128.0f);
int tbl[4] = {1, 2, 3, 4};
static int sumtbl() {
  int s = 0;
  for (int i = 0; i < 4; i++)
    s += tbl[i] * tbl[i];
  return s;
}
static int S = sumtbl();
// An array of class objects (a constructor per element, Eyeriss' `PE
// pes[..][..]`) is left to the old path: it crashed the first version.
struct PE {
  int c;
  PE() : c(0) {}
};
PE pes[2][2];
void top(float out[1], int iout[2], int in[2]) {
  pes[in[0] & 1][0].c += 1;
  tbl[0] += in[0];
  out[0] = scale * in[1];
  iout[0] = S + in[1];
  iout[1] = tbl[0];
}
