// A namespace-scope `hls::stream<T, D>` is constructed by the translation
// unit's initialiser, which cgeist does not emit, so no call in the MLIR names
// its D. cgeist keeps D on the global (`hls.stream_depth`), which hls-est's
// FIFO model reads when no constructor reaches the handle (CIRCT W9 round 6
// review defect 5, kernel w9r6rev n49b: Vitis 2 BRAM for the 1024 x 32
// FIFO). A static local's constructor runs in its function: no attribute.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <hls_stream.h>

// CHECK-DAG: memref.global "private" @_ZL2gs : memref<1xi32> = uninitialized {hls.stream_depth = 1024 : i64}
// CHECK-DAG: memref.global "private" @_ZL2gu : memref<1xi32> = uninitialized{{$}}
// CHECK-DAG: memref.global "private" @_ZL2ga : memref<3xi32> = uninitialized {hls.stream_depth = 16 : i64}
static hls::stream<int, 1024> gs;
static hls::stream<int> gu;
static hls::stream<int, 16> ga[3];

static void prod(hls::stream<int> &s, int v) { s.write(v); }

// CHECK-LABEL: func.func @_Z5g_topii
// CHECK-NOT: hls.stream_depth
void g_top(int v, int w) {
  static hls::stream<int, 64> sl;
  prod(gs, v);
  prod(gu, w);
  prod(ga[1], v);
  prod(sl, w);
}
