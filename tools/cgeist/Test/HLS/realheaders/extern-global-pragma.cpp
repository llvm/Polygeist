// An ARRAY_PARTITION in one translation unit naming an `extern` global that
// an earlier unit declared (and a later one defines) reaches the merged
// global, whatever the order of the units (CIRCT OPEN_ISSUES O75, flowgnn gin
// node_embedding.cc / globals.cc; before, two of three orders lost it). The
// definition gives the global its storage.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %S/Inputs/extern-global-first.cpp %s %S/Inputs/extern-global-def.cpp %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %S/Inputs/extern-global-def.cpp %S/Inputs/extern-global-first.cpp %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %S/Inputs/extern-global-first.cpp %S/Inputs/extern-global-def.cpp %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
typedef ap_fixed<16, 5> T;
extern T W[2][10][100];

// CHECK: memref.global @W : memref<2x10x100xi16> {{.*}}hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "W"}, {dim = 3 : i32, factor = 4 : i32, kind = "cyclic", variable = "W"}]
void xg_top(const T x[100], T y[2][10]) {
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=4 dim=3
  for (int l = 0; l < 2; l++)
    for (int i = 0; i < 100; i += 4) {
#pragma HLS pipeline II=1
      for (int o = 0; o < 10; o++)
        for (int k = 0; k < 4; k++) y[l][o] += W[l][o][i + k] * x[i + k];
    }
}
