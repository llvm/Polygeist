// Vitis 2025.1 compiles with the mingw sysroot of its Windows install, whose
// <iostream> / <cstdlib> declare ::memcpy and whose <hls_vector.h> brings
// ::uint32_t; the prelude declares both as glibc does, so designs that rely
// on them compile (CIRCT OPEN_ISSUES O76: oscar_baglivio lab_2 / lab_3,
// vitis_examples streaming_vadd). A later <cstring> / <cstdint> agrees.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <iostream>
#include <hls_vector.h>

// The memcpy is an element-wise copy of the two uint32_t arrays (O84 (a)).
// CHECK-LABEL: func.func @_Z5c_topPKjPj
// CHECK: affine.for %[[I:.*]] = 0 to 64
// CHECK: %[[V:.*]] = affine.load %arg0[%[[I]]]
// CHECK: affine.store %[[V]], %{{.*}}[%[[I]]]
void c_top(const uint32_t in[64], uint32_t out[64]) {
  uint32_t b[64];
  memcpy(b, in, 64 * sizeof(uint32_t));
  for (int i = 0; i < 64; i++)
    out[i] = b[i] + 1;
}
#include <cstring>
#include <cstdint>
