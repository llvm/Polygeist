// hls.array_partition drives hls-est's memory banking and BRAM accounting.
// REQUIRES: hls-est
// RUN: cgeist %s --function=kernel -S --hls-annotate --raise-scf-to-affine -o %t.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.mlir
// RUN: %hls-est %t.mlir --allow-unregistered-dialect --memory-banking-bram | FileCheck %s --check-prefix=BANK
// RUN: cgeist %s --function=kernel -S --hls-annotate --raise-scf-to-affine -DNO_PRAGMA -o %t.none.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.none.mlir
// RUN: %hls-est %t.none.mlir --allow-unregistered-dialect --memory-banking-bram | FileCheck %s --check-prefix=NOBANK
// RUN: %hls-est %t.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s --check-prefix=BRAM

// cyclic factor=4 on a 1024-element buffer: four banks of 256.
// BANK-LABEL: func.func @kernel(
// BANK-COUNT-4: memref.alloca() {{.*}}hls.bank_of = "buf"{{.*}} : memref<256xf32>
// BANK-NOT: memref<1024xf32>

// Without the pragma the buffer stays whole.
// NOBANK-LABEL: func.func @kernel(
// NOBANK: memref.alloca() {{.*}}polygeist.varname = "buf"{{.*}} : memref<1024xf32>
// NOBANK-NOT: memref<256xf32>

// BRAM-COUNT-4: memref.alloca @ loc(
// BRAM: Total BRAM:

void kernel(float A[64][1024], float *x, float *y) {
  float buf[1024];
#ifndef NO_PRAGMA
#pragma HLS array_partition variable=buf cyclic factor=4
#endif
  for (int i = 0; i < 1024; ++i)
    buf[i] = x[i];
  for (int i = 0; i < 64; ++i) {
    float s = 0;
    for (int j = 0; j < 1024; ++j)
      s += A[i][j] * buf[j];
    y[i] = s;
  }
}
