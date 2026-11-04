// --hls-keep-misfit-inline-off: an `inline off` call whose array argument does
// not fit the parameter's first dimension stays a call, the argument
// reinterpreted to the parameter type, as Vitis keeps the module whatever the
// argument's size (VITIS_NOTES N123, OPEN_ISSUES O156 / O167 in the hls-est tree;
// forgebench attn_breakdown: matmul_large(.., Q_weight_buffer[32][32], ..) with a
// [8][32] parameter, one instance under `allocation limit=1`, 17 BRAM = Vitis).
// A global is reinterpreted directly (its cast to a dynamic first dimension and
// on to the parameter would fold into an invalid memref.cast); a local array
// too. A call where a partition is involved (mm_pt's parameter) is still
// inlined.
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-keep-misfit-inline-off -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>/dev/null | FileCheck %s --check-prefix=OFF

// CHECK-LABEL: func.func @_Z3topPA32_iS0_(
// CHECK: %[[G:.*]] = memref.get_global @wbuf : memref<32x32xi32>
// CHECK: %[[V:.*]] = memref.reinterpret_cast %[[G]] to offset: [0], sizes: [8, 32], strides: [32, 1] : memref<32x32xi32> to memref<8x32xi32>
// CHECK: call @_Z2mmPA32_iS0_S0_(%{{.*}}, %[[V]], %{{.*}})
// CHECK: call @_Z2mmPA32_iS0_S0_(
// CHECK-NOT: call @_Z5mm_ptPA32_iS0_S0_(
// CHECK: return
// OFF-LABEL: func.func @_Z3topPA32_iS0_(
// OFF-NOT: call @_Z2mmPA32_iS0_S0_(
// OFF: return

int wbuf[32][32];

void mm(int a[8][32], int b[8][32], int c[8][32]) {
#pragma HLS inline off
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 32; j++)
      c[i][j] = a[i][j] * b[i][j];
}

void mm_pt(int a[8][32], int b[8][32], int c[8][32]) {
#pragma HLS inline off
#pragma HLS array_partition variable=b dim=2 complete
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 32; j++)
      c[i][j] = a[i][j] + b[i][j];
}

void top(int in[8][32], int out[8][32]) {
  int loc[32][32], loc2[32][32];
  int t[8][32];
  for (int i = 0; i < 32; i++)
    for (int j = 0; j < 32; j++) {
      wbuf[i][j] = i + j;
      loc[i][j] = i - j;
      loc2[i][j] = i * j;
    }
  mm(in, wbuf, t);
  mm(t, loc, out);
  mm_pt(t, loc2, out);
}
