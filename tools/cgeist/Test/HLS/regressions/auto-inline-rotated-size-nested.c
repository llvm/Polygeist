// N131's loop term with nested loops (VITIS_NOTES N131, W9 round 17 review; kernels
// w9r17rev_rv_rot_nest51 and w9r17fx_nest, Vitis 2025.1 td=off): the
// one-per-constant-loop-after-the-first term counts the loops at the top of the
// function and those directly in a variable-trip loop; constant-trip loops nested
// in a constant-trip loop add nothing.
// - blk: a 3-deep constant nest, then a constant loop. Vitis counts 54 (5 casts):
//   49 x 2 calls = 98 <= 100, inlined ([HLS 214-178] Inlining function 'blk').
//   Counting every constant loop gave 51 x 2 = 102, kept.
// - n_d: a variable loop holding two constant loops: Vitis 40 (4 casts) = 36.
// - dct8: spector_hls dct's DCT8_auto (a constant nest, then a constant loop, with
//   `buf[k] += a * b * c` in float): Vitis 56 (5 casts) = 51, kept (51 x 2 = 102);
//   clang 18 contracts the multiply-add into one llvm.fmuladd, which counts as
//   Vitis' fmul + fadd.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-rotated-size --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s

// CHECK-DAG: caller 'top' callee 'n_d' size 36 calls 1: kept (inline off)
// CHECK-DAG: caller 'top' callee 'blk' size 49 calls 2: inlined (size x calls <= threshold)
// CHECK-DAG: caller 'dct' callee 'dct8' size 51 calls 2: kept (size x calls > threshold)

static void blk(int a[256], int out[256], int s) {
  int t[256];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 8; j++)
      for (int k = 0; k < 8; k++) t[i * 64 + j * 8 + k] = a[i * 64 + k * 8 + j] * s + j;
  for (int i = 0; i < 256; i++) out[i] = t[255 - i] - (t[i] ^ (i*5)) + t[i>>1];
}
static void n_d(int a[64], int o[64], int n) {
#pragma HLS inline off
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 8; j++) o[(i * 8 + j) & 63] = a[j * 8] + i;
    for (int k = 0; k < 4; k++) o[(i + k) & 63] ^= a[k] * 3;
  }
}
void top(int a[256], int b[256], int c[256], int d[64], int n) {
  blk(a, b, n);
  blk(b, c, n + 1);
  n_d(a, d, n);
}

void dct8(int *D_O, int *D_I, int ostride, int istride) {
  float buf[8];
  float C[16] = {1.0f, 0.98078528f, 0.92387953f, 0.83146961f, 0.70710678f, 0.55557023f,
                 0.38268343f, 0.19509032f, 0, -0.19509032f, -0.38268343f, -0.55557023f,
                 -0.70710678f, -0.83146961f, -0.92387953f, -0.98078528f};
  for (int k = 0; k < 8; k++) {
    buf[k] = 0;
    for (int n = 0; n < 8; n++)
      buf[k] += D_I[n * istride] * C[(2 * n * k + k) % 16] *
                (((2 * n * k + k) / 16) % 2 ? -1 : 1);
  }
  D_O[0 * ostride] = buf[0] * 0.35355339f;
  for (int i = 1; i < 8; i++)
    D_O[i * ostride] = buf[i] * 0.5f;
}
void dct(int dst[64], int src[64]) {
  for (int k = 0; k < 8; k++)
    dct8(dst + k * 8, src + k * 8, 1, 1);
  for (int k = 0; k < 8; k++)
    dct8(dst + k, dst + k, 8, 8);
}
