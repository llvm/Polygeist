// VITIS_NOTES N142 (hls-est tree): a pipeline pragma unrolls the loops of every
// function its loop calls, transitively, before Vitis counts sizes for the
// automatic inlining ([HLS 214-291] for the callee's loop). Kernels
// w9r19p_iu_a (`sub_a` called in a `pipeline` loop and once after it: size 17
// -> 96 in Vitis, 96 x 2 > 100, kept), w9r19p_iu_d (through `mid_d`, kept;
// `sub_d` inlined into it) and w9r19p_iu_c (an automatically pipelined loop
// does not unroll the callee first: `sub_c` inlined).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-rotated-size --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-rotated-size --hls-auto-inline-report --hls-auto-inline-callee-unroll=false -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

// CHECK-DAG: 'sub_a' is called from a pipelined loop or function
// CHECK-DAG: 'sub_d' is called from a pipelined loop or function
// CHECK-DAG: 'mid_d' is called from a pipelined loop or function
// CHECK-DAG: caller 'top_a' callee 'sub_a' size {{[0-9]+}} calls 2: kept (size x calls > threshold)
// CHECK-DAG: caller 'mid_d' callee 'sub_d' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)
// CHECK-DAG: caller 'top_d' callee 'mid_d' size {{[0-9]+}} calls 2: kept (size x calls > threshold)
// CHECK-DAG: caller 'top_c' callee 'sub_c' size {{[0-9]+}} calls 2: inlined (size x calls <= threshold)
// CHECK-NOT: 'sub_c' is called from a pipelined loop
// OFF-NOT: is called from a pipelined loop
// OFF-DAG: caller 'top_a' callee 'sub_a' size {{[0-9]+}} calls 2: inlined (size x calls <= threshold)
// OFF-DAG: caller 'top_d' callee 'mid_d' size {{[0-9]+}} calls 2: inlined (size x calls <= threshold)

typedef unsigned char u8;
u8 tbl[256];

static void sub_a(u8 *b) {
  u8 i = 16;
  while (i--) b[i] = tbl[b[i]];
}
void top_a(u8 buf[16]) {
  for (int r = 0; r < 13; r++) {
#pragma HLS pipeline
    sub_a(buf);
  }
  sub_a(buf);
}

static void sub_d(u8 *b) {
  u8 i = 16;
  while (i--) b[i] = tbl[b[i]];
}
static void mid_d(u8 *b) { sub_d(b); }
void top_d(u8 buf[16]) {
  for (int r = 0; r < 13; r++) {
#pragma HLS pipeline
    mid_d(buf);
  }
  mid_d(buf);
}

static void sub_c(u8 *b) {
  u8 i = 16;
  while (i--) b[i] = tbl[b[i]];
}
void top_c(u8 buf[16]) {
  for (int r = 0; r < 13; r++)
    sub_c(buf);
  sub_c(buf);
}
