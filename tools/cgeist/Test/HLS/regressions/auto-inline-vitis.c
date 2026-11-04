// Vitis's automatic inlining ([HLS 214-178], VITIS_NOTES N18 in the hls-est
// tree, MEMOS decision #34): for a caller C and a callee F without an inline
// pragma, Vitis inlines all calls of F in C iff size(F) x calls(F in C) <=
// 100 and keeps them all otherwise; calls are counted after the user's
// unroll pragmas, size(F) is clang's -O0 IR after sroa / early-cse /
// instcombine / simplifycfg / dce without integer casts, plus size(G) - 2
// per call of a G inlined into F first; a pointer return is always inlined,
// a dataflow process always kept. cgeist keeps such calls as func.call with
// hls.inline_auto_off (on the call and on the callee); --hls-auto-inline=false
// inlines them as before. The callees are the Vitis kernels'
// (~/hls-est-work/exp-src/autoinline/ai_*.cpp, td=off, Vitis 2025.1).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=ATTR
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline=false -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='call @g' --implicit-check-not='call @stage' --implicit-check-not=hls.inline_auto_off
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='call @g' --implicit-check-not='call @stage' --implicit-check-not=hls.inline_auto_off


// ai_sc_k2_n50 (inlined, 50 x 2 = 100) and ai_sc_k2_n51 (kept, 51 x 2 = 102).

static int g50(int x, int y) {
  x = x * 37 + y;
  y = y ^ (x >> 3);
  x = x * 23 + y;
  y = y ^ (x >> 2);
  x = x * 9 + y;
  y = y ^ (x >> 1);
  x = x * 41 + y;
  y = y ^ (x >> 2);
  x = x * 53 + y;
  y = y ^ (x >> 1);
  x = x * 61 + y;
  y = y ^ (x >> 4);
  x = x * 29 + y;
  y = y ^ (x >> 1);
  x = x * 55 + y;
  y = y ^ (x >> 1);
  x = x * 55 + y;
  y = y ^ (x >> 2);
  x = x * 85 + y;
  y = y ^ (x >> 2);
  x = x * 59 + y;
  y = y ^ (x >> 2);
  x = x * 39 + y;
  y = y ^ (x >> 1);
  return x + y;
}

static int g51(int x, int y) {
  x = x * 89 + y;
  y = y ^ (x >> 5) ^ 946;
  x = x * 59 + y;
  y = y ^ (x >> 1);
  x = x * 33 + y;
  y = y ^ (x >> 4);
  x = x * 45 + y;
  y = y ^ (x >> 5);
  x = x * 35 + y;
  y = y ^ (x >> 5);
  x = x * 25 + y;
  y = y ^ (x >> 1);
  x = x * 3 + y;
  y = y ^ (x >> 1);
  x = x * 17 + y;
  y = y ^ (x >> 3);
  x = x * 17 + y;
  y = y ^ (x >> 2);
  x = x * 29 + y;
  y = y ^ (x >> 1);
  x = x * 47 + y;
  y = y ^ (x >> 4);
  x = x * 83 + y;
  y = y ^ (x >> 5);
  return x + y;
}

// REPORT-DAG: hls-auto-inline: caller 'two50' callee 'g50' size 50 calls 2: inlined (size x calls <= threshold)
void two50(int in[4], int out[2]) {
  out[0] = g50(in[0], in[1]);
  out[1] = g50(in[2], in[3]);
}

// ATTR-DAG: func.func private @g51({{.*}}hls.inline_auto_off = {size = 51 : i64}
// ATTR-DAG: func.func private @g60({{.*}}hls.inline_auto_off = {size = 60 : i64}
// ATTR-DAG: func.func private @g_bu({{.*}}hls.inline_auto_off = {size = 108 : i64}
// CHECK-LABEL: func.func @two51(
// CHECK-COUNT-2: call @g51({{.*}}) {hls.inline_auto_off = {calls = 2 : i64, size = 51 : i64}}
// REPORT-DAG: hls-auto-inline: caller 'two51' callee 'g51' size 51 calls 2: kept (size x calls > threshold)
void two51(int in[4], int out[2]) {
  out[0] = g51(in[0], in[1]);
  out[1] = g51(in[2], in[3]);
}

// Calls are counted per caller: one call of g51 here is inlined.
// CHECK-LABEL: func.func @once51(
// CHECK-NOT: call
// CHECK: return
// REPORT-DAG: hls-auto-inline: caller 'once51' callee 'g51' size 51 calls 1: inlined (size x calls <= threshold)
void once51(int in[2], int out[1]) { out[0] = g51(in[0], in[1]); }

// ai_sll_n60_K4 / ai_sll_n60_K4_u2: one call in a 4-trip loop is one call
// (inlined); `unroll factor=2` makes two (kept).

static int g60(int x, int y) {
  x = x * 41 + y;
  y = y ^ (x >> 5);
  x = x * 19 + y;
  y = y ^ (x >> 2);
  x = x * 15 + y;
  y = y ^ (x >> 5);
  x = x * 93 + y;
  y = y ^ (x >> 4);
  x = x * 21 + y;
  y = y ^ (x >> 2);
  x = x * 7 + y;
  y = y ^ (x >> 1);
  x = x * 17 + y;
  y = y ^ (x >> 1);
  x = x * 63 + y;
  y = y ^ (x >> 1);
  x = x * 13 + y;
  y = y ^ (x >> 5);
  x = x * 69 + y;
  y = y ^ (x >> 1);
  x = x * 61 + y;
  y = y ^ (x >> 2);
  x = x * 23 + y;
  y = y ^ (x >> 5);
  x = x * 15 + y;
  y = y ^ (x >> 4);
  x = x * 97 + y;
  y = y ^ (x >> 3);
  x = x * 55 + y;
  return x + y;
}

// CHECK-LABEL: func.func @loop60(
// CHECK-NOT: call
// CHECK: return
// REPORT-DAG: hls-auto-inline: caller 'loop60' callee 'g60' size 60 calls 1: inlined (size x calls <= threshold)
void loop60(int in[8], int out[4]) {
  for (int j = 0; j < 4; j++)
    out[j] = g60(in[2 * j], in[2 * j + 1]);
}

// CHECK-LABEL: func.func @loop60_u2(
// CHECK: affine.for
// CHECK: call @g60({{.*}}) {hls.inline_auto_off = {calls = 2 : i64, size = 60 : i64}}
// REPORT-DAG: hls-auto-inline: caller 'loop60_u2' callee 'g60' size 60 calls 2: kept (size x calls > threshold)
void loop60_u2(int in[8], int out[4]) {
  for (int j = 0; j < 4; j++) {
#pragma HLS unroll factor=2
    out[j] = g60(in[2 * j], in[2 * j + 1]);
  }
}

// ai_m_ptrret_big: a large callee that returns a pointer is inlined.

static int *pick(int *a, int x, int y) {
  x = x * 81 + y;
  y = y ^ (x >> 3);
  x = x * 97 + y;
  y = y ^ (x >> 3);
  x = x * 91 + y;
  y = y ^ (x >> 5);
  x = x * 5 + y;
  y = y ^ (x >> 4);
  x = x * 33 + y;
  y = y ^ (x >> 1);
  x = x * 23 + y;
  y = y ^ (x >> 1);
  x = x * 49 + y;
  y = y ^ (x >> 4);
  x = x * 33 + y;
  y = y ^ (x >> 4);
  x = x * 71 + y;
  y = y ^ (x >> 1);
  x = x * 75 + y;
  y = y ^ (x >> 2);
  x = x * 3 + y;
  y = y ^ (x >> 2);
  x = x * 55 + y;
  y = y ^ (x >> 3);
  x = x * 25 + y;
  y = y ^ (x >> 4);
  x = x * 23 + y;
  y = y ^ (x >> 1);
  x = x * 19 + y;
  y = y ^ (x >> 5);
  x = x * 81 + y;
  y = y ^ (x >> 4);
  x = x * 19 + y;
  y = y ^ (x >> 2);
  x = x * 3 + y;
  y = y ^ (x >> 1);
  x = x * 29 + y;
  y = y ^ (x >> 2);
  x = x * 23 + y;
  y = y ^ (x >> 2);
  x = x * 39 + y;
  y = y ^ (x >> 3);
  x = x * 27 + y;
  y = y ^ (x >> 5);
  x = x * 89 + y;
  y = y ^ (x >> 2);
  x = x * 25 + y;
  y = y ^ (x >> 2);
  x = x * 51 + y;
  y = y ^ (x >> 3);
  x = x * 5 + y;
  y = y ^ (x >> 3);
  x = x * 55 + y;
  y = y ^ (x >> 2);
  x = x * 21 + y;
  y = y ^ (x >> 3);
  x = x * 11 + y;
  y = y ^ (x >> 3);
  x = x * 41 + y;
  y = y ^ (x >> 5);
  return &a[(x + y) & 63];
}

// (cgeist keeps this call already, with or without the option: its
// inliner does not inline a callee that returns a memref.)
// CHECK-LABEL: func.func @ptr2(
// CHECK-NOT: hls.inline_auto_off
// CHECK: return
// REPORT-DAG: hls-auto-inline: caller 'ptr2' callee 'pick' size {{[0-9]+}} calls 2: inlined (returns a pointer)
void ptr2(int *a, int in[4], int out[2]) {
  out[0] = *pick(a, in[0], in[1]);
  out[1] = *pick(a, in[2], in[3]);
}

// ai_bu_own60_hs50: h50 (50) is inlined into g_bu first, which then has
// 60 + 50 - 2 = 108 instructions and is kept even for one call.

static int h50(int x, int y) {
  x = x * 67 + y;
  y = y ^ (x >> 4);
  x = x * 31 + y;
  y = y ^ (x >> 2);
  x = x * 45 + y;
  y = y ^ (x >> 4);
  x = x * 67 + y;
  y = y ^ (x >> 2);
  x = x * 39 + y;
  y = y ^ (x >> 5);
  x = x * 3 + y;
  y = y ^ (x >> 5);
  x = x * 69 + y;
  y = y ^ (x >> 1);
  x = x * 69 + y;
  y = y ^ (x >> 1);
  x = x * 61 + y;
  y = y ^ (x >> 5);
  x = x * 47 + y;
  y = y ^ (x >> 2);
  x = x * 21 + y;
  y = y ^ (x >> 4);
  x = x * 17 + y;
  y = y ^ (x >> 3);
  return x + y;
}

static int g_bu(int x, int y) {
  y = h50(x, y);
  x = x * 79 + y;
  y = y ^ (x >> 5) ^ 479;
  x = x * 95 + y;
  y = y ^ (x >> 4);
  x = x * 21 + y;
  y = y ^ (x >> 3);
  x = x * 31 + y;
  y = y ^ (x >> 2);
  x = x * 65 + y;
  y = y ^ (x >> 1);
  x = x * 37 + y;
  y = y ^ (x >> 4);
  x = x * 81 + y;
  y = y ^ (x >> 4);
  x = x * 33 + y;
  y = y ^ (x >> 2);
  x = x * 13 + y;
  y = y ^ (x >> 3);
  x = x * 39 + y;
  y = y ^ (x >> 3);
  x = x * 59 + y;
  y = y ^ (x >> 3);
  x = x * 15 + y;
  y = y ^ (x >> 2);
  x = x * 39 + y;
  y = y ^ (x >> 3);
  x = x * 83 + y;
  y = y ^ (x >> 5);
  return x + y;
}

// CHECK-LABEL: func.func @bu(
// CHECK: call @g_bu({{.*}}) {hls.inline_auto_off = {calls = 1 : i64, size = 108 : i64}}
// REPORT-DAG: hls-auto-inline: caller 'g_bu' callee 'h50' size 50 calls 1: inlined (size x calls <= threshold)
// REPORT-DAG: hls-auto-inline: caller 'bu' callee 'g_bu' size 108 calls 1: kept (size x calls > threshold)
void bu(int in[2], int out[1]) { out[0] = g_bu(in[0], in[1]); }

// ai_m_df_region: the processes of a dataflow region are kept whatever
// their size.
static void stage1(int in[16], int tmp[16]) {
  for (int i = 0; i < 16; i++)
    tmp[i] = in[i] + 1;
}
static void stage2(int tmp[16], int out[16]) {
  for (int i = 0; i < 16; i++)
    out[i] = tmp[i] * 3;
}
// CHECK-LABEL: func.func @df(
// CHECK: call @stage1({{.*}}) {hls.inline_auto_off =
// CHECK: call @stage2({{.*}}) {hls.inline_auto_off =
// REPORT-DAG: hls-auto-inline: caller 'df' callee 'stage1' size {{[0-9]+}} calls 1: kept (dataflow process)
void df(int in[16], int out[16]) {
#pragma HLS dataflow
  int tmp[16];
  stage1(in, tmp);
  stage2(tmp, out);
}

// w10df_loopcall: a call inside a loop of a dataflow region is not a process
// of its own; it follows the size rule (Vitis: [HLS 214-178] inlines g3 into
// the region, the loop becomes the process Loop_..._proc).
static int g3(int x) { return x * 3 + 1; }
// CHECK-LABEL: func.func @dfloop(
// CHECK: call @stage1({{.*}}) {hls.inline_auto_off =
// CHECK-NOT: call @g3
// CHECK: call @stage2({{.*}}) {hls.inline_auto_off =
// REPORT-DAG: hls-auto-inline: caller 'dfloop' callee 'g3' size {{[0-9]+}} calls 1: inlined (size x calls <= threshold)
void dfloop(int in[16], int out[16]) {
#pragma HLS dataflow
  int a[16], b[16];
  stage1(in, a);
  for (int i = 0; i < 16; i++)
    b[i] = g3(a[i]);
  stage2(b, out);
}

// w9r6rev n48a: a call in a completely unrolled loop of a dataflow region
// (constant bounds) is a process once Vitis has unrolled the loop
// ([HLS 214-186] before the inline decision): 4 x stage1 (size x 4 <= 100)
// stay 4 processes (Vitis: stage, stage_1..3, 4 BRAM). A loop unrolled only
// partly stays a loop, and its calls follow the size rule (g3).
// CHECK-LABEL: func.func @dfunroll(
// CHECK: call @stage1({{.*}}) {hls.inline_auto_off = {calls = 4 : i64
// CHECK-NOT: call @g3
// CHECK: return
// REPORT-DAG: hls-auto-inline: caller 'dfunroll' callee 'stage1' size {{[0-9]+}} calls 4: kept (dataflow process)
// REPORT-DAG: hls-auto-inline: caller 'dfunroll' callee 'g3' size {{[0-9]+}} calls {{[0-9]+}}: inlined (size x calls <= threshold)
void dfunroll(int in[16], int out[16]) {
#pragma HLS dataflow
  int buf[5][16];
  for (int i = 0; i < 16; i++)
    buf[0][i] = in[i];
  for (int k = 0; k < 4; k++) {
#pragma HLS unroll
    stage1(buf[k], buf[k + 1]);
  }
  for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=2
    out[i] = g3(buf[4][i]);
  }
}

// An always_inline callee is inlined by clang before Vitis's inliner runs,
// whatever its size: the Vitis headers mark every ap_int / ap_fixed member
// and operator so under __SYNTHESIS__ (AP_INLINE, etc/ap_common.h), and none
// of them appears in a [HLS 214-178] line of the corpus logs.
__attribute__((always_inline)) static inline int g51ai(int x, int y) {
  return g51(x, y) ^ g51(y, x);
}
// Clang inlines it before the callers are measured (as clang -O0 does in
// N18's proxy), so ai2 holds 4 calls of g51 (204 > 100): kept in ai2.
// CHECK-LABEL: func.func @ai2(
// CHECK-NOT: call @g51ai
// CHECK-COUNT-4: call @g51({{.*}}) {hls.inline_auto_off =
// CHECK-NOT: call @g51ai
// CHECK: return
// REPORT-DAG: hls-auto-inline: caller 'ai2' callee 'g51' size 51 calls 4: kept (size x calls > threshold)
void ai2(int in[4], int out[2]) {
  out[0] = g51ai(in[0], in[1]);
  out[1] = g51ai(in[2], in[3]);
}

// OFF-LABEL: func.func @two51(
// OFF-LABEL: func.func @loop60_u2(
// OFF-LABEL: func.func @bu(
// OFF-LABEL: func.func @df(
