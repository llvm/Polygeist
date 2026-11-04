// W10 review fixes to the automatic inlining decision (VITIS_NOTES N18,
// MEMOS decision #34); every kernel was run through Vitis 2025.1 (td=off,
// ~/hls-est-work/experiments/{rv_apf,rv_ai_pipe_inner,w10f_*}).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank %vitis_include --scalar-wrapper-abi --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank %vitis_include --scalar-wrapper-abi -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank %vitis_include --scalar-wrapper-abi -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank %vitis_include --scalar-wrapper-abi --hls-auto-inline-report --hls-auto-inline-unroll-budget=50 -o /dev/null 2>&1 | FileCheck %s --check-prefix=BUDGET

#include <ap_fixed.h>

// (1) The ap_fixed operators are always_inline (AP_INLINE): they are inlined
// before the callee is measured, as clang -O0 does in N18's proxy. Vitis:
// matrix_add 25 instructions (cmplx 002), both calls inlined (214-178); the
// proxy (clang 18 + opt by hand) 32. Before the fix cgeist measured 68 and
// kept both calls.
typedef ap_fixed<16, 5> T;
void matrix_add(T A[8][32], T B[8][32], T C[8][32]) {
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 32; j++)
      C[i][j] = A[i][j] + B[i][j];
}
// REPORT-DAG: hls-auto-inline: caller 'rv_apf' callee 'matrix_add' size {{[1-4][0-9]}} calls 2: inlined (size x calls <= threshold)
void rv_apf(T A[8][32], T B[8][32], T C[8][32], T D[8][32]) {
  matrix_add(A, B, C);
  matrix_add(B, C, D);
}

// (2) A pipeline pragma unrolls every loop nested in the pipelined loop
// completely before the inline decision ([HLS 214-291] "complete unroll
// implied by the pipeline pragma", then 214-186); g (about 30 instructions)
// is then called 4 times and kept (Vitis: module g).
int g(int x, int y) {
  x = x * 19 + y;
  y = y ^ (x >> 5);
  x = x * 11 + y;
  y = y ^ (x >> 3);
  x = x * 17 + y;
  y = y ^ (x >> 4);
  x = x * 59 + y;
  y = y ^ (x >> 4);
  x = x * 85 + y;
  y = y ^ (x >> 4);
  x = x * 29 + y;
  y = y ^ (x >> 1);
  x = x * 65 + y;
  y = y ^ (x >> 1);
  return x + y;
}

// REPORT-DAG: hls-auto-inline: caller 'rv_ai_pipe_inner' callee 'g' size {{[2-3][0-9]}} calls 4: kept (size x calls > threshold)
// CHECK-LABEL: func.func @_Z16rv_ai_pipe_innerPA4_iS0_
// CHECK: call @_Z1gii({{.*}}hls.inline_auto_off
// Over the unroll budget (a designer knob against nests LLVM takes forever
// to unroll), size and calls are estimated from the trip counts instead.
// BUDGET-DAG: hls-auto-inline: '_Z16rv_ai_pipe_innerPA4_iS0_' is not unrolled for measuring: over 50 instructions after the unroll; size and calls are estimated
// BUDGET-DAG: hls-auto-inline: caller 'rv_ai_pipe_inner' callee 'g' size {{[2-3][0-9]}} calls 4: kept (size x calls > threshold)
void rv_ai_pipe_inner(int in[64][4], int out[64][4]) {
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II=1
    for (int j = 0; j < 4; j++)
      out[i][j] = g(in[i][j], j);
  }
}

// Both loops of a nest under the pipelined loop: 2 x 2 calls.
// REPORT-DAG: hls-auto-inline: caller 'w10f_pipe_deep' callee 'g' size {{[2-3][0-9]}} calls 4: kept (size x calls > threshold)
void w10f_pipe_deep(int in[64][4], int out[64][4]) {
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II=1
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        out[i][j * 2 + k] = g(in[i][j * 2 + k], k);
  }
}

// A pipelined function: every loop in it is unrolled.
// REPORT-DAG: hls-auto-inline: caller 'w10f_fnpipe' callee 'g' size {{[2-3][0-9]}} calls 4: kept (size x calls > threshold)
void w10f_fnpipe(int in[4], int out[4]) {
#pragma HLS pipeline II=1
  for (int j = 0; j < 4; j++)
    out[j] = g(in[j], j);
}

// Without the pipeline pragma: one call, inlined.
// REPORT-DAG: hls-auto-inline: caller 'nopipe' callee 'g' size {{[2-3][0-9]}} calls 1: inlined (size x calls <= threshold)
void nopipe(int in[4], int out[4]) {
  for (int j = 0; j < 4; j++)
    out[j] = g(in[j], j);
}

// (3) Vitis takes a dataflow pragma only directly in a 'for' loop body or a
// function body; anywhere else it warns ([HLS 207-5576]) and inlines the
// callees by the size rule (214-178 for p1 and p2 in all three kernels).
static void p1(int a[16], int b[16]) { for (int i = 0; i < 16; i++) b[i] = a[i] + 1; }
static void p2(int b[16], int c[16]) { for (int i = 0; i < 16; i++) c[i] = b[i] * 3; }

// WARN-DAG: auto-inline-review.cpp:[[@LINE+5]]:{{[0-9]+}}: warning: '#pragma HLS dataflow' is ignored: only a 'for' loop body or a function body supports the dataflow pragma
// REPORT-DAG: hls-auto-inline: caller 'rv_df_nested3' callee 'p1' size {{[0-9]+}} calls 1: inlined
void rv_df_nested3(int a[16], int c[16]) {
  int b[16];
  {
#pragma HLS dataflow
    p1(a, b);
    p2(b, c);
  }
}

// WARN-DAG: auto-inline-review.cpp:[[@LINE+6]]:{{[0-9]+}}: warning: '#pragma HLS dataflow' is ignored
// REPORT-DAG: hls-auto-inline: caller 'w10f_df_while' callee 'p2' size {{[0-9]+}} calls 1: inlined
void w10f_df_while(int a[16], int c[16]) {
  int b[16];
  int n = 0;
  while (n < 4) {
#pragma HLS dataflow
    p1(a, b);
    p2(b, c);
    n++;
  }
}

// WARN-DAG: auto-inline-review.cpp:[[@LINE+6]]:{{[0-9]+}}: warning: '#pragma HLS dataflow' is ignored
// REPORT-DAG: hls-auto-inline: caller 'w10f_df_forblock' callee 'p1' size {{[0-9]+}} calls 1: inlined
void w10f_df_forblock(int a[16], int c[16]) {
  int b[16];
  for (int n = 0; n < 4; n++) {
    {
#pragma HLS dataflow
      p1(a, b);
      p2(b, c);
    }
  }
}

// Directly in a function body it is a dataflow region (a dataflow loop is
// not modelled by the inline decision, hls-doc/frontend-pragmas.md 3.2).
// REPORT-DAG: hls-auto-inline: caller 'df_body' callee 'p2' size {{[0-9]+}} calls 1: kept (dataflow process)
void df_body(int a[16], int c[16]) {
#pragma HLS dataflow
  int b[16];
  p1(a, b);
  p2(b, c);
}

// (4) A call in the body of an always_inline function counts in, and is kept
// in, the caller the function is inlined into: h calls g once, h is called
// twice, so the caller holds 2 calls of g51 (51 x 2 = 102 > 100). Vitis
// (w10f_ai_h2): h inlined first ([HLS 214-131]), cmplx 002 shows the 2 calls
// of g51 in the caller, and no 214-178 line inlines g51 (the later
// [XFORM 203-602] inliner, not modelled, removes it afterwards). Before the
// fix cgeist decided g51 x 1 inside h and inlined it.
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
__attribute__((always_inline)) static inline int h(int x, int y) {
  return g51(x, y) + 1;
}
// REPORT-DAG: hls-auto-inline: caller 'w10f_ai_h2' callee 'g51' size 51 calls 2: kept (size x calls > threshold)
// CHECK-LABEL: func.func @_Z10w10f_ai_h2PiS_
// CHECK-NOT: call @_ZL1hii
// CHECK-COUNT-2: call @_ZL3g51ii({{.*}}) {hls.inline_auto_off =
// CHECK-NOT: call @_ZL1hii
// CHECK: return
void w10f_ai_h2(int in[4], int out[2]) {
  out[0] = h(in[0], in[1]);
  out[1] = h(in[2], in[3]);
}
