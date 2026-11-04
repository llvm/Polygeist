// Loops and pragmas produced by macro expansion: placement is decided on the
// expansion, so a pragma inside a macro-generated loop body binds to it.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace

// CHECK-NOT: warning
// CHECK-NOT: error

#define FOR_PIPELINED(i, n, body)                                              \
  for (int i = 0; i < (n); ++i) {                                              \
    _Pragma("HLS pipeline II=1") body                                          \
  }

// CHECK-LABEL: func.func @generated(
// CHECK: {{^}}      } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK-NEXT: {{^}}    } {hls.unroll = [{factor = 2 : i32, skip_exit_check = false}]}
void generated(int a[4][8]) {
  for (int i = 0; i < 4; ++i) {
#pragma HLS unroll factor=2
    FOR_PIPELINED(j, 8, a[i][j] += j;)
  }
}

// Two macro-generated sibling loops on one source line.
// CHECK-LABEL: func.func @same_line(
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void same_line(int *a, int *b) {
  FOR_PIPELINED(i, 8, a[i] = i;) FOR_PIPELINED(i, 8, b[i] = a[i];)
}
