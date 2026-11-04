// Malformed pragmas produce a specific error and a clean exit (status 1),
// never a crash. These are all detected while preprocessing.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

void f(int *a, int n) {
  int buf[16];
  for (int i = 0; i < 16; ++i) {
// CHECK: syntax-errors.c:[[@LINE+1]]:22: error: expected a value after 'II=' in '#pragma HLS pipeline'
#pragma HLS pipeline II=
    a[i] = i;
  }
  for (int i = 0; i < 16; ++i) {
// CHECK: syntax-errors.c:[[@LINE+1]]:28: error: invalid value 'xyz' for option 'style' in '#pragma HLS pipeline': expected one of: stp, flp, frp
#pragma HLS pipeline style=xyz
    a[i] = i;
  }
  for (int i = 0; i < 16; ++i) {
// CHECK: syntax-errors.c:[[@LINE+2]]:27: error: option 'II' is specified more than once in '#pragma HLS pipeline'
// CHECK: syntax-errors.c:[[@LINE+1]]:22: note: previous occurrence is here
#pragma HLS pipeline II=1 II=2
    a[i] = i;
  }
  for (int i = 0; i < 16; ++i) {
// CHECK: syntax-errors.c:[[@LINE+1]]:36: error: invalid value 'maybe' for option 'skip_exit_check' in '#pragma HLS unroll': expected 'true' or 'false'
#pragma HLS unroll skip_exit_check=maybe
    a[i] = i;
  }
  for (int i = 0; i < 16; ++i) {
// CHECK: syntax-errors.c:[[@LINE+1]]:20: error: option 'factor' in '#pragma HLS unroll' requires a value
#pragma HLS unroll factor
    a[i] = i;
  }
// CHECK: syntax-errors.c:[[@LINE+1]]:1: error: '#pragma HLS bind_op' requires option 'op'
#pragma HLS bind_op variable=buf impl=dsp
// CHECK: syntax-errors.c:[[@LINE+1]]:1: error: '#pragma HLS array_partition' requires option 'variable'
#pragma HLS array_partition type=complete
// CHECK: syntax-errors.c:[[@LINE+1]]:38: error: invalid value 'buf+1' for option 'variable' in '#pragma HLS array_partition': expected a variable name
#pragma HLS array_partition variable=buf+1 complete
// CHECK: syntax-errors.c:[[@LINE+1]]:47: error: invalid value 'wide' for option 'type' in '#pragma HLS array_partition': expected one of: cyclic, block, complete
#pragma HLS array_partition variable=buf type=wide
// A token that is not an option name is a warning and the pragma is dropped, as
// in Vitis 2025.1 ([HLS 207-5571], VITIS_NOTES N32).
// CHECK: syntax-errors.c:[[@LINE+1]]:22: warning: unexpected argument '=' in '#pragma HLS pipeline', expected an option name; the pragma is ignored, as Vitis ignores it
#pragma HLS pipeline =2
// An empty `#pragma HLS` is accepted without a message, as Vitis 2025.1
// accepts it (Vitis_Libraries xf_max_suppression.hpp:203; CIRCT OPEN_ISSUES
// O241): no error for the next line.
// CHECK-NOT: syntax-errors.c:[[@LINE+1]]:{{.*}}error
#pragma HLS
// CHECK: syntax-errors.c:[[@LINE+1]]:13: error: expected a pragma name after '#pragma HLS'
#pragma HLS 42
// CHECK: syntax-errors.c:[[@LINE+1]]:43: error: invalid value 'a,,b' for option 'instances' in '#pragma HLS allocation': expected a comma-separated list
#pragma HLS allocation function instances=a,,b limit=1
// CHECK: syntax-errors.c:[[@LINE+1]]:3: error: invalid value '1.5' for option 'II' in '#pragma HLS pipeline': expected an integer constant
  _Pragma("HLS pipeline II=1.5")
  a[0] = buf[0];
}

// CHECK-NOT: error
// CHECK-NOT: Stack dump
