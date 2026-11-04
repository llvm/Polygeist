// unroll binds to the innermost enclosing loop for every loop form.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace

// CHECK-NOT: warning
// CHECK-NOT: error

// CHECK-LABEL: func.func @nested(
// CHECK: {{^}}      } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}
// CHECK-NEXT: {{^}}    } {hls.unroll = [{factor = 2 : i32, skip_exit_check = false}]}
void nested(int a[8][4]) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll factor=2
    for (int j = 0; j < 4; ++j) {
      a[i][j] = i + j;
#pragma HLS unroll
    }
  }
}

// CHECK-LABEL: func.func @imperfect(
// CHECK: {{^}}      }{{$}}
// CHECK: {{^}}    } {hls.unroll = [{factor = 4 : i32, skip_exit_check = false}]}
void imperfect(int a[8][4], int *b) {
  for (int i = 0; i < 8; ++i) {
    b[i] = 0;
    for (int j = 0; j < 4; ++j)
      b[i] += a[i][j];
#pragma HLS unroll factor=4
  }
}

// CHECK-LABEL: func.func @while_loop(
// CHECK: } {hls.unroll = [{factor = 2 : i32, skip_exit_check = true}]}
void while_loop(int *a, int n) {
  int i = 0;
  while (i < n) {
#pragma HLS unroll factor=2 skip_exit_check
    a[i] = i;
    i++;
  }
}

// CHECK-LABEL: func.func @do_while(
// CHECK: scf.while
// CHECK: } attributes {hls.unroll = [{factor = 2 : i32, skip_exit_check = false}]}
void do_while(int *a, int n) {
  int i = 0;
  do {
#pragma HLS unroll factor=2
    a[i] = i;
    i++;
  } while (i < n);
}

// CHECK-LABEL: func.func @dynamic(
// CHECK: } {hls.unroll = [{factor = 8 : i32, skip_exit_check = false}]}
void dynamic(int *a, int lo, int hi) {
  for (int i = lo; i < hi; i += 1) {
#pragma HLS unroll factor=8
    a[i] = i;
  }
}
