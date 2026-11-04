// A loop pragma binds to the innermost loop whose body contains it, for all
// loop forms, whether cgeist emits the loop as scf.for, scf.while or
// affine.for.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s --check-prefixes=CHECK,SCF --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,AFF --strict-whitespace

// CHECK-NOT: warning
// CHECK-NOT: error

// Top of the body.
// CHECK-LABEL: func.func @top_of_body(
// SCF:     scf.for
// AFF:     affine.for
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void top_of_body(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
    a[i] = 2 * a[i];
  }
}

// After other statements in the body.
// CHECK-LABEL: func.func @after_statements(
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void after_statements(int *a) {
  for (int i = 0; i < 8; ++i) {
    int t = a[i] * 3;
    a[i] = t + 1;
#pragma HLS pipeline II=1
  }
}

// Braceless body: the pragma sits between the header and the statement.
// CHECK-LABEL: func.func @braceless(
// CHECK: {{^}}    } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void braceless(int *a) {
  for (int i = 0; i < 8; ++i)
#pragma HLS pipeline II=2
    a[i] = i;
}

// Nested loops: binds to the innermost loop only (indentation = depth).
// CHECK-LABEL: func.func @nested_inner(
// CHECK: {{^}}      } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK-NEXT: {{^}}    }{{$}}
void nested_inner(int a[8][16]) {
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 16; ++j) {
#pragma HLS pipeline II=1
      a[i][j] += i;
    }
}

// Imperfect nest: the pragma is in the outer body, after the inner loop.
// CHECK-LABEL: func.func @imperfect_outer(
// CHECK: {{^}}      }{{$}}
// CHECK: {{^}}    } {hls.pipeline = [{II = 4 : i32, rewind = false, style = "stp"}]}
void imperfect_outer(int a[8][16], int *b) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 16; ++j)
      a[i][j] += i;
    b[i] = a[i][0];
#pragma HLS pipeline II=4
  }
}

// Three levels, pragma in the middle loop and another in the innermost.
// CHECK-LABEL: func.func @three_levels(
// CHECK: {{^}}        } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK-NEXT: {{^}}      } {hls.pipeline_off}
// CHECK-NEXT: {{^}}    }{{$}}
void three_levels(int a[4][8][16]) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
#pragma HLS pipeline off
      for (int k = 0; k < 16; ++k) {
#pragma HLS pipeline II=1
        a[i][j][k] += k;
      }
    }
}

// Sibling loops each get their own pragma.
// CHECK-LABEL: func.func @siblings(
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK: {{^}}    } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void siblings(int *a, int *b) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
    a[i] = i;
  }
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=2
    b[i] = a[i] + 1;
  }
}

// Inside an if in the loop body: still the loop.
// CHECK-LABEL: func.func @inside_if(
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void inside_if(int *a) {
  for (int i = 0; i < 8; ++i) {
    if (a[i] > 0) {
#pragma HLS pipeline II=1
      a[i] = 0;
    }
  }
}

// while loop.
// CHECK-LABEL: func.func @while_loop(
// CHECK: {{^}}    } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void while_loop(int *a, int n) {
  int i = 0;
  while (i < n) {
#pragma HLS pipeline II=2
    a[i] = i;
    ++i;
  }
}

// do-while loop (stays an scf.while).
// CHECK-LABEL: func.func @do_while(
// CHECK: scf.while
// CHECK: {{^}}    } attributes {hls.pipeline = [{II = 3 : i32, rewind = false, style = "stp"}]}
void do_while(int *a, int n) {
  int i = 0;
  do {
#pragma HLS pipeline II=3
    a[i] = i;
    ++i;
  } while (i < n);
}

// Non-constant bounds.
// CHECK-LABEL: func.func @dynamic_bounds(
// SCF: scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// AFF: affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void dynamic_bounds(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS pipeline II=1
    a[i] = i;
  }
}

// A loop with a data-dependent exit stays a while loop.
// CHECK-LABEL: func.func @data_dependent_exit(
// CHECK: scf.while
// CHECK: attributes {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void data_dependent_exit(int *a) {
  for (int i = 0; a[i] != 0; ++i) {
#pragma HLS pipeline II=1
    a[i] = a[i] - 1;
  }
}

// A loop with break.
// CHECK-LABEL: func.func @with_break(
// CHECK: {{^}}      } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
// CHECK-NEXT: {{^}}    }{{$}}
void with_break(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS pipeline II=2
    if (a[i] < 0)
      break;
    a[i] = 2 * a[i];
  }
}

// Labelled loop (Vitis style).
// CHECK-LABEL: func.func @labelled(
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void labelled(int *a) {
L1:
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
    a[i] = i;
  }
}
