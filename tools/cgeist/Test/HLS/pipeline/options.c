// #pragma HLS pipeline: each option alone, all together, and defaults.
// Loop form: hls.pipeline = [{II, rewind, style[, enable_flush]}] on the
// loop; `off` gives the unit attribute hls.pipeline_off instead.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

// Defaults: II = 0 ("not specified": the consumer applies its own target
// II, as before the rewrite), rewind=false, style=stp.
// CHECK-LABEL: func.func @defaults(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "stp"}]}
void defaults(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @ii(
// CHECK: } {hls.pipeline = [{II = 4 : i32, rewind = false, style = "stp"}]}
void ii(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=4
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @rewind(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = true, style = "stp"}]}
void rewind(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline rewind
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @rewind_false(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "stp"}]}
void rewind_false(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline rewind=false
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @style_stp(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "stp"}]}
void style_stp(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline style=stp
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @style_flp(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "flp"}]}
void style_flp(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline style=flp
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @style_frp(
// CHECK: } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "frp"}]}
void style_frp(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline style=frp
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @enable_flush(
// CHECK: } {hls.pipeline = [{II = 0 : i32, enable_flush = true, rewind = false, style = "stp"}]}
void enable_flush(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline enable_flush
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @all(
// CHECK: } {hls.pipeline = [{II = 3 : i32, enable_flush = true, rewind = true, style = "flp"}]}
void all(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=3 rewind style=flp enable_flush
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @off(
// CHECK-NOT: hls.pipeline =
// CHECK: } {hls.pipeline_off}
void off(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline off
    a[i] = i;
  }
}

// `off=true` is the same as `off`; `off=false` is a normal pipeline.
// CHECK-LABEL: func.func @off_true(
// CHECK: } {hls.pipeline_off}
void off_true(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline off=true
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @off_false(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void off_false(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline off=false II=2
    a[i] = i;
  }
}

// In a function body outside any loop, the function is pipelined.
// CHECK-LABEL: func.func @function_level(
// CHECK-SAME: hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]
void function_level(int *a) {
#pragma HLS pipeline II=1
  a[0] = 1;
  a[1] = 2;
}

// CHECK-LABEL: func.func @function_off(
// CHECK-SAME: hls.pipeline_off
void function_off(int *a) {
#pragma HLS pipeline off
  a[0] = 1;
}
