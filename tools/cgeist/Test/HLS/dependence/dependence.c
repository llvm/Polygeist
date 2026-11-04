// #pragma HLS dependence: hls.dependence = [{variable?, class?, type,
// direction?, distance?, dependent}] on the enclosing loop (or function).
// `dependent` is required; type defaults to "inter"; direction is upper-case.
// Vivado's positional spelling (`inter RAW false`) is accepted.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @minimal(
// CHECK: {{^}}    } {hls.dependence = [{dependent = false, type = "inter"}]}
void minimal(int *a) {
  for (int i = 1; i < 64; ++i) {
#pragma HLS dependence dependent=false
    a[i] = a[i - 1] + 1;
  }
}

// CHECK-LABEL: func.func @all_options(
// CHECK: {{^}}    } {hls.dependence = [{class = "array", dependent = true, direction = "RAW", distance = 2 : i32, type = "inter", variable = "buf"}]}
void all_options(int *a) {
  int buf[64];
  for (int i = 0; i < 64; ++i)
    buf[i] = a[i];
  for (int i = 2; i < 64; ++i) {
#pragma HLS dependence variable=buf class=array type=inter direction=raw distance=2 dependent=true
    buf[i] = buf[i - 2] + 1;
  }
  for (int i = 0; i < 64; ++i)
    a[i] = buf[i];
}

// Several dependence pragmas on one loop accumulate.
// CHECK-LABEL: func.func @positional(
// CHECK: {{^}}    } {hls.dependence = [{class = "pointer", dependent = false, direction = "WAR", type = "intra", variable = "a"}, {dependent = true, type = "inter", variable = "b"}]}
void positional(int *a, int *b) {
  for (int i = 1; i < 64; ++i) {
#pragma HLS DEPENDENCE variable=a pointer intra WAR false
#pragma HLS dependence variable=b inter true
    a[i] = a[i - 1] + b[i];
    b[i] = a[i];
  }
}

// CHECK-LABEL: func.func @function_level(
// CHECK-SAME: hls.dependence = [{dependent = false, type = "inter", variable = "a"}]
void function_level(int *a) {
#pragma HLS dependence variable=a dependent=false
  for (int i = 1; i < 64; ++i)
    a[i] = a[i - 1] + 1;
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.dependence = [{dependent = false, type = "inter"}]}
void top(int *a) { minimal(a); }

#ifdef ERRORS
void errors(int *a) {
  for (int i = 1; i < 64; ++i) {
// ERR-DAG: dependence.c:[[@LINE+1]]:1: error: '#pragma HLS dependence' requires option 'dependent'
#pragma HLS dependence variable=a type=inter
    a[i] = a[i - 1];
  }
  for (int i = 1; i < 64; ++i) {
// ERR-DAG: dependence.c:[[@LINE+1]]:33: error: use of undeclared variable 'nope' in '#pragma HLS dependence'
#pragma HLS dependence variable=nope dependent=false
    a[i] = a[i - 1];
  }
  for (int i = 1; i < 64; ++i) {
// ERR-DAG: dependence.c:[[@LINE+1]]:51: warning: 'distance' has no effect with dependent=false in '#pragma HLS dependence'
#pragma HLS dependence variable=a dependent=false distance=4
    a[i] = a[i - 1];
  }
  for (int i = 1; i < 64; ++i) {
// ERR-DAG: dependence.c:[[@LINE+1]]:45: error: invalid value 'rar' for option 'direction' in '#pragma HLS dependence': expected one of: raw, war, waw
#pragma HLS dependence variable=a direction=rar dependent=true
    a[i] = a[i - 1];
  }
}
#endif
