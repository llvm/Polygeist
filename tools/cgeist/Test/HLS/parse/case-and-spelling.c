// Pragma namespace, pragma names, option names and enum values are
// case-insensitive; whitespace around '=' and between options is free.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

// CHECK-LABEL: func.func @upper(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void upper(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS PIPELINE II=2
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @lower(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void lower(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma hls pipeline ii=2
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @mixed(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = true, style = "flp"}]}
void mixed(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma Hls PiPeLiNe Ii=2 ReWind STYLE=FLP
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @spaces(
// CHECK: } {hls.pipeline = [{II = 3 : i32, rewind = true, style = "frp"}]}
void spaces(int *a) {
  for (int i = 0; i < 8; ++i) {
  #  pragma   HLS	pipeline   II =  3	 rewind     style=  frp
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @enum_case(
// CHECK: hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}]
int enum_case(int *a) {
  int buf[8];
#pragma HLS ARRAY_PARTITION VARIABLE=buf TYPE=CYCLIC FACTOR=2
  for (int i = 0; i < 8; ++i)
    buf[i] = a[i];
  int s = 0;
  for (int i = 0; i < 8; ++i)
    s += buf[i];
  return s;
}

// Variable names are C identifiers and stay case-sensitive.
// CHECK-LABEL: func.func @var_case(
// CHECK: hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "Buf"}], hls.preserve, polygeist.varname = "Buf"
// CHECK-NOT: hls.array_partition
// CHECK: return
int var_case(int *a) {
  int Buf[4], buf[4];
#pragma HLS array_partition variable=Buf
  for (int i = 0; i < 4; ++i) {
    Buf[i] = a[i];
    buf[i] = a[i + 4];
  }
  int s = 0;
  for (int i = 0; i < 4; ++i)
    s += Buf[i] * buf[i];
  return s;
}
