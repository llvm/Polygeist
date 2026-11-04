// Regression (HLSFactory FLAT-acceleration): `latency=-1` (Vitis' "auto",
// its default) in bind_op / bind_storage was rejected as "must be >= 0".
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-DAG: hls.bind_storage = [{impl = "bram", latency = -1 : i32, type = "ram_2p", variable = "buf"}]
// CHECK-DAG: arith.muli {{.*}} {hls.bind_op = [{impl = "dsp", latency = -1 : i32, op = "mul", variable = "val"}]}
int kernel(int *a, int b) {
  int buf[16];
#pragma HLS bind_storage variable=buf type=ram_2p impl=bram latency=-1
  for (int i = 0; i < 16; ++i)
    buf[i] = a[i];
  int val = buf[3] * b;
#pragma HLS BIND_OP variable=val op=mul impl=dsp latency=-1
  return val + buf[15];
}
