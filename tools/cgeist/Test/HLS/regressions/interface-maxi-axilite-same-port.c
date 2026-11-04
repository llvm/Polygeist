// Regression: an m_axi interface and an s_axilite interface on the same port
// (the usual Vitis pattern for passing the m_axi offset over AXI-Lite) were
// reported as conflicting pragmas.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK: %arg0: memref<?xi32> {hls.interface = [{bundle = "gmem", mode = "m_axi", offset = "slave", port = "a"}, {bundle = "control", mode = "s_axilite", port = "a"}]}
void k(int *a) {
#pragma HLS interface m_axi port=a bundle=gmem offset=slave
#pragma HLS interface s_axilite port=a bundle=control
  a[0] = 1;
}
