// Two interface pragmas of one mode on one port: Vitis 2025.1 accepts both
// without a message and the later one decides (kernels w8b_iface_k_two_maxi
// and k_two_maxi_bundle: the port is inferred on the later bundle). cgeist
// used to reject the second as a conflict, so class_ece8893_2023_spring
// YOLOv4-Tiny CSP_Layer1Layer2 got no MLIR (W8b).
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK: warning: '#pragma HLS interface' replaces the earlier pragma on the same variable 'out'; the later one decides, as in Vitis
// CHECK: note: earlier pragma is here
// CHECK-NOT: error
// CHECK: %arg1: memref<?xi32> {hls.interface = [{bundle = "wt", depth = 32 : i32, mode = "m_axi", port = "out"}]}
void k(int *in, int *out) {
#pragma HLS interface m_axi port=out bundle=fm depth=64
#pragma HLS interface m_axi port=out bundle=wt depth=32
  for (int i = 0; i < 32; i++)
    out[i] = in[i] + 1;
}
