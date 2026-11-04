// The legacy `#pragma HLS RESOURCE variable=v core=C` (still accepted by
// Vitis 2025.1): a memory core is the bind_storage it names, so the array
// gets hls.bind_storage = [{type, impl?, latency?, variable}]. Vitis 2025.1
// td=off, kernels w9r13p_rs_res / rs_bind (CIRCT VITIS_NOTES N98): `RESOURCE
// core=RAM_2P_BRAM` and `bind_storage type=ram_2p impl=bram` on a 16 x 16-bit
// array both give RAM_2P_BRAM_1R1W, 1 BRAM. An operator core (Mul_LUT,
// AddSub_DSP) names no storage and stays an hls.resource attribute.
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s

// CHECK-NOT: warning

int g[64];
// CHECK-DAG: memref.global @g : memref<64xi32> = uninitialized {hls.bind_storage = [{impl = "lutram", type = "ram_1p", variable = "g"}]}

// CHECK-LABEL: func.func @cores(
// CHECK-DAG: {hls.bind_storage = [{impl = "bram", type = "ram_2p", variable = "a1"}], hls.preserve, polygeist.varname = "a1"}
// CHECK-DAG: {hls.bind_storage = [{type = "ram_1p", variable = "a2"}], hls.preserve, polygeist.varname = "a2"}
// CHECK-DAG: {hls.bind_storage = [{impl = "lutram", latency = 2 : i32, type = "ram_t2p", variable = "a3"}], hls.preserve, polygeist.varname = "a3"}
// CHECK-DAG: {hls.bind_storage = [{impl = "srl", type = "fifo", variable = "a4"}], hls.preserve, polygeist.varname = "a4"}
// CHECK-DAG: {hls.preserve, hls.resource = [{core = "Mul_LUT", variable = "a5"}], polygeist.varname = "a5"}
int cores(int *in) {
  int a1[16], a2[16], a3[16], a4[16], a5[16];
#pragma HLS RESOURCE variable=g core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=a1 core=RAM_2P_BRAM
#pragma HLS resource variable=a2 core=RAM_1P
#pragma HLS RESOURCE variable=a3 core=RAM_T2P_LUTRAM latency=2
#pragma HLS RESOURCE variable=a4 core=FIFO_SRL
#pragma HLS RESOURCE variable=a5 core=Mul_LUT
  for (int i = 0; i < 16; ++i) {
    a1[i] = in[i] + g[i];
    a2[i] = a1[i] + 1;
    a3[i] = a2[i] + 1;
    a4[i] = a3[i] + 1;
    a5[i] = a4[i] + 1;
  }
  int s = 0;
  for (int i = 0; i < 16; ++i)
    s += a1[15 - i] + a2[i] + a3[i] + a4[i] + a5[15 - i];
  return s;
}
