// #pragma HLS interface: hls.interface = [{mode, port, ...}] as an argument
// attribute on the named parameter, or on the function for port=return.
// mode is required (keyed or positional); options are typed.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @vadd(
// CHECK-SAME: %arg0: memref<{{.*}}> {hls.interface = [{bundle = "gmem0", depth = 1024 : i32, latency = 64 : i32, max_read_burst_length = 256 : i32, max_widen_bitwidth = 512 : i32, mode = "m_axi", num_read_outstanding = 16 : i32, offset = "slave", port = "in1"}, {bundle = "control", mode = "s_axilite", port = "in1"}]}
// CHECK-SAME: %arg1: memref<{{.*}}> {hls.interface = [{bundle = "gmem1", mode = "m_axi", num_write_outstanding = 8 : i32, offset = "direct", port = "out", register = true}]}
// CHECK-SAME: %arg2: i32 {hls.interface = [{mode = "s_axilite", offset = 16 : i32, port = "n", register = true}]}
// CHECK-SAME: attributes {{{.*}}hls.interface = [{bundle = "control", mode = "s_axilite", port = "return"}]
void vadd(const int in1[1024], int out[1024], int n) {
#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave depth=1024 latency=64 max_read_burst_length=256 num_read_outstanding=16 max_widen_bitwidth=512
#pragma HLS interface mode=m_axi port=out bundle=gmem1 offset=direct num_write_outstanding=8 register
#pragma HLS interface s_axilite port=in1 bundle=control
#pragma HLS interface s_axilite port=n offset=0x10 register
#pragma HLS interface s_axilite port=return bundle=control
  for (int i = 0; i < n; ++i)
    out[i] = in1[i] + 1;
}

// Handshake / memory / streaming modes and control protocols.
// CHECK-LABEL: func.func @modes(
// CHECK-SAME: %arg0: memref<{{.*}}> {hls.interface = [{mode = "ap_memory", port = "mem", storage_type = "ram_2p"}]}
// CHECK-SAME: %arg1: memref<{{.*}}> {hls.interface = [{depth = 32 : i32, mode = "ap_fifo", port = "fifo"}]}
// CHECK-SAME: %arg2: i32 {hls.interface = [{mode = "ap_none", port = "v"}]}
// CHECK-SAME: %arg3: memref<{{.*}}> {hls.interface = [{mode = "ap_vld", port = "o"}]}
// CHECK-SAME: attributes {{{.*}}hls.interface = [{mode = "ap_ctrl_none", port = "return"}]
void modes(int mem[64], int fifo[32], int v, int *o) {
#pragma HLS interface ap_memory port=mem storage_type=ram_2p
#pragma HLS interface ap_fifo port=fifo depth=32
#pragma HLS interface ap_none port=v
#pragma HLS interface ap_vld port=o
#pragma HLS interface ap_ctrl_none port=return
  int s = 0;
  for (int i = 0; i < 32; ++i)
    s += mem[i] * fifo[i];
  *o = s + v;
}

#ifdef ERRORS
int g;
void errors(int *p, int q) {
  int local = 0;
// ERR-DAG: interface.c:[[@LINE+1]]:1: error: '#pragma HLS interface' requires option 'mode'
#pragma HLS interface port=p
// ERR-DAG: interface.c:[[@LINE+1]]:1: error: '#pragma HLS interface' requires option 'port'
#pragma HLS interface m_axi bundle=b
// ERR-DAG: interface.c:[[@LINE+1]]:34: warning: 'local' in '#pragma HLS interface' is not a function parameter; the pragma is ignored, as Vitis ignores it
#pragma HLS interface m_axi port=local
// ERR-DAG: interface.c:[[@LINE+1]]:34: error: 'g' in '#pragma HLS interface' is not a function parameter
#pragma HLS interface m_axi port=g
// ERR-DAG: interface.c:[[@LINE+1]]:34: error: mode=ap_ctrl_hs in '#pragma HLS interface' requires port=return
#pragma HLS interface ap_ctrl_hs port=q
// ERR-DAG: interface.c:[[@LINE+1]]:29: error: port=return in '#pragma HLS interface' requires an ap_ctrl_* or s_axilite mode
#pragma HLS interface m_axi port=return
// ERR-DAG: interface.c:[[@LINE+1]]:28: error: invalid value 'axi_lite' for option 'mode' in '#pragma HLS interface'
#pragma HLS interface mode=axi_lite port=q
// ERR-DAG: interface.c:[[@LINE+1]]:43: error: invalid value 'sometimes' for option 'offset' in '#pragma HLS interface': expected an integer or one of: off, direct, slave
#pragma HLS interface m_axi port=p offset=sometimes
#pragma HLS interface m_axi port=p bundle=a
// A second pragma of one mode on one port replaces the first, as in Vitis.
// ERR-DAG: interface.c:[[@LINE+1]]:1: warning: '#pragma HLS interface' replaces the earlier pragma on the same variable 'p'; the later one decides, as in Vitis
#pragma HLS interface m_axi port=p bundle=b
  p[0] = local + q + g;
}
#endif
