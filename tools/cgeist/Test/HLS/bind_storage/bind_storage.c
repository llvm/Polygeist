// #pragma HLS bind_storage: hls.bind_storage = [{variable, type, impl?,
// latency?}] on the variable's storage. type is required.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

int grom[256];

// CHECK-DAG: memref.global @grom : memref<256xi32> = uninitialized {hls.bind_storage = [{impl = "bram", type = "rom_2p", variable = "grom"}]}

// Each type/impl combination, and latency.
// CHECK-LABEL: func.func @kinds(
// CHECK-DAG: {hls.bind_storage = [{type = "ram_1p", variable = "a1"}], hls.preserve, polygeist.varname = "a1"}
// CHECK-DAG: {hls.bind_storage = [{impl = "lutram", type = "ram_2p", variable = "a2"}], hls.preserve, polygeist.varname = "a2"}
// CHECK-DAG: {hls.bind_storage = [{impl = "uram", latency = 3 : i32, type = "ram_t2p", variable = "a3"}], hls.preserve, polygeist.varname = "a3"}
// CHECK-DAG: {hls.bind_storage = [{impl = "srl", type = "fifo", variable = "a4"}], hls.preserve, polygeist.varname = "a4"}
// CHECK-DAG: {hls.bind_storage = [{impl = "bram_ecc", type = "ram_s2p", variable = "a5"}], hls.preserve, polygeist.varname = "a5"}
int kinds(int *in) {
  int a1[16], a2[16], a3[16], a4[16], a5[16];
#pragma HLS bind_storage variable=a1 type=ram_1p
#pragma HLS bind_storage variable=a2 type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=a3 type=RAM_T2P impl=URAM latency=3
#pragma HLS bind_storage variable=a4 type=fifo impl=srl
#pragma HLS bind_storage variable=a5 type=ram_s2p impl=bram_ecc
  for (int i = 0; i < 16; ++i) {
    a1[i] = in[i];
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

// Parameters and globals.
// CHECK-LABEL: func.func @ports(
// CHECK-SAME: %arg0: memref<{{[?0-9]+}}xi32> {hls.bind_storage = [{impl = "bram", type = "ram_1p", variable = "coef"}]}
int ports(int coef[64], int i) {
#pragma HLS bind_storage variable=coef type=ram_1p impl=bram
#pragma HLS bind_storage variable=grom type=rom_2p impl=bram
  return coef[i] + grom[i];
}

// CHECK-LABEL: func.func @top(
// INLINE-DAG: {hls.bind_storage = [{type = "ram_1p", variable = "a1"}], hls.preserve, polygeist.varname = "a1"}
// NOINLINE: call @kinds
int top(int *in) { return kinds(in); }

#ifdef ERRORS
void errors(int *a) {
  int buf[8];
// ERR-DAG: bind_storage.c:[[@LINE+1]]:44: error: invalid value 'ram_3p' for option 'type' in '#pragma HLS bind_storage': expected one of: fifo, ram_1p, ram_1wnr, ram_2p, ram_s2p, ram_t2p, rom_1p, rom_2p, rom_np
#pragma HLS bind_storage variable=buf type=ram_3p
// ERR-DAG: bind_storage.c:[[@LINE+1]]:56: error: invalid value 'dsp' for option 'impl' in '#pragma HLS bind_storage': expected one of: auto, bram, bram_ecc, lutram, uram, uram_ecc, srl, memory
#pragma HLS bind_storage variable=buf type=ram_1p impl=dsp
#pragma HLS bind_storage variable=buf type=ram_1p
// ERR-DAG: bind_storage.c:[[@LINE+1]]:1: error: '#pragma HLS bind_storage' conflicts with an earlier pragma on the same variable 'buf'
#pragma HLS bind_storage variable=buf type=ram_2p
// ERR-DAG: bind_storage.c:[[@LINE+1]]:35: error: use of undeclared variable 'missing' in '#pragma HLS bind_storage'
#pragma HLS bind_storage variable=missing type=ram_1p
  buf[0] = 1;
  a[0] = buf[0];
}
#endif
