// Two array_partition pragmas on the same dimension of one array are merged
// the way Vitis 2025.1 merges them instead of being rejected (MEMOS decision
// #33; VITIS_NOTES N17 (c) and N28 in the hls-est tree; hls-doc
// frontend-pragmas.md): the same type keeps the larger factor whatever the
// order, `complete` wins over cyclic or block in either order, `dim=0` is
// split per dimension first, and cyclic against block stays an error (Vitis:
// [HLS 214-290] Multiple conflicting array_partition). The pragma that loses
// gets a remark. --hls-reject-partition-conflicts restores the old error.
// RUN: cgeist %s --function='*' -S --hls-annotate -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: not cgeist %s --function='*' -S --hls-annotate -DKIND_CONFLICT -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not cgeist %s --function='*' -S --hls-annotate --hls-reject-partition-conflicts -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD

static int G[64][32];

// fbh6_conf_up: cyclic 2, then cyclic 4 on dim 1 -> factor 4.
// CHECK-LABEL: memref.global "private" @G
// CHECK-SAME: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "G"}]
// REMARK-DAG: array-partition-merge.c:[[@LINE+4]]:1: remark: '#pragma HLS array_partition' (cyclic factor=2) on dimension 1 of variable 'G' is dropped: the later pragma (cyclic factor=4) partitions it more finely, which Vitis keeps
// REMARK-DAG: array-partition-merge.c:[[@LINE+4]]:1: note: later pragma is here
// OLD: array-partition-merge.c:[[@LINE+3]]:1: error: '#pragma HLS array_partition' conflicts with an earlier pragma on the same variable 'G'
void up(int in[64][32], int out[64][32]) {
#pragma HLS array_partition variable=G cyclic factor=2 dim=1
#pragma HLS array_partition variable=G cyclic factor=4 dim=1
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      G[i][j] = in[i][j] * 3;
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      out[i][j] = G[i][j] + G[63 - i][j];
}

// fbh6_conf_down / w10k_loc_cyc4_cyc2: cyclic 4, then cyclic 2 -> factor 4.
// CHECK-LABEL: func.func @down(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "L"}]
// REMARK-DAG: array-partition-merge.c:[[@LINE+4]]:1: remark: '#pragma HLS array_partition' (cyclic factor=2) on dimension 1 of variable 'L' is dropped: the earlier pragma (cyclic factor=4) partitions it at least as finely, which Vitis keeps
void down(int in[64][32], int out[64][32]) {
  int L[64][32];
#pragma HLS array_partition variable=L cyclic factor=4 dim=1
#pragma HLS array_partition variable=L cyclic factor=2 dim=1
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      L[i][j] = in[i][j] * 3;
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      out[i][j] = L[i][j] + L[63 - i][j];
}

// w10k_blk2_blk4: block 2, then block 4 -> block 4; w10k_cyc4_cpl /
// w10k_cpl_cyc4 / w10k_blk4_cpl: complete wins in either order.
// CHECK-LABEL: func.func @kinds(
// CHECK-DAG: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "block", variable = "B"}]
// CHECK-DAG: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "C1"}]
// CHECK-DAG: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "C2"}]
void kinds(int in[64][32], int out[64][32]) {
  int B[64][32], C1[64][32], C2[64][32];
#pragma HLS array_partition variable=B block factor=2 dim=1
#pragma HLS array_partition variable=B block factor=4 dim=1
#pragma HLS array_partition variable=C1 cyclic factor=4 dim=1
#pragma HLS array_partition variable=C1 complete dim=1
#pragma HLS array_partition variable=C2 complete dim=1
#pragma HLS array_partition variable=C2 block factor=4 dim=1
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++) {
      B[i][j] = in[i][j];
      C1[i][j] = in[i][j] + 1;
      C2[i][j] = in[i][j] + 2;
    }
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      out[i][j] = B[63 - i][j] + C1[63 - i][j] + C2[63 - i][j];
}

// w10k_d0c2_d1c4: cyclic 2 on every dimension, then cyclic 4 on dim 1 ->
// factor 4 on dimension 1 and factor 2 on dimension 2.
// CHECK-LABEL: func.func @dim0(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "D"}, {dim = 2 : i32, factor = 2 : i32, kind = "cyclic", variable = "D"}]
void dim0(int in[64][32], int out[64][32]) {
  int D[64][32];
#pragma HLS array_partition variable=D cyclic factor=2 dim=0
#pragma HLS array_partition variable=D cyclic factor=4 dim=1
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      D[i][j] = in[i][j];
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      out[i][j] = D[63 - i][j] + D[i][31 - j];
}

// w10pt_d1c4_d0c2: the same pragmas in the other order give the same result.
// CHECK-LABEL: func.func @dim0_rev(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "E"}, {dim = 2 : i32, factor = 2 : i32, kind = "cyclic", variable = "E"}]
void dim0_rev(int in[64][32], int out[64][32]) {
  int E[64][32];
#pragma HLS array_partition variable=E cyclic factor=4 dim=1
#pragma HLS array_partition variable=E cyclic factor=2 dim=0
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      E[i][j] = in[i][j];
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      out[i][j] = E[63 - i][j] + E[i][31 - j];
}

#ifdef KIND_CONFLICT
// w10k_cyc2_blk4 / w10k_blk4_cyc2 / w10k_cyc4_blk4: Vitis rejects cyclic
// against block on one dimension, whatever the factors.
// CONFLICT: array-partition-merge.c:[[@LINE+4]]:1: error: '#pragma HLS array_partition' conflicts with an earlier pragma on the same variable 'K'
void kind_conflict(int in[64], int out[64]) {
  int K[64];
#pragma HLS array_partition variable=K cyclic factor=4 dim=1
#pragma HLS array_partition variable=K block factor=4 dim=1
  for (int i = 0; i < 64; i++)
    K[i] = in[i];
  for (int i = 0; i < 64; i++)
    out[i] = K[63 - i];
}
#endif
