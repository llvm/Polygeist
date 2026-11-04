// With --hls-trip-one-pipeline-unroll, a pipeline pragma on a loop whose trip
// count is 1 (VITIS_NOTES N40, Vitis
// 2025.1 kernels w9p5_t1_flat / w9p5_t1_noflat): Vitis completely unrolls the
// loops inside it, then removes the one-iteration loop with its pipeline and
// loop_flatten pragmas, and automatic pipelining decides on the rest of the
// nest. cgeist gives every loop nested in it a complete unroll (a partial
// unroll pragma of its own is overridden to complete, as Vitis' [HLS 214-291]
// does, kernel w9rv_t1_part) and withdraws the pipeline pragma. A
// trip count of 2 keeps the pragma (w9p5_t2_flat).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-trip-one-pipeline-unroll -o - 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>&1 | FileCheck %s --check-prefix=OFF
// The option is off by default (OPEN_ISSUES O57): the pragma is lost with
// the folded loop, and the inner loops keep no unroll pragma.
// OFF: warning: '#pragma HLS pipeline': the loop it applies to was removed
// OFF-LABEL: func.func @trip_one
// OFF-NOT: factor = 0
// OFF: } {hls.unroll = [{factor = 2 : i32, skip_exit_check = false}]}
// OFF-NOT: factor = 0
// OFF-LABEL: func.func @trip_two

// CHECK: trip_one.c:[[@LINE+12]]:1: warning: '#pragma HLS pipeline' is on a loop with a trip count of 1: the loops inside it are completely unrolled and the loop is removed with its pragmas, as Vitis does
// CHECK: trip_one.c:[[@LINE+12]]:1: warning: '#pragma HLS loop_flatten': the loop it applies to was removed
// CHECK-NOT: '#pragma HLS pipeline': the loop it applies to was removed
// CHECK-LABEL: func.func @trip_one
// CHECK-NOT: hls.pipeline
// CHECK-NOT: factor = 2
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}
// CHECK-NOT: factor = 2
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}
void trip_one(int a[4][8][8], int o[4][8]) {
  for (int i = 0; i < 4; i++)
    for (int t = 0; t < 1; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_flatten
      for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
#pragma HLS unroll factor=2
          o[i][j] += a[i][j][k];
        }
    }
}

// CHECK-LABEL: func.func @trip_two
// CHECK: } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void trip_two(int a[4][8][8], int o[4][8]) {
  for (int i = 0; i < 4; i++)
    for (int t = 0; t < 2; t++) {
#pragma HLS pipeline II=1
      for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++)
          o[i][j] += a[i][j][k] + t;
    }
}
