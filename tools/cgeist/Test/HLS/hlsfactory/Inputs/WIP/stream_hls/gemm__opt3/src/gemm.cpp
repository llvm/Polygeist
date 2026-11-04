
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_half.h>
#include <cassert>
#include <hls_math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace std;

void node0(
  hls::stream<float> v0[4][11],
  float v1[200][220]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=11

  loop0: for (int v2 = 0; v2 < 50; v2++) {
    loop1: for (int v3 = 0; v3 < 20; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 4; v4++) {
        loop3: for (int v5 = 0; v5 < 11; v5++) {
          float v6 = v0[v4][v5].read();
          v1[(v4 + (v2 * 4))][(v5 + (v3 * 11))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[4][11],
  float v8[200][220],
  hls::stream<float> v9[4][11],
  double v10,
  float v11
) {
  #pragma HLS array_partition variable=v8 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v8 cyclic dim=2 factor=11

  loop4: for (int v12 = 0; v12 < 50; v12++) {
    loop5: for (int v13 = 0; v13 < 20; v13++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v14 = 0; v14 < 4; v14++) {
        loop7: for (int v15 = 0; v15 < 11; v15++) {
          float v16 = v7[v14][v15].read();
          float v17 = v8[(v14 + (v12 * 4))][(v15 + (v13 * 11))];
          float v18 = v10;
          float v19 = v17 * v18;
          float v20 = v16 * v11;
          float v21 = v20 + v19;
          v9[v14][v15].write(v21);
        }
      }
    }
  }
  return ;
}

void node2(
  float v22[240][220],
  float v23[200][240],
  hls::stream<float> v24[4][11],
  float v25
) {
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=11

  #pragma HLS array_partition variable=v23 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v23 cyclic dim=2 factor=10

  float v26[200][220];
  #pragma HLS array_partition variable=v26 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v26 cyclic dim=2 factor=11

  loop8: for (int v27 = 0; v27 < 50; v27++) {
    loop9: for (int v28 = 0; v28 < 20; v28++) {
      loop10: for (int v29 = 0; v29 < 24; v29++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v30 = 0; v30 < 4; v30++) {
          loop12: for (int v31 = 0; v31 < 11; v31++) {
            loop13: for (int v32 = 0; v32 < 10; v32++) {
              float v33 = v23[(v30 + (v27 * 4))][(v32 + (v29 * 10))];
              float v34 = v22[(v32 + (v29 * 10))][(v31 + (v28 * 11))];
              if ((v32 + (v29 * 10)) == 0) {
                v26[(v30 + (v27 * 4))][(v31 + (v28 * 11))] = v25;
              }
              float v35 = v26[(v30 + (v27 * 4))][(v31 + (v28 * 11))];
              float v36 = v33 * v34;
              float v37 = v35 + v36;
              v26[(v30 + (v27 * 4))][(v31 + (v28 * 11))] = v37;
              if (((v32 + (v29 * 10)) - 239) == 0) {
                float v38 = v26[(v30 + (v27 * 4))][(v31 + (v28 * 11))];
                v24[v30][v31].write(v38);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void forward(
  float v39[200][240],
  float v40[240][220],
  float v41[200][220],
  float v42[200][220]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v39 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v39 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v40 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v40 cyclic dim=2 factor=11

  #pragma HLS array_partition variable=v41 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v41 cyclic dim=2 factor=11

  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v42 cyclic dim=2 factor=11

  hls::stream<float> v43[4][11];
	#pragma HLS STREAM variable=v43 depth=1000
  hls::stream<float> v44[4][11];
	#pragma HLS STREAM variable=v44 depth=1000
  node2(v40, v39, v44, 0.000000);
  node1(v44, v41, v43, 0.100000, 0.500000);
  node0(v43, v42);
  return ;
}

