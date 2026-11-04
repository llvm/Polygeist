
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
  hls::stream<float> &v0,
  float v1[200][220]
) {
  loop0: for (int v2 = 0; v2 < 200; v2++) {
    loop1: for (int v3 = 0; v3 < 220; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v4 = v0.read();
      v1[v2][v3] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> &v5,
  float v6[200][220],
  hls::stream<float> &v7,
  double v8,
  float v9
) {
  loop2: for (int v10 = 0; v10 < 200; v10++) {
    loop3: for (int v11 = 0; v11 < 220; v11++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v12 = v5.read();
      float v13 = v6[v10][v11];
      float v14 = v8;
      float v15 = v13 * v14;
      float v16 = v12 * v9;
      float v17 = v16 + v15;
      v7.write(v17);
    }
  }
  return ;
}

void node2(
  float v18[200][240],
  float v19[240][220],
  hls::stream<float> &v20,
  float v21
) {
  float v22[200][220];
  loop4: for (int v23 = 0; v23 < 200; v23++) {
    loop5: for (int v24 = 0; v24 < 220; v24++) {
      loop6: for (int v25 = 0; v25 < 240; v25++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v26 = v18[v23][v25];
        float v27 = v19[v25][v24];
        if (v25 == 0) {
          v22[v23][v24] = v21;
        }
        float v28 = v22[v23][v24];
        float v29 = v26 * v27;
        float v30 = v28 + v29;
        v22[v23][v24] = v30;
        if ((v25 - 239) == 0) {
          float v31 = v22[v23][v24];
          v20.write(v31);
        }
      }
    }
  }
  return ;
}

void forward(
  float v32[200][240],
  float v33[240][220],
  float v34[200][220],
  float v35[200][220]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v36("fifo_0");
	#pragma HLS STREAM variable=v36 depth=44000
  hls::stream<float> v37("fifo_1");
	#pragma HLS STREAM variable=v37 depth=44000
  node2(v32, v33, v37, 0.000000);
  node1(v37, v34, v36, 0.100000, 0.500000);
  node0(v36, v35);
  return ;
}

