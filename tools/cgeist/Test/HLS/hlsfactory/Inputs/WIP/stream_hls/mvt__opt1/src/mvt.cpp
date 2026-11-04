
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
  float v1[400]
) {
  loop0: for (int v2 = 0; v2 < 400; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    float v3 = v0.read();
    v1[v2] = v3;
  }
  return ;
}

void node1(
  hls::stream<float> &v4,
  float v5[400]
) {
  loop1: for (int v6 = 0; v6 < 400; v6++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    float v7 = v4.read();
    v5[v6] = v7;
  }
  return ;
}

void node2(
  float v8[400][400],
  float v9[400],
  hls::stream<float> &v10,
  float v11
) {
  float v12[400];
  loop2: for (int v13 = 0; v13 < 400; v13++) {
    loop3: for (int v14 = 0; v14 < 400; v14++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v15 = v8[v13][v14];
      float v16 = v9[v14];
      if (v14 == 0) {
        v12[v13] = v11;
      }
      float v17 = v12[v13];
      float v18 = v15 * v16;
      float v19 = v17 + v18;
      v12[v13] = v19;
      if ((v14 - 399) == 0) {
        float v20 = v12[v13];
        v10.write(v20);
      }
    }
  }
  return ;
}

void node3(
  float v21[400][400],
  float v22[400][400]
) {
  loop4: for (int v23 = 0; v23 < 400; v23++) {
    loop5: for (int v24 = 0; v24 < 400; v24++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v25 = v21[v23][v24];
      v22[v24][v23] = v25;
    }
  }
  return ;
}

void node4(
  float v26[400][400],
  float v27[400],
  hls::stream<float> &v28,
  float v29
) {
  float v30[400];
  loop6: for (int v31 = 0; v31 < 400; v31++) {
    loop7: for (int v32 = 0; v32 < 400; v32++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v33 = v26[v31][v32];
      float v34 = v27[v32];
      if (v32 == 0) {
        v30[v31] = v29;
      }
      float v35 = v30[v31];
      float v36 = v33 * v34;
      float v37 = v35 + v36;
      v30[v31] = v37;
      if ((v32 - 399) == 0) {
        float v38 = v30[v31];
        v28.write(v38);
      }
    }
  }
  return ;
}

void forward(
  float v39[400][400],
  float v40[400][400],
  float v41[400],
  float v42[400],
  float v43[400],
  float v44[400]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v45("fifo_0");
	#pragma HLS STREAM variable=v45 depth=400
  float v46[400][400];
  hls::stream<float> v47("fifo_1");
	#pragma HLS STREAM variable=v47 depth=400
  node4(v39, v41, v47, 0.000000);
  node3(v40, v46);
  node2(v46, v42, v45, 0.000000);
  node1(v47, v43);
  node0(v45, v44);
  return ;
}

