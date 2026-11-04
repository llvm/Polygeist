
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
  float v1[410]
) {
  loop0: for (int v2 = 0; v2 < 410; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    float v3 = v0.read();
    v1[v2] = v3;
  }
  return ;
}

void node1(
  hls::stream<float> &v4,
  float v5[390]
) {
  loop1: for (int v6 = 0; v6 < 390; v6++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    float v7 = v4.read();
    v5[v6] = v7;
  }
  return ;
}

void node2(
  float v8[410][390],
  float v9[390],
  hls::stream<float> &v10,
  float v11
) {
  float v12[410];
  loop2: for (int v13 = 0; v13 < 410; v13++) {
    loop3: for (int v14 = 0; v14 < 390; v14++) {
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
      if ((v14 - 389) == 0) {
        float v20 = v12[v13];
        v10.write(v20);
      }
    }
  }
  return ;
}

void node3(
  float v21[410],
  float v22[410][390],
  hls::stream<float> &v23,
  float v24
) {
  float v25[390];
  loop4: for (int v26 = 0; v26 < 390; v26++) {
    loop5: for (int v27 = 0; v27 < 410; v27++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v28 = v21[v27];
      float v29 = v22[v27][v26];
      if (v27 == 0) {
        v25[v26] = v24;
      }
      float v30 = v25[v26];
      float v31 = v28 * v29;
      float v32 = v30 + v31;
      v25[v26] = v32;
      if ((v27 - 409) == 0) {
        float v33 = v25[v26];
        v23.write(v33);
      }
    }
  }
  return ;
}

void forward(
  float v34[410][390],
  float v35[410][390],
  float v36[410],
  float v37[390],
  float v38[390],
  float v39[410]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v40("fifo_0");
	#pragma HLS STREAM variable=v40 depth=410
  hls::stream<float> v41("fifo_1");
	#pragma HLS STREAM variable=v41 depth=390
  node3(v36, v34, v41, 0.000000);
  node2(v35, v37, v40, 0.000000);
  node1(v41, v38);
  node0(v40, v39);
  return ;
}

