
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
  hls::stream<float> &v5,
  hls::stream<float> &v6,
  float v7
) {
  float v8[410];
  float v9[390];
  loop1: for (int v10 = 0; v10 < 390; v10++) {
    loop2: for (int v11 = 0; v11 < 410; v11++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v12 = v4.read();
      if (v11 == 0) {
        float v13 = v5.read();
        v9[v10] = v13;
      }
      float v14 = v9[v10];
      if (v10 == 0) {
        v8[v11] = v7;
      }
      float v15 = v8[v11];
      float v16 = v12 * v14;
      float v17 = v15 + v16;
      v8[v11] = v17;
      if ((v10 - 389) == 0) {
        float v18 = v8[v11];
        v6.write(v18);
      }
    }
  }
  return ;
}

void node2(
  float v19[390][410],
  hls::stream<float> &v20
) {
  loop3: for (int v21 = 0; v21 < 390; v21++) {
    loop4: for (int v22 = 0; v22 < 410; v22++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v23 = v19[v21][v22];
      v20.write(v23);
    }
  }
  return ;
}

void node3(
  float v24[390][410],
  float v25[410],
  hls::stream<float> &v26,
  float v27
) {
  float v28[390];
  loop5: for (int v29 = 0; v29 < 410; v29++) {
    loop6: for (int v30 = 0; v30 < 390; v30++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v31 = v24[v30][v29];
      float v32 = v25[v29];
      if (v29 == 0) {
        v28[v30] = v27;
      }
      float v33 = v28[v30];
      float v34 = v31 * v32;
      float v35 = v33 + v34;
      v28[v30] = v35;
      if ((v29 - 409) == 0) {
        float v36 = v28[v30];
        v26.write(v36);
      }
    }
  }
  return ;
}

void forward(
  float v37[390][410],
  float v38[410],
  float v39[410]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v40("fifo_0");
	#pragma HLS STREAM variable=v40 depth=410
  hls::stream<float> v41("fifo_1");
	#pragma HLS STREAM variable=v41 depth=159900
  hls::stream<float> v42("fifo_2");
	#pragma HLS STREAM variable=v42 depth=390
  node3(v37, v38, v42, 0.000000);
  node2(v37, v41);
  node1(v41, v42, v40, 0.000000);
  node0(v40, v39);
  return ;
}

