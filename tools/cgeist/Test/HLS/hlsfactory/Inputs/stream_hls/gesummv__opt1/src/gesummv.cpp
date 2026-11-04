
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
  float v1[250]
) {
  loop0: for (int v2 = 0; v2 < 250; v2++) {
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
  double v7,
  float v8
) {
  loop1: for (int v9 = 0; v9 < 250; v9++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    float v10 = v4.read();
    float v11 = v5.read();
    float v12 = v7;
    float v13 = v11 * v12;
    float v14 = v10 * v8;
    float v15 = v14 + v13;
    v6.write(v15);
  }
  return ;
}

void node2(
  float v16[250][250],
  float v17[250],
  hls::stream<float> &v18,
  float v19
) {
  float v20[250];
  loop2: for (int v21 = 0; v21 < 250; v21++) {
    loop3: for (int v22 = 0; v22 < 250; v22++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v23 = v16[v21][v22];
      float v24 = v17[v22];
      if (v22 == 0) {
        v20[v21] = v19;
      }
      float v25 = v20[v21];
      float v26 = v23 * v24;
      float v27 = v25 + v26;
      v20[v21] = v27;
      if ((v22 - 249) == 0) {
        float v28 = v20[v21];
        v18.write(v28);
      }
    }
  }
  return ;
}

void node3(
  float v29[250][250],
  float v30[250],
  hls::stream<float> &v31,
  float v32
) {
  float v33[250];
  loop4: for (int v34 = 0; v34 < 250; v34++) {
    loop5: for (int v35 = 0; v35 < 250; v35++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v36 = v29[v34][v35];
      float v37 = v30[v35];
      if (v35 == 0) {
        v33[v34] = v32;
      }
      float v38 = v33[v34];
      float v39 = v36 * v37;
      float v40 = v38 + v39;
      v33[v34] = v40;
      if ((v35 - 249) == 0) {
        float v41 = v33[v34];
        v31.write(v41);
      }
    }
  }
  return ;
}

void forward(
  float v42[250][250],
  float v43[250][250],
  float v44[250],
  float v45[250]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v46("fifo_0");
	#pragma HLS STREAM variable=v46 depth=250
  hls::stream<float> v47("fifo_1");
	#pragma HLS STREAM variable=v47 depth=250
  hls::stream<float> v48("fifo_2");
	#pragma HLS STREAM variable=v48 depth=250
  node3(v42, v44, v48, 0.000000);
  node2(v43, v44, v47, 0.000000);
  node1(v48, v47, v46, 1.200000, 1.500000);
  node0(v46, v45);
  return ;
}

