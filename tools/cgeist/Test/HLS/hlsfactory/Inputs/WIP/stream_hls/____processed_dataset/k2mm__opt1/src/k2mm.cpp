
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
  float v1[180][220]
) {
  loop0: for (int v2 = 0; v2 < 180; v2++) {
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
  float v6[180][220],
  hls::stream<float> &v7,
  double v8,
  float v9
) {
  loop2: for (int v10 = 0; v10 < 180; v10++) {
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
  hls::stream<float> &v18,
  float v19[190][220],
  hls::stream<float> &v20,
  float v21
) {
  float v22[180][220];
  float v23[180][190];
  loop4: for (int v24 = 0; v24 < 180; v24++) {
    loop5: for (int v25 = 0; v25 < 220; v25++) {
      loop6: for (int v26 = 0; v26 < 190; v26++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v25 == 0) {
          float v27 = v18.read();
          v23[v24][v26] = v27;
        }
        float v28 = v23[v24][v26];
        float v29 = v19[v26][v25];
        if (v26 == 0) {
          v22[v24][v25] = v21;
        }
        float v30 = v22[v24][v25];
        float v31 = v28 * v29;
        float v32 = v30 + v31;
        v22[v24][v25] = v32;
        if ((v26 - 189) == 0) {
          float v33 = v22[v24][v25];
          v20.write(v33);
        }
      }
    }
  }
  return ;
}

void node3(
  float v34[180][210],
  float v35[210][190],
  hls::stream<float> &v36,
  float v37
) {
  float v38[180][190];
  loop7: for (int v39 = 0; v39 < 180; v39++) {
    loop8: for (int v40 = 0; v40 < 190; v40++) {
      loop9: for (int v41 = 0; v41 < 210; v41++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v42 = v34[v39][v41];
        float v43 = v35[v41][v40];
        if (v41 == 0) {
          v38[v39][v40] = v37;
        }
        float v44 = v38[v39][v40];
        float v45 = v42 * v43;
        float v46 = v44 + v45;
        v38[v39][v40] = v46;
        if ((v41 - 209) == 0) {
          float v47 = v38[v39][v40];
          v36.write(v47);
        }
      }
    }
  }
  return ;
}

void forward(
  float v48[180][210],
  float v49[210][190],
  float v50[190][220],
  float v51[180][220],
  float v52[180][220]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v53("fifo_0");
	#pragma HLS STREAM variable=v53 depth=39600
  hls::stream<float> v54("fifo_1");
	#pragma HLS STREAM variable=v54 depth=39600
  hls::stream<float> v55("fifo_2");
	#pragma HLS STREAM variable=v55 depth=34200
  node3(v48, v49, v55, 0.000000);
  node2(v55, v50, v54, 0.000000);
  node1(v54, v51, v53, 0.100000, 0.500000);
  node0(v53, v52);
  return ;
}

