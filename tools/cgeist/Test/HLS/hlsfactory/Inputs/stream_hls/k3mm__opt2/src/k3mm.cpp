
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <hls_half.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace std;

void node0(hls::stream<float> &v0, float v1[180][210]) {
loop0:
  for (int v2 = 0; v2 < 210; v2++) {
  loop1:
    for (int v3 = 0; v3 < 180; v3++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_flatten
      float v4 = v0.read();
      v1[v3][v2] = v4;
    }
  }
  return;
}

void node1(hls::stream<float> &v5, hls::stream<float> &v6,
           hls::stream<float> &v7, float v8) {
  float v9[180][210];
  float v10;
  float v11[180];
loop2:
  for (int v12 = 0; v12 < 190; v12++) {
  loop3:
    for (int v13 = 0; v13 < 210; v13++) {
    loop4:
      for (int v14 = 0; v14 < 180; v14++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_flatten
        if (v13 == 0) {
          float v15 = v5.read();
          v11[v14] = v15;
        }
        float v16 = v11[v14];
        if (v14 == 0) {
          float v17 = v6.read();
          v10 = v17;
        }
        float v18 = v10;
        if (v12 == 0) {
          v9[v14][v13] = v8;
        }
        float v19 = v9[v14][v13];
        float v20 = v16 * v18;
        float v21 = v19 + v20;
        v9[v14][v13] = v21;
        if ((v12 - 189) == 0) {
          float v22 = v9[v14][v13];
          v7.write(v22);
        }
      }
    }
  }
  return;
}

void node2(float v23[190][220], float v24[220][210], hls::stream<float> &v25,
           float v26) {
  float v27[190][210];
loop5:
  for (int v28 = 0; v28 < 190; v28++) {
  loop6:
    for (int v29 = 0; v29 < 220; v29++) {
    loop7:
      for (int v30 = 0; v30 < 210; v30++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_flatten
        float v31 = v23[v28][v29];
        float v32 = v24[v29][v30];
        if (v29 == 0) {
          v27[v28][v30] = v26;
        }
        float v33 = v27[v28][v30];
        float v34 = v31 * v32;
        float v35 = v33 + v34;
        v27[v28][v30] = v35;
        if ((v29 - 219) == 0) {
          float v36 = v27[v28][v30];
          v25.write(v36);
        }
      }
    }
  }
  return;
}

void node3(float v37[180][200], float v38[200][190], hls::stream<float> &v39,
           float v40) {
  float v41[180][190];
loop8:
  for (int v42 = 0; v42 < 190; v42++) {
  loop9:
    for (int v43 = 0; v43 < 200; v43++) {
    loop10:
      for (int v44 = 0; v44 < 180; v44++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_flatten
        float v45 = v37[v44][v43];
        float v46 = v38[v43][v42];
        if (v43 == 0) {
          v41[v44][v42] = v40;
        }
        float v47 = v41[v44][v42];
        float v48 = v45 * v46;
        float v49 = v47 + v48;
        v41[v44][v42] = v49;
        if ((v43 - 199) == 0) {
          float v50 = v41[v44][v42];
          v39.write(v50);
        }
      }
    }
  }
  return;
}

void forward(float v51[180][200], float v52[200][190], float v53[190][220],
             float v54[220][210], float v55[180][210]) {
#pragma HLS DATAFLOW
  hls::stream<float> v56("fifo_0");
#pragma HLS STREAM variable = v56 depth = 37800
  hls::stream<float> v57("fifo_1");
#pragma HLS STREAM variable = v57 depth = 39900
  hls::stream<float> v58("fifo_2");
#pragma HLS STREAM variable = v58 depth = 34200
  node3(v51, v52, v58, 0.000000);
  node2(v53, v54, v57, 0.000000);
  node1(v58, v57, v56, 0.000000);
  node0(v56, v55);
  return;
}
