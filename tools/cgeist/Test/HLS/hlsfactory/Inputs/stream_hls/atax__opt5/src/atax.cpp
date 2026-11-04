
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
  hls::stream<float> v0[10],
  float v1[410]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=10

  loop0: for (int v2 = 0; v2 < 41; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop1: for (int v3 = 0; v3 < 10; v3++) {
      float v4 = v0[v3].read();
      v1[(v3 + (v2 * 10))] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> v5[15],
  hls::stream<float> v6[10][15],
  hls::stream<float> v7[10],
  float v8
) {
  float v9[410];
  #pragma HLS array_partition variable=v9 cyclic dim=1 factor=10

  float v10[390];
  #pragma HLS array_partition variable=v10 cyclic dim=1 factor=15

  loop2: for (int v11 = 0; v11 < 26; v11++) {
    loop3: for (int v12 = 0; v12 < 41; v12++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop4: for (int v13 = 0; v13 < 15; v13++) {
        loop5: for (int v14 = 0; v14 < 10; v14++) {
          float v15 = v6[v14][v13].read();
          if ((v14 + (v12 * 10)) == 0) {
            float v16 = v5[v13].read();
            v10[(v13 + (v11 * 15))] = v16;
          }
          float v17 = v10[(v13 + (v11 * 15))];
          if ((v13 + (v11 * 15)) == 0) {
            v9[(v14 + (v12 * 10))] = v8;
          }
          float v18 = v9[(v14 + (v12 * 10))];
          float v19 = v15 * v17;
          float v20 = v18 + v19;
          v9[(v14 + (v12 * 10))] = v20;
          if (((v13 + (v11 * 15)) - 389) == 0) {
            float v21 = v9[(v14 + (v12 * 10))];
            v7[v14].write(v21);
          }
        }
      }
    }
  }
  return ;
}

void node2(
  float v22[390][410],
  hls::stream<float> v23[10][15]
) {
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=15
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=10

  loop6: for (int v24 = 0; v24 < 26; v24++) {
    loop7: for (int v25 = 0; v25 < 41; v25++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop8: for (int v26 = 0; v26 < 15; v26++) {
        loop9: for (int v27 = 0; v27 < 10; v27++) {
          float v28 = v22[(v26 + (v24 * 15))][(v27 + (v25 * 10))];
          v23[v27][v26].write(v28);
        }
      }
    }
  }
  return ;
}

void node3(
  float v29[390][410],
  float v30[410],
  hls::stream<float> v31[15],
  float v32
) {
  #pragma HLS array_partition variable=v29 cyclic dim=1 factor=15
  #pragma HLS array_partition variable=v29 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v30 cyclic dim=1 factor=10

  float v33[390];
  #pragma HLS array_partition variable=v33 cyclic dim=1 factor=15

  loop10: for (int v34 = 0; v34 < 41; v34++) {
    loop11: for (int v35 = 0; v35 < 26; v35++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop12: for (int v36 = 0; v36 < 10; v36++) {
        loop13: for (int v37 = 0; v37 < 15; v37++) {
          float v38 = v29[(v37 + (v35 * 15))][(v36 + (v34 * 10))];
          float v39 = v30[(v36 + (v34 * 10))];
          if ((v36 + (v34 * 10)) == 0) {
            v33[(v37 + (v35 * 15))] = v32;
          }
          float v40 = v33[(v37 + (v35 * 15))];
          float v41 = v38 * v39;
          float v42 = v40 + v41;
          v33[(v37 + (v35 * 15))] = v42;
          if (((v36 + (v34 * 10)) - 409) == 0) {
            float v43 = v33[(v37 + (v35 * 15))];
            v31[v37].write(v43);
          }
        }
      }
    }
  }
  return ;
}

void forward(
  float v44[390][410],
  float v45[410],
  float v46[410]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v44 cyclic dim=1 factor=15
  #pragma HLS array_partition variable=v44 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v45 cyclic dim=1 factor=10

  #pragma HLS array_partition variable=v46 cyclic dim=1 factor=10

  hls::stream<float> v47[10];
	#pragma HLS STREAM variable=v47 depth=41
  hls::stream<float> v48[10][15];
	#pragma HLS STREAM variable=v48 depth=1066
  hls::stream<float> v49[15];
	#pragma HLS STREAM variable=v49 depth=26
  node3(v44, v45, v49, 0.000000);
  node2(v44, v48);
  node1(v49, v48, v47, 0.000000);
  node0(v47, v46);
  return ;
}

