
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
  hls::stream<float> v0[16],
  float v1[400]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=16

  loop0: for (int v2 = 0; v2 < 25; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop1: for (int v3 = 0; v3 < 16; v3++) {
      float v4 = v0[v3].read();
      v1[(v3 + (v2 * 16))] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> v5[16],
  float v6[400]
) {
  #pragma HLS array_partition variable=v6 cyclic dim=1 factor=16

  loop2: for (int v7 = 0; v7 < 25; v7++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop3: for (int v8 = 0; v8 < 16; v8++) {
      float v9 = v5[v8].read();
      v6[(v8 + (v7 * 16))] = v9;
    }
  }
  return ;
}

void node2(
  hls::stream<float> v10[16][16],
  float v11[400],
  hls::stream<float> v12[16],
  float v13
) {
  #pragma HLS array_partition variable=v11 cyclic dim=1 factor=16

  float v14[400];
  #pragma HLS array_partition variable=v14 cyclic dim=1 factor=16

  loop4: for (int v15 = 0; v15 < 25; v15++) {
    loop5: for (int v16 = 0; v16 < 25; v16++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v17 = 0; v17 < 16; v17++) {
        loop7: for (int v18 = 0; v18 < 16; v18++) {
          float v19 = v10[v18][v17].read();
          float v20 = v11[(v17 + (v15 * 16))];
          if ((v17 + (v15 * 16)) == 0) {
            v14[(v18 + (v16 * 16))] = v13;
          }
          float v21 = v14[(v18 + (v16 * 16))];
          float v22 = v19 * v20;
          float v23 = v21 + v22;
          v14[(v18 + (v16 * 16))] = v23;
          if (((v17 + (v15 * 16)) - 399) == 0) {
            float v24 = v14[(v18 + (v16 * 16))];
            v12[v18].write(v24);
          }
        }
      }
    }
  }
  return ;
}

void node3(
  float v25[400][400],
  hls::stream<float> v26[16][16]
) {
  #pragma HLS array_partition variable=v25 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v25 cyclic dim=2 factor=16

  loop8: for (int v27 = 0; v27 < 25; v27++) {
    loop9: for (int v28 = 0; v28 < 25; v28++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop10: for (int v29 = 0; v29 < 16; v29++) {
        loop11: for (int v30 = 0; v30 < 16; v30++) {
          float v31 = v25[(v29 + (v27 * 16))][(v30 + (v28 * 16))];
          v26[v30][v29].write(v31);
        }
      }
    }
  }
  return ;
}

void node4(
  float v32[400][400],
  float v33[400],
  hls::stream<float> v34[16],
  float v35
) {
  #pragma HLS array_partition variable=v32 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v32 cyclic dim=2 factor=16

  #pragma HLS array_partition variable=v33 cyclic dim=1 factor=16

  float v36[400];
  #pragma HLS array_partition variable=v36 cyclic dim=1 factor=16

  loop12: for (int v37 = 0; v37 < 25; v37++) {
    loop13: for (int v38 = 0; v38 < 25; v38++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop14: for (int v39 = 0; v39 < 16; v39++) {
        loop15: for (int v40 = 0; v40 < 16; v40++) {
          float v41 = v32[(v40 + (v38 * 16))][(v39 + (v37 * 16))];
          float v42 = v33[(v39 + (v37 * 16))];
          if ((v39 + (v37 * 16)) == 0) {
            v36[(v40 + (v38 * 16))] = v35;
          }
          float v43 = v36[(v40 + (v38 * 16))];
          float v44 = v41 * v42;
          float v45 = v43 + v44;
          v36[(v40 + (v38 * 16))] = v45;
          if (((v39 + (v37 * 16)) - 399) == 0) {
            float v46 = v36[(v40 + (v38 * 16))];
            v34[v40].write(v46);
          }
        }
      }
    }
  }
  return ;
}

void forward(
  float v47[400][400],
  float v48[400][400],
  float v49[400],
  float v50[400],
  float v51[400],
  float v52[400]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v47 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v47 cyclic dim=2 factor=16

  #pragma HLS array_partition variable=v48 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v48 cyclic dim=2 factor=16

  #pragma HLS array_partition variable=v49 cyclic dim=1 factor=16

  #pragma HLS array_partition variable=v50 cyclic dim=1 factor=16

  #pragma HLS array_partition variable=v51 cyclic dim=1 factor=16

  #pragma HLS array_partition variable=v52 cyclic dim=1 factor=16

  hls::stream<float> v53[16];
	#pragma HLS STREAM variable=v53 depth=25
  hls::stream<float> v54[16][16];
	#pragma HLS STREAM variable=v54 depth=625
  hls::stream<float> v55[16];
	#pragma HLS STREAM variable=v55 depth=25
  node4(v47, v49, v55, 0.000000);
  node3(v48, v54);
  node2(v54, v50, v53, 0.000000);
  node1(v55, v51);
  node0(v53, v52);
  return ;
}

