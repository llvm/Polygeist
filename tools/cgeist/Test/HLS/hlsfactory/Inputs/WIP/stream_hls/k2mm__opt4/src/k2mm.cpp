
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
  float v1[180][220]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=11

  loop0: for (int v2 = 0; v2 < 20; v2++) {
    loop1: for (int v3 = 0; v3 < 45; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 11; v4++) {
        loop3: for (int v5 = 0; v5 < 4; v5++) {
          float v6 = v0[v5][v4].read();
          v1[(v5 + (v3 * 4))][(v4 + (v2 * 11))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[4][11],
  float v8[180][220],
  hls::stream<float> v9[4][11],
  double v10,
  float v11
) {
  #pragma HLS array_partition variable=v8 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v8 cyclic dim=2 factor=11

  loop4: for (int v12 = 0; v12 < 20; v12++) {
    loop5: for (int v13 = 0; v13 < 45; v13++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v14 = 0; v14 < 11; v14++) {
        loop7: for (int v15 = 0; v15 < 4; v15++) {
          float v16 = v7[v15][v14].read();
          float v17 = v8[(v15 + (v13 * 4))][(v14 + (v12 * 11))];
          float v18 = v10;
          float v19 = v17 * v18;
          float v20 = v16 * v11;
          float v21 = v20 + v19;
          v9[v15][v14].write(v21);
        }
      }
    }
  }
  return ;
}

void node2(
  float v22[190][220],
  hls::stream<float> v23[4][5],
  hls::stream<float> v24[4][11],
  float v25
) {
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=5
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=11

  float v26[180][220];
  #pragma HLS array_partition variable=v26 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v26 cyclic dim=2 factor=11

  float v27[180][190];
  #pragma HLS array_partition variable=v27 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v27 cyclic dim=2 factor=5

  loop8: for (int v28 = 0; v28 < 38; v28++) {
    loop9: for (int v29 = 0; v29 < 20; v29++) {
      loop10: for (int v30 = 0; v30 < 45; v30++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v31 = 0; v31 < 5; v31++) {
          loop12: for (int v32 = 0; v32 < 11; v32++) {
            loop13: for (int v33 = 0; v33 < 4; v33++) {
              if ((v32 + (v29 * 11)) == 0) {
                float v34 = v23[v33][v31].read();
                v27[(v33 + (v30 * 4))][(v31 + (v28 * 5))] = v34;
              }
              float v35 = v27[(v33 + (v30 * 4))][(v31 + (v28 * 5))];
              float v36 = v22[(v31 + (v28 * 5))][(v32 + (v29 * 11))];
              if ((v31 + (v28 * 5)) == 0) {
                v26[(v33 + (v30 * 4))][(v32 + (v29 * 11))] = v25;
              }
              float v37 = v26[(v33 + (v30 * 4))][(v32 + (v29 * 11))];
              float v38 = v35 * v36;
              float v39 = v37 + v38;
              v26[(v33 + (v30 * 4))][(v32 + (v29 * 11))] = v39;
              if (((v31 + (v28 * 5)) - 189) == 0) {
                float v40 = v26[(v33 + (v30 * 4))][(v32 + (v29 * 11))];
                v24[v33][v32].write(v40);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node3(
  float v41[210][190],
  float v42[180][210],
  hls::stream<float> v43[4][5],
  float v44
) {
  #pragma HLS array_partition variable=v41 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v41 cyclic dim=2 factor=5

  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v42 cyclic dim=2 factor=10

  float v45[180][190];
  #pragma HLS array_partition variable=v45 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v45 cyclic dim=2 factor=5

  loop14: for (int v46 = 0; v46 < 38; v46++) {
    loop15: for (int v47 = 0; v47 < 21; v47++) {
      loop16: for (int v48 = 0; v48 < 45; v48++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop17: for (int v49 = 0; v49 < 5; v49++) {
          loop18: for (int v50 = 0; v50 < 10; v50++) {
            loop19: for (int v51 = 0; v51 < 4; v51++) {
              float v52 = v42[(v51 + (v48 * 4))][(v50 + (v47 * 10))];
              float v53 = v41[(v50 + (v47 * 10))][(v49 + (v46 * 5))];
              if ((v50 + (v47 * 10)) == 0) {
                v45[(v51 + (v48 * 4))][(v49 + (v46 * 5))] = v44;
              }
              float v54 = v45[(v51 + (v48 * 4))][(v49 + (v46 * 5))];
              float v55 = v52 * v53;
              float v56 = v54 + v55;
              v45[(v51 + (v48 * 4))][(v49 + (v46 * 5))] = v56;
              if (((v50 + (v47 * 10)) - 209) == 0) {
                float v57 = v45[(v51 + (v48 * 4))][(v49 + (v46 * 5))];
                v43[v51][v49].write(v57);
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
  float v58[180][210],
  float v59[210][190],
  float v60[190][220],
  float v61[180][220],
  float v62[180][220]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v58 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v58 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v59 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v59 cyclic dim=2 factor=5

  #pragma HLS array_partition variable=v60 cyclic dim=1 factor=5
  #pragma HLS array_partition variable=v60 cyclic dim=2 factor=11

  #pragma HLS array_partition variable=v61 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v61 cyclic dim=2 factor=11

  #pragma HLS array_partition variable=v62 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v62 cyclic dim=2 factor=11

  hls::stream<float> v63[4][11];
	#pragma HLS STREAM variable=v63 depth=900
  hls::stream<float> v64[4][11];
	#pragma HLS STREAM variable=v64 depth=900
  hls::stream<float> v65[4][5];
	#pragma HLS STREAM variable=v65 depth=1710
  node3(v59, v58, v65, 0.000000);
  node2(v60, v65, v64, 0.000000);
  node1(v64, v61, v63, 0.100000, 0.500000);
  node0(v63, v62);
  return ;
}

