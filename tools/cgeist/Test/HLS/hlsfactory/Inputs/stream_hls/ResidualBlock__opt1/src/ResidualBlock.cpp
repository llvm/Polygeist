
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

void forward(
  float v0[1][16][56][56],
  float v1[16][16][3][3],
  float v2[16][16][3][3],
  float v3[1][16][56][56]
) {
	#pragma HLS DATAFLOW
  float v4[16][58][58];
  loop0: for (int v5 = 0; v5 < 16; v5++) {
    loop1: for (int v6 = 0; v6 < 58; v6++) {
      loop2: for (int v7 = 0; v7 < 58; v7++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        v4[v5][v6][v7] = 0.000000;
      }
    }
  }
  loop3: for (int v8 = 0; v8 < 16; v8++) {
    loop4: for (int v9 = 0; v9 < 56; v9++) {
      loop5: for (int v10 = 0; v10 < 56; v10++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v11 = v0[0][v8][v9][v10];
        v4[v8][(v9 + 1)][(v10 + 1)] = v11;
      }
    }
  }
  float v12[16][56][56];
  hls::stream<float> v13("fifo_0");
	#pragma HLS STREAM variable=v13 depth=50176
  loop6: for (int v14 = 0; v14 < 16; v14++) {
    loop7: for (int v15 = 0; v15 < 56; v15++) {
      loop8: for (int v16 = 0; v16 < 56; v16++) {
        loop9: for (int v17 = 0; v17 < 16; v17++) {
          loop10: for (int v18 = 0; v18 < 3; v18++) {
            loop11: for (int v19 = 0; v19 < 3; v19++) {
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              float v20 = v4[v17][(v15 + v18)][(v16 + v19)];
              float v21 = v1[v14][v17][v18][v19];
              if (v17 == 0 && v18 == 0 && v19 == 0) {
                v12[v14][v15][v16] = 0.000000;
              }
              float v22 = v12[v14][v15][v16];
              float v23 = v20 * v21;
              float v24 = v22 + v23;
              v12[v14][v15][v16] = v24;
              if ((v17 - 15) == 0 && (v18 - 2) == 0 && (v19 - 2) == 0) {
                float v25 = v12[v14][v15][v16];
                v13.write(v25);
              }
            }
          }
        }
      }
    }
  }
  hls::stream<float> v26("fifo_1");
	#pragma HLS STREAM variable=v26 depth=50176
  loop12: for (int v27 = 0; v27 < 16; v27++) {
    loop13: for (int v28 = 0; v28 < 56; v28++) {
      loop14: for (int v29 = 0; v29 < 56; v29++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v30 = v13.read();
        float v31 = 0.000010;
        float v32 = v31 + 1.000000;
        float v33 = 1.0 / sqrt(v32);
        float v34 = v30 * v33;
        float v35 = v34 + 0.000000;
        bool v36 = v35 > 0.000000;
        float v37 = v36 ? (float)v35 : (float)0.000000;
        v26.write(v37);
      }
    }
  }
  float v38[16][58][58];
  loop15: for (int v39 = 0; v39 < 16; v39++) {
    loop16: for (int v40 = 0; v40 < 58; v40++) {
      loop17: for (int v41 = 0; v41 < 58; v41++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        v38[v39][v40][v41] = 0.000000;
      }
    }
  }
  loop18: for (int v42 = 0; v42 < 16; v42++) {
    loop19: for (int v43 = 0; v43 < 56; v43++) {
      loop20: for (int v44 = 0; v44 < 56; v44++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v45 = v26.read();
        v38[v42][(v43 + 1)][(v44 + 1)] = v45;
      }
    }
  }
  float v46[16][56][56];
  hls::stream<float> v47("fifo_2");
	#pragma HLS STREAM variable=v47 depth=50176
  loop21: for (int v48 = 0; v48 < 16; v48++) {
    loop22: for (int v49 = 0; v49 < 56; v49++) {
      loop23: for (int v50 = 0; v50 < 56; v50++) {
        loop24: for (int v51 = 0; v51 < 16; v51++) {
          loop25: for (int v52 = 0; v52 < 3; v52++) {
            loop26: for (int v53 = 0; v53 < 3; v53++) {
              #pragma HLS pipeline II=1
              #pragma HLS loop_flatten
              float v54 = v38[v51][(v49 + v52)][(v50 + v53)];
              float v55 = v2[v48][v51][v52][v53];
              if (v51 == 0 && v52 == 0 && v53 == 0) {
                v46[v48][v49][v50] = 0.000000;
              }
              float v56 = v46[v48][v49][v50];
              float v57 = v54 * v55;
              float v58 = v56 + v57;
              v46[v48][v49][v50] = v58;
              if ((v51 - 15) == 0 && (v52 - 2) == 0 && (v53 - 2) == 0) {
                float v59 = v46[v48][v49][v50];
                v47.write(v59);
              }
            }
          }
        }
      }
    }
  }
  hls::stream<float> v60("fifo_3");
	#pragma HLS STREAM variable=v60 depth=50176
  loop27: for (int v61 = 0; v61 < 16; v61++) {
    loop28: for (int v62 = 0; v62 < 56; v62++) {
      loop29: for (int v63 = 0; v63 < 56; v63++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v64 = v47.read();
        float v65 = v0[0][v61][v62][v63];
        float v66 = 0.000010;
        float v67 = v66 + 1.000000;
        float v68 = 1.0 / sqrt(v67);
        float v69 = v64 * v68;
        float v70 = v69 + 0.000000;
        float v71 = v70 + v65;
        bool v72 = v71 > 0.000000;
        float v73 = v72 ? (float)v71 : (float)0.000000;
        v60.write(v73);
      }
    }
  }
  loop30: for (int v74 = 0; v74 < 16; v74++) {
    loop31: for (int v75 = 0; v75 < 56; v75++) {
      loop32: for (int v76 = 0; v76 < 56; v76++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v77 = v60.read();
        v3[0][v74][v75][v76] = v77;
      }
    }
  }
  return ;
}

