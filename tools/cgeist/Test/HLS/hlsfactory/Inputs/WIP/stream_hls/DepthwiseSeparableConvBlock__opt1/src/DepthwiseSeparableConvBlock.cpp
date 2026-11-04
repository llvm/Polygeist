
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
  float v0[1][8][112][112],
  float v1[8][8][1][1],
  float v2[8][3][3],
  float v3[1][8][112][112]
) {
	#pragma HLS DATAFLOW
  float v4[8][114][114];
  loop0: for (int v5 = 0; v5 < 8; v5++) {
    loop1: for (int v6 = 0; v6 < 114; v6++) {
      loop2: for (int v7 = 0; v7 < 114; v7++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        v4[v5][v6][v7] = 0.000000;
      }
    }
  }
  loop3: for (int v8 = 0; v8 < 8; v8++) {
    loop4: for (int v9 = 0; v9 < 112; v9++) {
      loop5: for (int v10 = 0; v10 < 112; v10++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v11 = v0[0][v8][v9][v10];
        v4[v8][(v9 + 1)][(v10 + 1)] = v11;
      }
    }
  }
  float v12[8][112][112];
  hls::stream<float> v13("fifo_0");
	#pragma HLS STREAM variable=v13 depth=100352
  float v14[8][112][112];
  loop6: for (int v15 = 0; v15 < 112; v15++) {
    loop7: for (int v16 = 0; v16 < 112; v16++) {
      loop8: for (int v17 = 0; v17 < 8; v17++) {
        loop9: for (int v18 = 0; v18 < 3; v18++) {
          loop10: for (int v19 = 0; v19 < 3; v19++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_flatten
            float v20 = v4[v17][(v15 + v18)][(v16 + v19)];
            float v21 = v2[v17][v18][v19];
            if (v18 == 0 && v19 == 0) {
              v12[v17][v15][v16] = 0.000000;
            }
            float v22 = v12[v17][v15][v16];
            float v23 = v20 * v21;
            float v24 = v22 + v23;
            v12[v17][v15][v16] = v24;
            if ((v18 - 2) == 0 && (v19 - 2) == 0) {
              float v25 = v12[v17][v15][v16];
              v13.write(v25);
            }
          }
        }
      }
    }
  }
  float v26[8][112][112];
  hls::stream<float> v27("fifo_1");
	#pragma HLS STREAM variable=v27 depth=100352
  loop11: for (int v28 = 0; v28 < 8; v28++) {
    loop12: for (int v29 = 0; v29 < 112; v29++) {
      loop13: for (int v30 = 0; v30 < 112; v30++) {
        loop14: for (int v31 = 0; v31 < 8; v31++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          if (v28 == 0) {
            float v32 = v13.read();
            v14[v31][v29][v30] = v32;
          }
          float v33 = v14[v31][v29][v30];
          float v34 = v1[v28][v31][0][0];
          if (v31 == 0) {
            v26[v28][v29][v30] = 0.000000;
          }
          float v35 = v26[v28][v29][v30];
          float v36 = v33 * v34;
          float v37 = v35 + v36;
          v26[v28][v29][v30] = v37;
          if ((v31 - 7) == 0) {
            float v38 = v26[v28][v29][v30];
            v27.write(v38);
          }
        }
      }
    }
  }
  hls::stream<float> v39("fifo_2");
	#pragma HLS STREAM variable=v39 depth=100352
  loop15: for (int v40 = 0; v40 < 8; v40++) {
    loop16: for (int v41 = 0; v41 < 112; v41++) {
      loop17: for (int v42 = 0; v42 < 112; v42++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v43 = v27.read();
        float v44 = 0.000010;
        float v45 = v44 + 1.000000;
        float v46 = 1.0 / sqrt(v45);
        float v47 = v43 * v46;
        float v48 = v47 + 0.000000;
        bool v49 = v48 > 0.000000;
        float v50 = v49 ? (float)v48 : (float)0.000000;
        v39.write(v50);
      }
    }
  }
  loop18: for (int v51 = 0; v51 < 8; v51++) {
    loop19: for (int v52 = 0; v52 < 112; v52++) {
      loop20: for (int v53 = 0; v53 < 112; v53++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v54 = v39.read();
        v3[0][v51][v52][v53] = v54;
      }
    }
  }
  return ;
}

