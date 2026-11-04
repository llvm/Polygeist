
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
  float v1[1][512][128]
) {
  loop0: for (int v2 = 0; v2 < 512; v2++) {
    loop1: for (int v3 = 0; v3 < 128; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v4 = v0.read();
      v1[0][v2][v3] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> &v5,
  float v6[128],
  hls::stream<float> &v7
) {
  loop2: for (int v8 = 0; v8 < 512; v8++) {
    loop3: for (int v9 = 0; v9 < 128; v9++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v10 = v5.read();
      float v11 = v6[v9];
      float v12 = v10 + v11;
      v7.write(v12);
    }
  }
  return ;
}

void node2(
  hls::stream<float> &v13,
  float v14[256][128],
  hls::stream<float> &v15,
  float v16
) {
  float v17[512][128];
  float v18[256][128];
  float v19[512][256];
  loop4: for (int v20 = 0; v20 < 512; v20++) {
    loop5: for (int v21 = 0; v21 < 128; v21++) {
      loop6: for (int v22 = 0; v22 < 256; v22++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v21 == 0) {
          float v23 = v13.read();
          v19[v20][v22] = v23;
        }
        float v24 = v19[v20][v22];
        if (v20 == 0) {
          float v25 = v14[v22][v21];
          v18[v22][v21] = v25;
        }
        float v26 = v18[v22][v21];
        if (v22 == 0) {
          v17[v20][v21] = v16;
        }
        float v27 = v17[v20][v21];
        float v28 = v24 * v26;
        float v29 = v27 + v28;
        v17[v20][v21] = v29;
        if ((v22 - 255) == 0) {
          float v30 = v17[v20][v21];
          v15.write(v30);
        }
      }
    }
  }
  return ;
}

void node3(
  float v31[128][256],
  float v32[256][128]
) {
  loop7: for (int v33 = 0; v33 < 256; v33++) {
    loop8: for (int v34 = 0; v34 < 128; v34++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v35 = v31[v34][v33];
      v32[v33][v34] = v35;
    }
  }
  return ;
}

void node4(
  hls::stream<float> &v36,
  float v37[256],
  hls::stream<float> &v38,
  float v39
) {
  loop9: for (int v40 = 0; v40 < 512; v40++) {
    loop10: for (int v41 = 0; v41 < 256; v41++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v42 = v36.read();
      float v43 = v37[v41];
      float v44 = v42 + v43;
      bool v45 = v44 > v39;
      float v46 = v45 ? (float)v44 : (float)v39;
      v38.write(v46);
    }
  }
  return ;
}

void node5(
  hls::stream<float> &v47,
  float v48[128][256],
  hls::stream<float> &v49,
  float v50
) {
  float v51[512][256];
  float v52[128][256];
  float v53[512][128];
  loop11: for (int v54 = 0; v54 < 512; v54++) {
    loop12: for (int v55 = 0; v55 < 256; v55++) {
      loop13: for (int v56 = 0; v56 < 128; v56++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v55 == 0) {
          float v57 = v47.read();
          v53[v54][v56] = v57;
        }
        float v58 = v53[v54][v56];
        if (v54 == 0) {
          float v59 = v48[v56][v55];
          v52[v56][v55] = v59;
        }
        float v60 = v52[v56][v55];
        if (v56 == 0) {
          v51[v54][v55] = v50;
        }
        float v61 = v51[v54][v55];
        float v62 = v58 * v60;
        float v63 = v61 + v62;
        v51[v54][v55] = v63;
        if ((v56 - 127) == 0) {
          float v64 = v51[v54][v55];
          v49.write(v64);
        }
      }
    }
  }
  return ;
}

void node6(
  float v65[256][128],
  float v66[128][256]
) {
  loop14: for (int v67 = 0; v67 < 128; v67++) {
    loop15: for (int v68 = 0; v68 < 256; v68++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v69 = v65[v68][v67];
      v66[v67][v68] = v69;
    }
  }
  return ;
}

void node7(
  float v70[1][512][128],
  hls::stream<float> &v71
) {
  loop16: for (int v72 = 0; v72 < 512; v72++) {
    loop17: for (int v73 = 0; v73 < 128; v73++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v74 = v70[0][v72][v73];
      v71.write(v74);
    }
  }
  return ;
}

void forward(
  float v75[1][512][128],
  float v76[256],
  float v77[256][128],
  float v78[128],
  float v79[128][256],
  float v80[1][512][128]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v81("fifo_0");
	#pragma HLS STREAM variable=v81 depth=65536
  hls::stream<float> v82("fifo_1");
	#pragma HLS STREAM variable=v82 depth=65536
  float v83[256][128];
  hls::stream<float> v84("fifo_2");
	#pragma HLS STREAM variable=v84 depth=131072
  hls::stream<float> v85("fifo_3");
	#pragma HLS STREAM variable=v85 depth=131072
  float v86[128][256];
  hls::stream<float> v87("fifo_4");
	#pragma HLS STREAM variable=v87 depth=65536
  node7(v75, v87);
  node6(v77, v86);
  node5(v87, v86, v85, 0.000000);
  node4(v85, v76, v84, 0.000000);
  node3(v79, v83);
  node2(v84, v83, v82, 0.000000);
  node1(v82, v78, v81);
  node0(v81, v80);
  return ;
}

