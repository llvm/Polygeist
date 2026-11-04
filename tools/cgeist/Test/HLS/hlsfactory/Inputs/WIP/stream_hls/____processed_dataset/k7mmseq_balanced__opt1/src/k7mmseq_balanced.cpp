
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
  float v1[64][64]
) {
  loop0: for (int v2 = 0; v2 < 64; v2++) {
    loop1: for (int v3 = 0; v3 < 64; v3++) {
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
  float v6[64][64],
  hls::stream<float> &v7,
  float v8
) {
  float v9[64][64];
  float v10[64][64];
  loop2: for (int v11 = 0; v11 < 64; v11++) {
    loop3: for (int v12 = 0; v12 < 64; v12++) {
      loop4: for (int v13 = 0; v13 < 64; v13++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v12 == 0) {
          float v14 = v5.read();
          v10[v11][v13] = v14;
        }
        float v15 = v10[v11][v13];
        float v16 = v6[v13][v12];
        if (v13 == 0) {
          v9[v11][v12] = v8;
        }
        float v17 = v9[v11][v12];
        float v18 = v15 * v16;
        float v19 = v17 + v18;
        v9[v11][v12] = v19;
        if ((v13 - 63) == 0) {
          float v20 = v9[v11][v12];
          v7.write(v20);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> &v21,
  float v22[64][64],
  hls::stream<float> &v23,
  float v24
) {
  float v25[64][64];
  float v26[64][64];
  loop5: for (int v27 = 0; v27 < 64; v27++) {
    loop6: for (int v28 = 0; v28 < 64; v28++) {
      loop7: for (int v29 = 0; v29 < 64; v29++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v28 == 0) {
          float v30 = v21.read();
          v26[v27][v29] = v30;
        }
        float v31 = v26[v27][v29];
        float v32 = v22[v29][v28];
        if (v29 == 0) {
          v25[v27][v28] = v24;
        }
        float v33 = v25[v27][v28];
        float v34 = v31 * v32;
        float v35 = v33 + v34;
        v25[v27][v28] = v35;
        if ((v29 - 63) == 0) {
          float v36 = v25[v27][v28];
          v23.write(v36);
        }
      }
    }
  }
  return ;
}

void node3(
  hls::stream<float> &v37,
  float v38[64][64],
  hls::stream<float> &v39,
  float v40
) {
  float v41[64][64];
  float v42[64][64];
  loop8: for (int v43 = 0; v43 < 64; v43++) {
    loop9: for (int v44 = 0; v44 < 64; v44++) {
      loop10: for (int v45 = 0; v45 < 64; v45++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v44 == 0) {
          float v46 = v37.read();
          v42[v43][v45] = v46;
        }
        float v47 = v42[v43][v45];
        float v48 = v38[v45][v44];
        if (v45 == 0) {
          v41[v43][v44] = v40;
        }
        float v49 = v41[v43][v44];
        float v50 = v47 * v48;
        float v51 = v49 + v50;
        v41[v43][v44] = v51;
        if ((v45 - 63) == 0) {
          float v52 = v41[v43][v44];
          v39.write(v52);
        }
      }
    }
  }
  return ;
}

void node4(
  hls::stream<float> &v53,
  float v54[64][64],
  hls::stream<float> &v55,
  float v56
) {
  float v57[64][64];
  float v58[64][64];
  loop11: for (int v59 = 0; v59 < 64; v59++) {
    loop12: for (int v60 = 0; v60 < 64; v60++) {
      loop13: for (int v61 = 0; v61 < 64; v61++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v60 == 0) {
          float v62 = v53.read();
          v58[v59][v61] = v62;
        }
        float v63 = v58[v59][v61];
        float v64 = v54[v61][v60];
        if (v61 == 0) {
          v57[v59][v60] = v56;
        }
        float v65 = v57[v59][v60];
        float v66 = v63 * v64;
        float v67 = v65 + v66;
        v57[v59][v60] = v67;
        if ((v61 - 63) == 0) {
          float v68 = v57[v59][v60];
          v55.write(v68);
        }
      }
    }
  }
  return ;
}

void node5(
  hls::stream<float> &v69,
  float v70[64][64],
  hls::stream<float> &v71,
  float v72
) {
  float v73[64][64];
  float v74[64][64];
  loop14: for (int v75 = 0; v75 < 64; v75++) {
    loop15: for (int v76 = 0; v76 < 64; v76++) {
      loop16: for (int v77 = 0; v77 < 64; v77++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v76 == 0) {
          float v78 = v69.read();
          v74[v75][v77] = v78;
        }
        float v79 = v74[v75][v77];
        float v80 = v70[v77][v76];
        if (v77 == 0) {
          v73[v75][v76] = v72;
        }
        float v81 = v73[v75][v76];
        float v82 = v79 * v80;
        float v83 = v81 + v82;
        v73[v75][v76] = v83;
        if ((v77 - 63) == 0) {
          float v84 = v73[v75][v76];
          v71.write(v84);
        }
      }
    }
  }
  return ;
}

void node6(
  hls::stream<float> &v85,
  float v86[64][64],
  hls::stream<float> &v87,
  float v88
) {
  float v89[64][64];
  float v90[64][64];
  loop17: for (int v91 = 0; v91 < 64; v91++) {
    loop18: for (int v92 = 0; v92 < 64; v92++) {
      loop19: for (int v93 = 0; v93 < 64; v93++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v92 == 0) {
          float v94 = v85.read();
          v90[v91][v93] = v94;
        }
        float v95 = v90[v91][v93];
        float v96 = v86[v93][v92];
        if (v93 == 0) {
          v89[v91][v92] = v88;
        }
        float v97 = v89[v91][v92];
        float v98 = v95 * v96;
        float v99 = v97 + v98;
        v89[v91][v92] = v99;
        if ((v93 - 63) == 0) {
          float v100 = v89[v91][v92];
          v87.write(v100);
        }
      }
    }
  }
  return ;
}

void node7(
  float v101[64][64],
  float v102[64][64],
  hls::stream<float> &v103,
  float v104
) {
  float v105[64][64];
  loop20: for (int v106 = 0; v106 < 64; v106++) {
    loop21: for (int v107 = 0; v107 < 64; v107++) {
      loop22: for (int v108 = 0; v108 < 64; v108++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v109 = v101[v106][v108];
        float v110 = v102[v108][v107];
        if (v108 == 0) {
          v105[v106][v107] = v104;
        }
        float v111 = v105[v106][v107];
        float v112 = v109 * v110;
        float v113 = v111 + v112;
        v105[v106][v107] = v113;
        if ((v108 - 63) == 0) {
          float v114 = v105[v106][v107];
          v103.write(v114);
        }
      }
    }
  }
  return ;
}

void forward(
  float v115[64][64],
  float v116[64][64],
  float v117[64][64],
  float v118[64][64],
  float v119[64][64],
  float v120[64][64],
  float v121[64][64],
  float v122[64][64],
  float v123[64][64]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v124("fifo_0");
	#pragma HLS STREAM variable=v124 depth=4096
  hls::stream<float> v125("fifo_1");
	#pragma HLS STREAM variable=v125 depth=4096
  hls::stream<float> v126("fifo_2");
	#pragma HLS STREAM variable=v126 depth=4096
  hls::stream<float> v127("fifo_3");
	#pragma HLS STREAM variable=v127 depth=4096
  hls::stream<float> v128("fifo_4");
	#pragma HLS STREAM variable=v128 depth=4096
  hls::stream<float> v129("fifo_5");
	#pragma HLS STREAM variable=v129 depth=4096
  hls::stream<float> v130("fifo_6");
	#pragma HLS STREAM variable=v130 depth=4096
  node7(v115, v116, v130, 0.000000);
  node6(v130, v117, v129, 0.000000);
  node5(v129, v118, v128, 0.000000);
  node4(v128, v119, v127, 0.000000);
  node3(v127, v120, v126, 0.000000);
  node2(v126, v121, v125, 0.000000);
  node1(v125, v122, v124, 0.000000);
  node0(v124, v123);
  return ;
}

