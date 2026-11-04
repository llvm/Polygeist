
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
  hls::stream<float> v0[4][1],
  float v1[64][16]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=4

  loop0: for (int v2 = 0; v2 < 16; v2++) {
    loop1: for (int v3 = 0; v3 < 16; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 4; v4++) {
        loop3: for (int v5 = 0; v5 < 1; v5++) {
          float v6 = v0[v4][v5].read();
          v1[(v4 + (v2 * 4))][(v3 + v5)] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[4][2],
  float v8[32][16],
  hls::stream<float> v9[4][1],
  float v10
) {
  #pragma HLS array_partition variable=v8 cyclic dim=1 factor=2

  float v11[64][16];
  #pragma HLS array_partition variable=v11 cyclic dim=1 factor=4

  float v12[64][32];
  #pragma HLS array_partition variable=v12 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v12 cyclic dim=2 factor=2

  loop4: for (int v13 = 0; v13 < 16; v13++) {
    loop5: for (int v14 = 0; v14 < 16; v14++) {
      loop6: for (int v15 = 0; v15 < 16; v15++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop7: for (int v16 = 0; v16 < 4; v16++) {
          loop8: for (int v17 = 0; v17 < 2; v17++) {
            loop9: for (int v18 = 0; v18 < 1; v18++) {
              if ((v15 + v18) == 0) {
                float v19 = v7[v16][v17].read();
                v12[(v16 + (v13 * 4))][(v17 + (v14 * 2))] = v19;
              }
              float v20 = v12[(v16 + (v13 * 4))][(v17 + (v14 * 2))];
              float v21 = v8[(v17 + (v14 * 2))][(v15 + v18)];
              if ((v17 + (v14 * 2)) == 0) {
                v11[(v16 + (v13 * 4))][(v15 + v18)] = v10;
              }
              float v22 = v11[(v16 + (v13 * 4))][(v15 + v18)];
              float v23 = v20 * v21;
              float v24 = v22 + v23;
              v11[(v16 + (v13 * 4))][(v15 + v18)] = v24;
              if (((v17 + (v14 * 2)) - 31) == 0) {
                float v25 = v11[(v16 + (v13 * 4))][(v15 + v18)];
                v9[v16][v18].write(v25);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v26[4][4],
  float v27[64][32],
  hls::stream<float> v28[4][2],
  float v29
) {
  #pragma HLS array_partition variable=v27 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v27 cyclic dim=2 factor=2

  float v30[64][32];
  #pragma HLS array_partition variable=v30 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v30 cyclic dim=2 factor=2

  float v31[64][64];
  #pragma HLS array_partition variable=v31 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v31 cyclic dim=2 factor=4

  loop10: for (int v32 = 0; v32 < 16; v32++) {
    loop11: for (int v33 = 0; v33 < 16; v33++) {
      loop12: for (int v34 = 0; v34 < 16; v34++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop13: for (int v35 = 0; v35 < 4; v35++) {
          loop14: for (int v36 = 0; v36 < 4; v36++) {
            loop15: for (int v37 = 0; v37 < 2; v37++) {
              if ((v37 + (v34 * 2)) == 0) {
                float v38 = v26[v35][v36].read();
                v31[(v35 + (v32 * 4))][(v36 + (v33 * 4))] = v38;
              }
              float v39 = v31[(v35 + (v32 * 4))][(v36 + (v33 * 4))];
              float v40 = v27[(v36 + (v33 * 4))][(v37 + (v34 * 2))];
              if ((v36 + (v33 * 4)) == 0) {
                v30[(v35 + (v32 * 4))][(v37 + (v34 * 2))] = v29;
              }
              float v41 = v30[(v35 + (v32 * 4))][(v37 + (v34 * 2))];
              float v42 = v39 * v40;
              float v43 = v41 + v42;
              v30[(v35 + (v32 * 4))][(v37 + (v34 * 2))] = v43;
              if (((v36 + (v33 * 4)) - 63) == 0) {
                float v44 = v30[(v35 + (v32 * 4))][(v37 + (v34 * 2))];
                v28[v35][v37].write(v44);
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
  float v45[128][64],
  hls::stream<float> v46[4][4],
  hls::stream<float> v47[4][4],
  float v48
) {
  #pragma HLS array_partition variable=v45 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v45 cyclic dim=2 factor=4

  float v49[64][64];
  #pragma HLS array_partition variable=v49 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v49 cyclic dim=2 factor=4

  float v50[64][128];
  #pragma HLS array_partition variable=v50 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v50 cyclic dim=2 factor=4

  loop16: for (int v51 = 0; v51 < 16; v51++) {
    loop17: for (int v52 = 0; v52 < 32; v52++) {
      loop18: for (int v53 = 0; v53 < 16; v53++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop19: for (int v54 = 0; v54 < 4; v54++) {
          loop20: for (int v55 = 0; v55 < 4; v55++) {
            loop21: for (int v56 = 0; v56 < 4; v56++) {
              if ((v56 + (v53 * 4)) == 0) {
                float v57 = v46[v54][v55].read();
                v50[(v54 + (v51 * 4))][(v55 + (v52 * 4))] = v57;
              }
              float v58 = v50[(v54 + (v51 * 4))][(v55 + (v52 * 4))];
              float v59 = v45[(v55 + (v52 * 4))][(v56 + (v53 * 4))];
              if ((v55 + (v52 * 4)) == 0) {
                v49[(v54 + (v51 * 4))][(v56 + (v53 * 4))] = v48;
              }
              float v60 = v49[(v54 + (v51 * 4))][(v56 + (v53 * 4))];
              float v61 = v58 * v59;
              float v62 = v60 + v61;
              v49[(v54 + (v51 * 4))][(v56 + (v53 * 4))] = v62;
              if (((v55 + (v52 * 4)) - 127) == 0) {
                float v63 = v49[(v54 + (v51 * 4))][(v56 + (v53 * 4))];
                v47[v54][v56].write(v63);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node4(
  hls::stream<float> v64[4][8],
  float v65[128][128],
  hls::stream<float> v66[4][4],
  float v67
) {
  #pragma HLS array_partition variable=v65 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v65 cyclic dim=2 factor=4

  float v68[64][128];
  #pragma HLS array_partition variable=v68 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v68 cyclic dim=2 factor=4

  float v69[64][128];
  #pragma HLS array_partition variable=v69 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v69 cyclic dim=2 factor=8

  loop22: for (int v70 = 0; v70 < 16; v70++) {
    loop23: for (int v71 = 0; v71 < 16; v71++) {
      loop24: for (int v72 = 0; v72 < 32; v72++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v73 = 0; v73 < 4; v73++) {
          loop26: for (int v74 = 0; v74 < 8; v74++) {
            loop27: for (int v75 = 0; v75 < 4; v75++) {
              if ((v75 + (v72 * 4)) == 0) {
                float v76 = v64[v73][v74].read();
                v69[(v73 + (v70 * 4))][(v74 + (v71 * 8))] = v76;
              }
              float v77 = v69[(v73 + (v70 * 4))][(v74 + (v71 * 8))];
              float v78 = v65[(v74 + (v71 * 8))][(v75 + (v72 * 4))];
              if ((v74 + (v71 * 8)) == 0) {
                v68[(v73 + (v70 * 4))][(v75 + (v72 * 4))] = v67;
              }
              float v79 = v68[(v73 + (v70 * 4))][(v75 + (v72 * 4))];
              float v80 = v77 * v78;
              float v81 = v79 + v80;
              v68[(v73 + (v70 * 4))][(v75 + (v72 * 4))] = v81;
              if (((v74 + (v71 * 8)) - 127) == 0) {
                float v82 = v68[(v73 + (v70 * 4))][(v75 + (v72 * 4))];
                v66[v73][v75].write(v82);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node5(
  float v83[64][128],
  hls::stream<float> v84[4][4],
  hls::stream<float> v85[4][8],
  float v86
) {
  #pragma HLS array_partition variable=v83 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v83 cyclic dim=2 factor=8

  float v87[64][128];
  #pragma HLS array_partition variable=v87 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v87 cyclic dim=2 factor=8

  float v88[64][64];
  #pragma HLS array_partition variable=v88 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v88 cyclic dim=2 factor=4

  loop28: for (int v89 = 0; v89 < 16; v89++) {
    loop29: for (int v90 = 0; v90 < 16; v90++) {
      loop30: for (int v91 = 0; v91 < 16; v91++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop31: for (int v92 = 0; v92 < 4; v92++) {
          loop32: for (int v93 = 0; v93 < 4; v93++) {
            loop33: for (int v94 = 0; v94 < 8; v94++) {
              if ((v94 + (v91 * 8)) == 0) {
                float v95 = v84[v92][v93].read();
                v88[(v92 + (v89 * 4))][(v93 + (v90 * 4))] = v95;
              }
              float v96 = v88[(v92 + (v89 * 4))][(v93 + (v90 * 4))];
              float v97 = v83[(v93 + (v90 * 4))][(v94 + (v91 * 8))];
              if ((v93 + (v90 * 4)) == 0) {
                v87[(v92 + (v89 * 4))][(v94 + (v91 * 8))] = v86;
              }
              float v98 = v87[(v92 + (v89 * 4))][(v94 + (v91 * 8))];
              float v99 = v96 * v97;
              float v100 = v98 + v99;
              v87[(v92 + (v89 * 4))][(v94 + (v91 * 8))] = v100;
              if (((v93 + (v90 * 4)) - 63) == 0) {
                float v101 = v87[(v92 + (v89 * 4))][(v94 + (v91 * 8))];
                v85[v92][v94].write(v101);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node6(
  float v102[64][64],
  hls::stream<float> v103[4][4],
  hls::stream<float> v104[4][4],
  float v105
) {
  #pragma HLS array_partition variable=v102 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v102 cyclic dim=2 factor=4

  float v106[64][64];
  #pragma HLS array_partition variable=v106 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v106 cyclic dim=2 factor=4

  float v107[64][64];
  #pragma HLS array_partition variable=v107 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v107 cyclic dim=2 factor=4

  loop34: for (int v108 = 0; v108 < 16; v108++) {
    loop35: for (int v109 = 0; v109 < 16; v109++) {
      loop36: for (int v110 = 0; v110 < 16; v110++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop37: for (int v111 = 0; v111 < 4; v111++) {
          loop38: for (int v112 = 0; v112 < 4; v112++) {
            loop39: for (int v113 = 0; v113 < 4; v113++) {
              if ((v112 + (v109 * 4)) == 0) {
                float v114 = v103[v111][v113].read();
                v107[(v111 + (v108 * 4))][(v113 + (v110 * 4))] = v114;
              }
              float v115 = v107[(v111 + (v108 * 4))][(v113 + (v110 * 4))];
              float v116 = v102[(v113 + (v110 * 4))][(v112 + (v109 * 4))];
              if ((v113 + (v110 * 4)) == 0) {
                v106[(v111 + (v108 * 4))][(v112 + (v109 * 4))] = v105;
              }
              float v117 = v106[(v111 + (v108 * 4))][(v112 + (v109 * 4))];
              float v118 = v115 * v116;
              float v119 = v117 + v118;
              v106[(v111 + (v108 * 4))][(v112 + (v109 * 4))] = v119;
              if (((v113 + (v110 * 4)) - 63) == 0) {
                float v120 = v106[(v111 + (v108 * 4))][(v112 + (v109 * 4))];
                v104[v111][v112].write(v120);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node7(
  float v121[64][32],
  float v122[32][64],
  hls::stream<float> v123[4][4],
  float v124
) {
  #pragma HLS array_partition variable=v121 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v121 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v122 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v122 cyclic dim=2 factor=4

  float v125[64][64];
  #pragma HLS array_partition variable=v125 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v125 cyclic dim=2 factor=4

  loop40: for (int v126 = 0; v126 < 16; v126++) {
    loop41: for (int v127 = 0; v127 < 16; v127++) {
      loop42: for (int v128 = 0; v128 < 16; v128++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop43: for (int v129 = 0; v129 < 4; v129++) {
          loop44: for (int v130 = 0; v130 < 4; v130++) {
            loop45: for (int v131 = 0; v131 < 2; v131++) {
              float v132 = v121[(v129 + (v126 * 4))][(v131 + (v128 * 2))];
              float v133 = v122[(v131 + (v128 * 2))][(v130 + (v127 * 4))];
              if ((v131 + (v128 * 2)) == 0) {
                v125[(v129 + (v126 * 4))][(v130 + (v127 * 4))] = v124;
              }
              float v134 = v125[(v129 + (v126 * 4))][(v130 + (v127 * 4))];
              float v135 = v132 * v133;
              float v136 = v134 + v135;
              v125[(v129 + (v126 * 4))][(v130 + (v127 * 4))] = v136;
              if (((v131 + (v128 * 2)) - 31) == 0) {
                float v137 = v125[(v129 + (v126 * 4))][(v130 + (v127 * 4))];
                v123[v129][v130].write(v137);
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
  float v138[64][32],
  float v139[32][64],
  float v140[64][64],
  float v141[64][128],
  float v142[128][128],
  float v143[128][64],
  float v144[64][32],
  float v145[32][16],
  float v146[64][16]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v138 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v138 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v139 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v139 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v140 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v140 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v141 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v141 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v142 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v142 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v143 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v143 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v144 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v144 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v145 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v146 cyclic dim=1 factor=4

  hls::stream<float> v147[4][1];
	#pragma HLS STREAM variable=v147 depth=256
  hls::stream<float> v148[4][2];
	#pragma HLS STREAM variable=v148 depth=256
  hls::stream<float> v149[4][4];
	#pragma HLS STREAM variable=v149 depth=256
  hls::stream<float> v150[4][4];
	#pragma HLS STREAM variable=v150 depth=512
  hls::stream<float> v151[4][8];
	#pragma HLS STREAM variable=v151 depth=256
  hls::stream<float> v152[4][4];
	#pragma HLS STREAM variable=v152 depth=256
  hls::stream<float> v153[4][4];
	#pragma HLS STREAM variable=v153 depth=256
  node7(v138, v139, v153, 0.000000);
  node6(v140, v153, v152, 0.000000);
  node5(v141, v152, v151, 0.000000);
  node4(v151, v142, v150, 0.000000);
  node3(v143, v150, v149, 0.000000);
  node2(v149, v144, v148, 0.000000);
  node1(v148, v145, v147, 0.000000);
  node0(v147, v146);
  return ;
}

