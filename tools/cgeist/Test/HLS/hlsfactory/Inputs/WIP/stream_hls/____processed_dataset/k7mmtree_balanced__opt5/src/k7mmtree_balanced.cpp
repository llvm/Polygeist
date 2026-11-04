
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
  hls::stream<float> v0[4][4],
  float v1[64][64]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=4

  loop0: for (int v2 = 0; v2 < 16; v2++) {
    loop1: for (int v3 = 0; v3 < 16; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 4; v4++) {
        loop3: for (int v5 = 0; v5 < 4; v5++) {
          float v6 = v0[v5][v4].read();
          v1[(v5 + (v3 * 4))][(v4 + (v2 * 4))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[4][4],
  hls::stream<float> v8[4][4],
  hls::stream<float> v9[4][4],
  float v10
) {
  float v11[64][64];
  #pragma HLS array_partition variable=v11 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v11 cyclic dim=2 factor=4

  float v12[64][64];
  #pragma HLS array_partition variable=v12 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v12 cyclic dim=2 factor=4

  float v13[64][64];
  #pragma HLS array_partition variable=v13 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v13 cyclic dim=2 factor=4

  loop4: for (int v14 = 0; v14 < 16; v14++) {
    loop5: for (int v15 = 0; v15 < 16; v15++) {
      loop6: for (int v16 = 0; v16 < 16; v16++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop7: for (int v17 = 0; v17 < 4; v17++) {
          loop8: for (int v18 = 0; v18 < 4; v18++) {
            loop9: for (int v19 = 0; v19 < 4; v19++) {
              if ((v18 + (v15 * 4)) == 0) {
                float v20 = v7[v19][v17].read();
                v13[(v19 + (v16 * 4))][(v17 + (v14 * 4))] = v20;
              }
              float v21 = v13[(v19 + (v16 * 4))][(v17 + (v14 * 4))];
              if ((v19 + (v16 * 4)) == 0) {
                float v22 = v8[v17][v18].read();
                v12[(v17 + (v14 * 4))][(v18 + (v15 * 4))] = v22;
              }
              float v23 = v12[(v17 + (v14 * 4))][(v18 + (v15 * 4))];
              if ((v17 + (v14 * 4)) == 0) {
                v11[(v19 + (v16 * 4))][(v18 + (v15 * 4))] = v10;
              }
              float v24 = v11[(v19 + (v16 * 4))][(v18 + (v15 * 4))];
              float v25 = v21 * v23;
              float v26 = v24 + v25;
              v11[(v19 + (v16 * 4))][(v18 + (v15 * 4))] = v26;
              if (((v17 + (v14 * 4)) - 63) == 0) {
                float v27 = v11[(v19 + (v16 * 4))][(v18 + (v15 * 4))];
                v9[v19][v18].write(v27);
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
  hls::stream<float> v28[4][4],
  hls::stream<float> v29[4][4],
  hls::stream<float> v30[4][4],
  float v31
) {
  float v32[64][64];
  #pragma HLS array_partition variable=v32 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v32 cyclic dim=2 factor=4

  float v33[64][64];
  #pragma HLS array_partition variable=v33 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v33 cyclic dim=2 factor=4

  float v34[64][64];
  #pragma HLS array_partition variable=v34 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v34 cyclic dim=2 factor=4

  loop10: for (int v35 = 0; v35 < 16; v35++) {
    loop11: for (int v36 = 0; v36 < 16; v36++) {
      loop12: for (int v37 = 0; v37 < 16; v37++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop13: for (int v38 = 0; v38 < 4; v38++) {
          loop14: for (int v39 = 0; v39 < 4; v39++) {
            loop15: for (int v40 = 0; v40 < 4; v40++) {
              if ((v40 + (v37 * 4)) == 0) {
                float v41 = v29[v38][v39].read();
                v34[(v38 + (v35 * 4))][(v39 + (v36 * 4))] = v41;
              }
              float v42 = v34[(v38 + (v35 * 4))][(v39 + (v36 * 4))];
              if ((v38 + (v35 * 4)) == 0) {
                float v43 = v28[v39][v40].read();
                v33[(v39 + (v36 * 4))][(v40 + (v37 * 4))] = v43;
              }
              float v44 = v33[(v39 + (v36 * 4))][(v40 + (v37 * 4))];
              if ((v39 + (v36 * 4)) == 0) {
                v32[(v38 + (v35 * 4))][(v40 + (v37 * 4))] = v31;
              }
              float v45 = v32[(v38 + (v35 * 4))][(v40 + (v37 * 4))];
              float v46 = v42 * v44;
              float v47 = v45 + v46;
              v32[(v38 + (v35 * 4))][(v40 + (v37 * 4))] = v47;
              if (((v39 + (v36 * 4)) - 63) == 0) {
                float v48 = v32[(v38 + (v35 * 4))][(v40 + (v37 * 4))];
                v30[v38][v40].write(v48);
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
  hls::stream<float> v49[4][4],
  hls::stream<float> v50[4][4],
  hls::stream<float> v51[4][4],
  float v52
) {
  float v53[64][64];
  #pragma HLS array_partition variable=v53 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v53 cyclic dim=2 factor=4

  float v54[64][64];
  #pragma HLS array_partition variable=v54 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v54 cyclic dim=2 factor=4

  float v55[64][64];
  #pragma HLS array_partition variable=v55 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v55 cyclic dim=2 factor=4

  loop16: for (int v56 = 0; v56 < 16; v56++) {
    loop17: for (int v57 = 0; v57 < 16; v57++) {
      loop18: for (int v58 = 0; v58 < 16; v58++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop19: for (int v59 = 0; v59 < 4; v59++) {
          loop20: for (int v60 = 0; v60 < 4; v60++) {
            loop21: for (int v61 = 0; v61 < 4; v61++) {
              if ((v59 + (v56 * 4)) == 0) {
                float v62 = v49[v61][v60].read();
                v55[(v61 + (v58 * 4))][(v60 + (v57 * 4))] = v62;
              }
              float v63 = v55[(v61 + (v58 * 4))][(v60 + (v57 * 4))];
              if ((v61 + (v58 * 4)) == 0) {
                float v64 = v50[v60][v59].read();
                v54[(v60 + (v57 * 4))][(v59 + (v56 * 4))] = v64;
              }
              float v65 = v54[(v60 + (v57 * 4))][(v59 + (v56 * 4))];
              if ((v60 + (v57 * 4)) == 0) {
                v53[(v61 + (v58 * 4))][(v59 + (v56 * 4))] = v52;
              }
              float v66 = v53[(v61 + (v58 * 4))][(v59 + (v56 * 4))];
              float v67 = v63 * v65;
              float v68 = v66 + v67;
              v53[(v61 + (v58 * 4))][(v59 + (v56 * 4))] = v68;
              if (((v60 + (v57 * 4)) - 63) == 0) {
                float v69 = v53[(v61 + (v58 * 4))][(v59 + (v56 * 4))];
                v51[v61][v59].write(v69);
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
  float v70[64][64],
  float v71[64][64],
  hls::stream<float> v72[4][4],
  float v73
) {
  #pragma HLS array_partition variable=v70 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v70 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v71 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v71 cyclic dim=2 factor=4

  float v74[64][64];
  #pragma HLS array_partition variable=v74 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v74 cyclic dim=2 factor=4

  loop22: for (int v75 = 0; v75 < 16; v75++) {
    loop23: for (int v76 = 0; v76 < 16; v76++) {
      loop24: for (int v77 = 0; v77 < 16; v77++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v78 = 0; v78 < 4; v78++) {
          loop26: for (int v79 = 0; v79 < 4; v79++) {
            loop27: for (int v80 = 0; v80 < 4; v80++) {
              float v81 = v70[(v79 + (v76 * 4))][(v78 + (v75 * 4))];
              float v82 = v71[(v78 + (v75 * 4))][(v80 + (v77 * 4))];
              if ((v78 + (v75 * 4)) == 0) {
                v74[(v79 + (v76 * 4))][(v80 + (v77 * 4))] = v73;
              }
              float v83 = v74[(v79 + (v76 * 4))][(v80 + (v77 * 4))];
              float v84 = v81 * v82;
              float v85 = v83 + v84;
              v74[(v79 + (v76 * 4))][(v80 + (v77 * 4))] = v85;
              if (((v78 + (v75 * 4)) - 63) == 0) {
                float v86 = v74[(v79 + (v76 * 4))][(v80 + (v77 * 4))];
                v72[v79][v80].write(v86);
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
  float v87[64][64],
  float v88[64][64],
  hls::stream<float> v89[4][4],
  float v90
) {
  #pragma HLS array_partition variable=v87 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v87 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v88 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v88 cyclic dim=2 factor=4

  float v91[64][64];
  #pragma HLS array_partition variable=v91 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v91 cyclic dim=2 factor=4

  loop28: for (int v92 = 0; v92 < 16; v92++) {
    loop29: for (int v93 = 0; v93 < 16; v93++) {
      loop30: for (int v94 = 0; v94 < 16; v94++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop31: for (int v95 = 0; v95 < 4; v95++) {
          loop32: for (int v96 = 0; v96 < 4; v96++) {
            loop33: for (int v97 = 0; v97 < 4; v97++) {
              float v98 = v88[(v96 + (v93 * 4))][(v95 + (v92 * 4))];
              float v99 = v87[(v95 + (v92 * 4))][(v97 + (v94 * 4))];
              if ((v95 + (v92 * 4)) == 0) {
                v91[(v96 + (v93 * 4))][(v97 + (v94 * 4))] = v90;
              }
              float v100 = v91[(v96 + (v93 * 4))][(v97 + (v94 * 4))];
              float v101 = v98 * v99;
              float v102 = v100 + v101;
              v91[(v96 + (v93 * 4))][(v97 + (v94 * 4))] = v102;
              if (((v95 + (v92 * 4)) - 63) == 0) {
                float v103 = v91[(v96 + (v93 * 4))][(v97 + (v94 * 4))];
                v89[v96][v97].write(v103);
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
  float v104[64][64],
  float v105[64][64],
  hls::stream<float> v106[4][4],
  float v107
) {
  #pragma HLS array_partition variable=v104 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v104 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v105 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v105 cyclic dim=2 factor=4

  float v108[64][64];
  #pragma HLS array_partition variable=v108 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v108 cyclic dim=2 factor=4

  loop34: for (int v109 = 0; v109 < 16; v109++) {
    loop35: for (int v110 = 0; v110 < 16; v110++) {
      loop36: for (int v111 = 0; v111 < 16; v111++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop37: for (int v112 = 0; v112 < 4; v112++) {
          loop38: for (int v113 = 0; v113 < 4; v113++) {
            loop39: for (int v114 = 0; v114 < 4; v114++) {
              float v115 = v105[(v114 + (v111 * 4))][(v113 + (v110 * 4))];
              float v116 = v104[(v113 + (v110 * 4))][(v112 + (v109 * 4))];
              if ((v113 + (v110 * 4)) == 0) {
                v108[(v114 + (v111 * 4))][(v112 + (v109 * 4))] = v107;
              }
              float v117 = v108[(v114 + (v111 * 4))][(v112 + (v109 * 4))];
              float v118 = v115 * v116;
              float v119 = v117 + v118;
              v108[(v114 + (v111 * 4))][(v112 + (v109 * 4))] = v119;
              if (((v113 + (v110 * 4)) - 63) == 0) {
                float v120 = v108[(v114 + (v111 * 4))][(v112 + (v109 * 4))];
                v106[v114][v112].write(v120);
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
  float v121[64][64],
  float v122[64][64],
  hls::stream<float> v123[4][4],
  float v124
) {
  #pragma HLS array_partition variable=v121 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v121 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v122 cyclic dim=1 factor=4
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
            loop45: for (int v131 = 0; v131 < 4; v131++) {
              float v132 = v121[(v131 + (v128 * 4))][(v129 + (v126 * 4))];
              float v133 = v122[(v129 + (v126 * 4))][(v130 + (v127 * 4))];
              if ((v129 + (v126 * 4)) == 0) {
                v125[(v131 + (v128 * 4))][(v130 + (v127 * 4))] = v124;
              }
              float v134 = v125[(v131 + (v128 * 4))][(v130 + (v127 * 4))];
              float v135 = v132 * v133;
              float v136 = v134 + v135;
              v125[(v131 + (v128 * 4))][(v130 + (v127 * 4))] = v136;
              if (((v129 + (v126 * 4)) - 63) == 0) {
                float v137 = v125[(v131 + (v128 * 4))][(v130 + (v127 * 4))];
                v123[v131][v130].write(v137);
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
  float v138[64][64],
  float v139[64][64],
  float v140[64][64],
  float v141[64][64],
  float v142[64][64],
  float v143[64][64],
  float v144[64][64],
  float v145[64][64],
  float v146[64][64]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v138 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v138 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v139 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v139 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v140 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v140 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v141 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v141 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v142 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v142 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v143 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v143 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v144 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v144 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v145 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v145 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v146 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v146 cyclic dim=2 factor=4

  hls::stream<float> v147[4][4];
	#pragma HLS STREAM variable=v147 depth=256
  hls::stream<float> v148[4][4];
	#pragma HLS STREAM variable=v148 depth=256
  hls::stream<float> v149[4][4];
	#pragma HLS STREAM variable=v149 depth=256
  hls::stream<float> v150[4][4];
	#pragma HLS STREAM variable=v150 depth=256
  hls::stream<float> v151[4][4];
	#pragma HLS STREAM variable=v151 depth=256
  hls::stream<float> v152[4][4];
	#pragma HLS STREAM variable=v152 depth=256
  hls::stream<float> v153[4][4];
	#pragma HLS STREAM variable=v153 depth=256
  node7(v138, v139, v153, 0.000000);
  node6(v141, v140, v152, 0.000000);
  node5(v143, v142, v151, 0.000000);
  node4(v144, v145, v150, 0.000000);
  node3(v153, v152, v149, 0.000000);
  node2(v150, v151, v148, 0.000000);
  node1(v149, v148, v147, 0.000000);
  node0(v147, v146);
  return ;
}

