
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
          float v6 = v0[v4][v5].read();
          v1[(v4 + (v2 * 4))][(v5 + (v3 * 4))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[4][4],
  hls::stream<float> v8[4][4],
  float v9
) {
  loop4: for (int v10 = 0; v10 < 16; v10++) {
    loop5: for (int v11 = 0; v11 < 16; v11++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v12 = 0; v12 < 4; v12++) {
        loop7: for (int v13 = 0; v13 < 4; v13++) {
          float v14 = v7[v12][v13].read();
          float v15 = v14 / v9;
          v8[v12][v13].write(v15);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v16[4][4],
  hls::stream<float> v17[4][4],
  hls::stream<float> v18[4][4],
  float v19
) {
  float v20[64][64];
  #pragma HLS array_partition variable=v20 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v20 cyclic dim=2 factor=4

  float v21[64][64];
  #pragma HLS array_partition variable=v21 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v21 cyclic dim=2 factor=4

  float v22[64][64];
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=4

  loop8: for (int v23 = 0; v23 < 16; v23++) {
    loop9: for (int v24 = 0; v24 < 16; v24++) {
      loop10: for (int v25 = 0; v25 < 16; v25++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v26 = 0; v26 < 4; v26++) {
          loop12: for (int v27 = 0; v27 < 4; v27++) {
            loop13: for (int v28 = 0; v28 < 4; v28++) {
              if ((v28 + (v25 * 4)) == 0) {
                float v29 = v16[v27][v26].read();
                v22[(v27 + (v24 * 4))][(v26 + (v23 * 4))] = v29;
              }
              float v30 = v22[(v27 + (v24 * 4))][(v26 + (v23 * 4))];
              if ((v27 + (v24 * 4)) == 0) {
                float v31 = v17[v26][v28].read();
                v21[(v26 + (v23 * 4))][(v28 + (v25 * 4))] = v31;
              }
              float v32 = v21[(v26 + (v23 * 4))][(v28 + (v25 * 4))];
              if ((v26 + (v23 * 4)) == 0) {
                v20[(v27 + (v24 * 4))][(v28 + (v25 * 4))] = v19;
              }
              float v33 = v20[(v27 + (v24 * 4))][(v28 + (v25 * 4))];
              float v34 = v30 * v32;
              float v35 = v33 + v34;
              v20[(v27 + (v24 * 4))][(v28 + (v25 * 4))] = v35;
              if (((v26 + (v23 * 4)) - 63) == 0) {
                float v36 = v20[(v27 + (v24 * 4))][(v28 + (v25 * 4))];
                v18[v27][v28].write(v36);
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
  hls::stream<float> v37[2][4],
  hls::stream<float> v38[4][2],
  hls::stream<float> v39[4][4],
  float v40
) {
  float v41[64][64];
  #pragma HLS array_partition variable=v41 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v41 cyclic dim=2 factor=4

  float v42[64][64];
  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v42 cyclic dim=2 factor=4

  float v43[64][64];
  #pragma HLS array_partition variable=v43 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v43 cyclic dim=2 factor=2

  loop14: for (int v44 = 0; v44 < 32; v44++) {
    loop15: for (int v45 = 0; v45 < 16; v45++) {
      loop16: for (int v46 = 0; v46 < 16; v46++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop17: for (int v47 = 0; v47 < 2; v47++) {
          loop18: for (int v48 = 0; v48 < 4; v48++) {
            loop19: for (int v49 = 0; v49 < 4; v49++) {
              if ((v49 + (v46 * 4)) == 0) {
                float v50 = v38[v48][v47].read();
                v43[(v48 + (v45 * 4))][(v47 + (v44 * 2))] = v50;
              }
              float v51 = v43[(v48 + (v45 * 4))][(v47 + (v44 * 2))];
              if ((v48 + (v45 * 4)) == 0) {
                float v52 = v37[v47][v49].read();
                v42[(v47 + (v44 * 2))][(v49 + (v46 * 4))] = v52;
              }
              float v53 = v42[(v47 + (v44 * 2))][(v49 + (v46 * 4))];
              if ((v47 + (v44 * 2)) == 0) {
                v41[(v48 + (v45 * 4))][(v49 + (v46 * 4))] = v40;
              }
              float v54 = v41[(v48 + (v45 * 4))][(v49 + (v46 * 4))];
              float v55 = v51 * v53;
              float v56 = v54 + v55;
              v41[(v48 + (v45 * 4))][(v49 + (v46 * 4))] = v56;
              if (((v47 + (v44 * 2)) - 63) == 0) {
                float v57 = v41[(v48 + (v45 * 4))][(v49 + (v46 * 4))];
                v39[v48][v49].write(v57);
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
  hls::stream<float> v58[4][2],
  hls::stream<float> v59[2][4],
  hls::stream<float> v60[4][4],
  float v61
) {
  float v62[64][64];
  #pragma HLS array_partition variable=v62 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v62 cyclic dim=2 factor=4

  float v63[64][64];
  #pragma HLS array_partition variable=v63 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v63 cyclic dim=2 factor=4

  float v64[64][64];
  #pragma HLS array_partition variable=v64 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v64 cyclic dim=2 factor=2

  loop20: for (int v65 = 0; v65 < 16; v65++) {
    loop21: for (int v66 = 0; v66 < 32; v66++) {
      loop22: for (int v67 = 0; v67 < 16; v67++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop23: for (int v68 = 0; v68 < 4; v68++) {
          loop24: for (int v69 = 0; v69 < 2; v69++) {
            loop25: for (int v70 = 0; v70 < 4; v70++) {
              if ((v68 + (v65 * 4)) == 0) {
                float v71 = v58[v70][v69].read();
                v64[(v70 + (v67 * 4))][(v69 + (v66 * 2))] = v71;
              }
              float v72 = v64[(v70 + (v67 * 4))][(v69 + (v66 * 2))];
              if ((v70 + (v67 * 4)) == 0) {
                float v73 = v59[v69][v68].read();
                v63[(v69 + (v66 * 2))][(v68 + (v65 * 4))] = v73;
              }
              float v74 = v63[(v69 + (v66 * 2))][(v68 + (v65 * 4))];
              if ((v69 + (v66 * 2)) == 0) {
                v62[(v70 + (v67 * 4))][(v68 + (v65 * 4))] = v61;
              }
              float v75 = v62[(v70 + (v67 * 4))][(v68 + (v65 * 4))];
              float v76 = v72 * v74;
              float v77 = v75 + v76;
              v62[(v70 + (v67 * 4))][(v68 + (v65 * 4))] = v77;
              if (((v69 + (v66 * 2)) - 63) == 0) {
                float v78 = v62[(v70 + (v67 * 4))][(v68 + (v65 * 4))];
                v60[v70][v68].write(v78);
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
  hls::stream<float> v79[4][4],
  hls::stream<float> v80[2][4],
  hls::stream<float> v81[2][4],
  float v82
) {
  float v83[64][64];
  #pragma HLS array_partition variable=v83 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v83 cyclic dim=2 factor=4

  float v84[64][64];
  #pragma HLS array_partition variable=v84 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v84 cyclic dim=2 factor=4

  float v85[64][64];
  #pragma HLS array_partition variable=v85 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v85 cyclic dim=2 factor=4

  loop26: for (int v86 = 0; v86 < 32; v86++) {
    loop27: for (int v87 = 0; v87 < 16; v87++) {
      loop28: for (int v88 = 0; v88 < 16; v88++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop29: for (int v89 = 0; v89 < 2; v89++) {
          loop30: for (int v90 = 0; v90 < 4; v90++) {
            loop31: for (int v91 = 0; v91 < 4; v91++) {
              if ((v91 + (v88 * 4)) == 0) {
                float v92 = v80[v89][v90].read();
                v85[(v89 + (v86 * 2))][(v90 + (v87 * 4))] = v92;
              }
              float v93 = v85[(v89 + (v86 * 2))][(v90 + (v87 * 4))];
              if ((v89 + (v86 * 2)) == 0) {
                float v94 = v79[v90][v91].read();
                v84[(v90 + (v87 * 4))][(v91 + (v88 * 4))] = v94;
              }
              float v95 = v84[(v90 + (v87 * 4))][(v91 + (v88 * 4))];
              if ((v90 + (v87 * 4)) == 0) {
                v83[(v89 + (v86 * 2))][(v91 + (v88 * 4))] = v82;
              }
              float v96 = v83[(v89 + (v86 * 2))][(v91 + (v88 * 4))];
              float v97 = v93 * v95;
              float v98 = v96 + v97;
              v83[(v89 + (v86 * 2))][(v91 + (v88 * 4))] = v98;
              if (((v90 + (v87 * 4)) - 63) == 0) {
                float v99 = v83[(v89 + (v86 * 2))][(v91 + (v88 * 4))];
                v81[v89][v91].write(v99);
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
  hls::stream<float> v100[4][2],
  hls::stream<float> v101[4][4],
  hls::stream<float> v102[4][2],
  float v103
) {
  float v104[64][64];
  #pragma HLS array_partition variable=v104 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v104 cyclic dim=2 factor=2

  float v105[64][64];
  #pragma HLS array_partition variable=v105 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v105 cyclic dim=2 factor=2

  float v106[64][64];
  #pragma HLS array_partition variable=v106 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v106 cyclic dim=2 factor=4

  loop32: for (int v107 = 0; v107 < 32; v107++) {
    loop33: for (int v108 = 0; v108 < 16; v108++) {
      loop34: for (int v109 = 0; v109 < 16; v109++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop35: for (int v110 = 0; v110 < 2; v110++) {
          loop36: for (int v111 = 0; v111 < 4; v111++) {
            loop37: for (int v112 = 0; v112 < 4; v112++) {
              if ((v110 + (v107 * 2)) == 0) {
                float v113 = v101[v112][v111].read();
                v106[(v112 + (v109 * 4))][(v111 + (v108 * 4))] = v113;
              }
              float v114 = v106[(v112 + (v109 * 4))][(v111 + (v108 * 4))];
              if ((v112 + (v109 * 4)) == 0) {
                float v115 = v100[v111][v110].read();
                v105[(v111 + (v108 * 4))][(v110 + (v107 * 2))] = v115;
              }
              float v116 = v105[(v111 + (v108 * 4))][(v110 + (v107 * 2))];
              if ((v111 + (v108 * 4)) == 0) {
                v104[(v112 + (v109 * 4))][(v110 + (v107 * 2))] = v103;
              }
              float v117 = v104[(v112 + (v109 * 4))][(v110 + (v107 * 2))];
              float v118 = v114 * v116;
              float v119 = v117 + v118;
              v104[(v112 + (v109 * 4))][(v110 + (v107 * 2))] = v119;
              if (((v111 + (v108 * 4)) - 63) == 0) {
                float v120 = v104[(v112 + (v109 * 4))][(v110 + (v107 * 2))];
                v102[v112][v110].write(v120);
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
  hls::stream<float> v121[4][4],
  hls::stream<float> v122[2][4],
  hls::stream<float> v123[2][4],
  float v124
) {
  float v125[64][64];
  #pragma HLS array_partition variable=v125 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v125 cyclic dim=2 factor=4

  float v126[64][64];
  #pragma HLS array_partition variable=v126 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v126 cyclic dim=2 factor=4

  float v127[64][64];
  #pragma HLS array_partition variable=v127 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v127 cyclic dim=2 factor=4

  loop38: for (int v128 = 0; v128 < 16; v128++) {
    loop39: for (int v129 = 0; v129 < 16; v129++) {
      loop40: for (int v130 = 0; v130 < 32; v130++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v131 = 0; v131 < 4; v131++) {
          loop42: for (int v132 = 0; v132 < 4; v132++) {
            loop43: for (int v133 = 0; v133 < 2; v133++) {
              if ((v131 + (v128 * 4)) == 0) {
                float v134 = v122[v133][v132].read();
                v127[(v133 + (v130 * 2))][(v132 + (v129 * 4))] = v134;
              }
              float v135 = v127[(v133 + (v130 * 2))][(v132 + (v129 * 4))];
              if ((v133 + (v130 * 2)) == 0) {
                float v136 = v121[v132][v131].read();
                v126[(v132 + (v129 * 4))][(v131 + (v128 * 4))] = v136;
              }
              float v137 = v126[(v132 + (v129 * 4))][(v131 + (v128 * 4))];
              if ((v132 + (v129 * 4)) == 0) {
                v125[(v133 + (v130 * 2))][(v131 + (v128 * 4))] = v124;
              }
              float v138 = v125[(v133 + (v130 * 2))][(v131 + (v128 * 4))];
              float v139 = v135 * v137;
              float v140 = v138 + v139;
              v125[(v133 + (v130 * 2))][(v131 + (v128 * 4))] = v140;
              if (((v132 + (v129 * 4)) - 63) == 0) {
                float v141 = v125[(v133 + (v130 * 2))][(v131 + (v128 * 4))];
                v123[v133][v131].write(v141);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node8(
  hls::stream<float> v142[4][4],
  hls::stream<float> v143[4][2],
  hls::stream<float> v144[4][2],
  float v145
) {
  float v146[64][64];
  #pragma HLS array_partition variable=v146 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v146 cyclic dim=2 factor=2

  float v147[64][64];
  #pragma HLS array_partition variable=v147 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v147 cyclic dim=2 factor=2

  float v148[64][64];
  #pragma HLS array_partition variable=v148 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v148 cyclic dim=2 factor=4

  loop44: for (int v149 = 0; v149 < 16; v149++) {
    loop45: for (int v150 = 0; v150 < 32; v150++) {
      loop46: for (int v151 = 0; v151 < 16; v151++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop47: for (int v152 = 0; v152 < 4; v152++) {
          loop48: for (int v153 = 0; v153 < 2; v153++) {
            loop49: for (int v154 = 0; v154 < 4; v154++) {
              if ((v153 + (v150 * 2)) == 0) {
                float v155 = v142[v154][v152].read();
                v148[(v154 + (v151 * 4))][(v152 + (v149 * 4))] = v155;
              }
              float v156 = v148[(v154 + (v151 * 4))][(v152 + (v149 * 4))];
              if ((v154 + (v151 * 4)) == 0) {
                float v157 = v143[v152][v153].read();
                v147[(v152 + (v149 * 4))][(v153 + (v150 * 2))] = v157;
              }
              float v158 = v147[(v152 + (v149 * 4))][(v153 + (v150 * 2))];
              if ((v152 + (v149 * 4)) == 0) {
                v146[(v154 + (v151 * 4))][(v153 + (v150 * 2))] = v145;
              }
              float v159 = v146[(v154 + (v151 * 4))][(v153 + (v150 * 2))];
              float v160 = v156 * v158;
              float v161 = v159 + v160;
              v146[(v154 + (v151 * 4))][(v153 + (v150 * 2))] = v161;
              if (((v152 + (v149 * 4)) - 63) == 0) {
                float v162 = v146[(v154 + (v151 * 4))][(v153 + (v150 * 2))];
                v144[v154][v153].write(v162);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node9(
  float v163[64][64],
  float v164[64][64],
  hls::stream<float> v165[4][4],
  float v166
) {
  #pragma HLS array_partition variable=v163 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v163 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v164 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v164 cyclic dim=2 factor=4

  float v167[64][64];
  #pragma HLS array_partition variable=v167 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v167 cyclic dim=2 factor=4

  loop50: for (int v168 = 0; v168 < 16; v168++) {
    loop51: for (int v169 = 0; v169 < 32; v169++) {
      loop52: for (int v170 = 0; v170 < 16; v170++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop53: for (int v171 = 0; v171 < 4; v171++) {
          loop54: for (int v172 = 0; v172 < 2; v172++) {
            loop55: for (int v173 = 0; v173 < 4; v173++) {
              float v174 = v163[(v171 + (v168 * 4))][(v172 + (v169 * 2))];
              float v175 = v164[(v172 + (v169 * 2))][(v173 + (v170 * 4))];
              if ((v172 + (v169 * 2)) == 0) {
                v167[(v171 + (v168 * 4))][(v173 + (v170 * 4))] = v166;
              }
              float v176 = v167[(v171 + (v168 * 4))][(v173 + (v170 * 4))];
              float v177 = v174 * v175;
              float v178 = v176 + v177;
              v167[(v171 + (v168 * 4))][(v173 + (v170 * 4))] = v178;
              if (((v172 + (v169 * 2)) - 63) == 0) {
                float v179 = v167[(v171 + (v168 * 4))][(v173 + (v170 * 4))];
                v165[v171][v173].write(v179);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node10(
  float v180[64][64],
  float v181[64][64],
  hls::stream<float> v182[2][4],
  float v183
) {
  #pragma HLS array_partition variable=v180 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v180 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v181 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v181 cyclic dim=2 factor=4

  float v184[64][64];
  #pragma HLS array_partition variable=v184 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v184 cyclic dim=2 factor=4

  loop56: for (int v185 = 0; v185 < 32; v185++) {
    loop57: for (int v186 = 0; v186 < 16; v186++) {
      loop58: for (int v187 = 0; v187 < 16; v187++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop59: for (int v188 = 0; v188 < 2; v188++) {
          loop60: for (int v189 = 0; v189 < 4; v189++) {
            loop61: for (int v190 = 0; v190 < 4; v190++) {
              float v191 = v180[(v188 + (v185 * 2))][(v189 + (v186 * 4))];
              float v192 = v181[(v189 + (v186 * 4))][(v190 + (v187 * 4))];
              if ((v189 + (v186 * 4)) == 0) {
                v184[(v188 + (v185 * 2))][(v190 + (v187 * 4))] = v183;
              }
              float v193 = v184[(v188 + (v185 * 2))][(v190 + (v187 * 4))];
              float v194 = v191 * v192;
              float v195 = v193 + v194;
              v184[(v188 + (v185 * 2))][(v190 + (v187 * 4))] = v195;
              if (((v189 + (v186 * 4)) - 63) == 0) {
                float v196 = v184[(v188 + (v185 * 2))][(v190 + (v187 * 4))];
                v182[v188][v190].write(v196);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node11(
  float v197[64][64],
  float v198[64][64],
  hls::stream<float> v199[4][2],
  float v200
) {
  #pragma HLS array_partition variable=v197 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v197 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v198 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v198 cyclic dim=2 factor=2

  float v201[64][64];
  #pragma HLS array_partition variable=v201 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v201 cyclic dim=2 factor=2

  loop62: for (int v202 = 0; v202 < 32; v202++) {
    loop63: for (int v203 = 0; v203 < 16; v203++) {
      loop64: for (int v204 = 0; v204 < 16; v204++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop65: for (int v205 = 0; v205 < 2; v205++) {
          loop66: for (int v206 = 0; v206 < 4; v206++) {
            loop67: for (int v207 = 0; v207 < 4; v207++) {
              float v208 = v197[(v207 + (v204 * 4))][(v206 + (v203 * 4))];
              float v209 = v198[(v206 + (v203 * 4))][(v205 + (v202 * 2))];
              if ((v206 + (v203 * 4)) == 0) {
                v201[(v207 + (v204 * 4))][(v205 + (v202 * 2))] = v200;
              }
              float v210 = v201[(v207 + (v204 * 4))][(v205 + (v202 * 2))];
              float v211 = v208 * v209;
              float v212 = v210 + v211;
              v201[(v207 + (v204 * 4))][(v205 + (v202 * 2))] = v212;
              if (((v206 + (v203 * 4)) - 63) == 0) {
                float v213 = v201[(v207 + (v204 * 4))][(v205 + (v202 * 2))];
                v199[v207][v205].write(v213);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node12(
  float v214[64][64],
  float v215[64][64],
  hls::stream<float> v216[4][4],
  float v217
) {
  #pragma HLS array_partition variable=v214 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v214 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v215 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v215 cyclic dim=2 factor=2

  float v218[64][64];
  #pragma HLS array_partition variable=v218 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v218 cyclic dim=2 factor=4

  loop68: for (int v219 = 0; v219 < 16; v219++) {
    loop69: for (int v220 = 0; v220 < 32; v220++) {
      loop70: for (int v221 = 0; v221 < 16; v221++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v222 = 0; v222 < 4; v222++) {
          loop72: for (int v223 = 0; v223 < 2; v223++) {
            loop73: for (int v224 = 0; v224 < 4; v224++) {
              float v225 = v215[(v224 + (v221 * 4))][(v223 + (v220 * 2))];
              float v226 = v214[(v223 + (v220 * 2))][(v222 + (v219 * 4))];
              if ((v223 + (v220 * 2)) == 0) {
                v218[(v224 + (v221 * 4))][(v222 + (v219 * 4))] = v217;
              }
              float v227 = v218[(v224 + (v221 * 4))][(v222 + (v219 * 4))];
              float v228 = v225 * v226;
              float v229 = v227 + v228;
              v218[(v224 + (v221 * 4))][(v222 + (v219 * 4))] = v229;
              if (((v223 + (v220 * 2)) - 63) == 0) {
                float v230 = v218[(v224 + (v221 * 4))][(v222 + (v219 * 4))];
                v216[v224][v222].write(v230);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node13(
  float v231[64][64],
  float v232[64][64],
  hls::stream<float> v233[4][4],
  float v234
) {
  #pragma HLS array_partition variable=v231 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v232 cyclic dim=2 factor=4

  float v235[64][64];
  #pragma HLS array_partition variable=v235 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v235 cyclic dim=2 factor=4

  loop74: for (int v236 = 0; v236 < 16; v236++) {
    loop75: for (int v237 = 0; v237 < 64; v237++) {
      loop76: for (int v238 = 0; v238 < 16; v238++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop77: for (int v239 = 0; v239 < 4; v239++) {
          loop78: for (int v240 = 0; v240 < 1; v240++) {
            loop79: for (int v241 = 0; v241 < 4; v241++) {
              float v242 = v231[(v241 + (v238 * 4))][(v237 + v240)];
              float v243 = v232[(v237 + v240)][(v239 + (v236 * 4))];
              if ((v237 + v240) == 0) {
                v235[(v241 + (v238 * 4))][(v239 + (v236 * 4))] = v234;
              }
              float v244 = v235[(v241 + (v238 * 4))][(v239 + (v236 * 4))];
              float v245 = v242 * v243;
              float v246 = v244 + v245;
              v235[(v241 + (v238 * 4))][(v239 + (v236 * 4))] = v246;
              if (((v237 + v240) - 63) == 0) {
                float v247 = v235[(v241 + (v238 * 4))][(v239 + (v236 * 4))];
                v233[v241][v239].write(v247);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node14(
  float v248[64][64],
  float v249[64][64],
  hls::stream<float> v250[2][4],
  float v251
) {
  #pragma HLS array_partition variable=v248 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v248 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v249 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v249 cyclic dim=2 factor=4

  float v252[64][64];
  #pragma HLS array_partition variable=v252 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v252 cyclic dim=2 factor=4

  loop80: for (int v253 = 0; v253 < 16; v253++) {
    loop81: for (int v254 = 0; v254 < 16; v254++) {
      loop82: for (int v255 = 0; v255 < 32; v255++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop83: for (int v256 = 0; v256 < 4; v256++) {
          loop84: for (int v257 = 0; v257 < 4; v257++) {
            loop85: for (int v258 = 0; v258 < 2; v258++) {
              float v259 = v249[(v258 + (v255 * 2))][(v257 + (v254 * 4))];
              float v260 = v248[(v257 + (v254 * 4))][(v256 + (v253 * 4))];
              if ((v257 + (v254 * 4)) == 0) {
                v252[(v258 + (v255 * 2))][(v256 + (v253 * 4))] = v251;
              }
              float v261 = v252[(v258 + (v255 * 2))][(v256 + (v253 * 4))];
              float v262 = v259 * v260;
              float v263 = v261 + v262;
              v252[(v258 + (v255 * 2))][(v256 + (v253 * 4))] = v263;
              if (((v257 + (v254 * 4)) - 63) == 0) {
                float v264 = v252[(v258 + (v255 * 2))][(v256 + (v253 * 4))];
                v250[v258][v256].write(v264);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node15(
  float v265[64][64],
  float v266[64][64],
  hls::stream<float> v267[4][2],
  float v268
) {
  #pragma HLS array_partition variable=v265 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v265 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v266 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v266 cyclic dim=2 factor=2

  float v269[64][64];
  #pragma HLS array_partition variable=v269 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v269 cyclic dim=2 factor=2

  loop86: for (int v270 = 0; v270 < 16; v270++) {
    loop87: for (int v271 = 0; v271 < 16; v271++) {
      loop88: for (int v272 = 0; v272 < 32; v272++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop89: for (int v273 = 0; v273 < 4; v273++) {
          loop90: for (int v274 = 0; v274 < 4; v274++) {
            loop91: for (int v275 = 0; v275 < 2; v275++) {
              float v276 = v265[(v273 + (v270 * 4))][(v274 + (v271 * 4))];
              float v277 = v266[(v274 + (v271 * 4))][(v275 + (v272 * 2))];
              if ((v274 + (v271 * 4)) == 0) {
                v269[(v273 + (v270 * 4))][(v275 + (v272 * 2))] = v268;
              }
              float v278 = v269[(v273 + (v270 * 4))][(v275 + (v272 * 2))];
              float v279 = v276 * v277;
              float v280 = v278 + v279;
              v269[(v273 + (v270 * 4))][(v275 + (v272 * 2))] = v280;
              if (((v274 + (v271 * 4)) - 63) == 0) {
                float v281 = v269[(v273 + (v270 * 4))][(v275 + (v272 * 2))];
                v267[v273][v275].write(v281);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node16(
  float v282[64][64],
  float v283[64][64],
  hls::stream<float> v284[4][4],
  float v285
) {
  #pragma HLS array_partition variable=v282 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v282 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v283 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v283 cyclic dim=2 factor=4

  float v286[64][64];
  #pragma HLS array_partition variable=v286 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v286 cyclic dim=2 factor=4

  loop92: for (int v287 = 0; v287 < 16; v287++) {
    loop93: for (int v288 = 0; v288 < 32; v288++) {
      loop94: for (int v289 = 0; v289 < 16; v289++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop95: for (int v290 = 0; v290 < 4; v290++) {
          loop96: for (int v291 = 0; v291 < 2; v291++) {
            loop97: for (int v292 = 0; v292 < 4; v292++) {
              float v293 = v282[(v292 + (v289 * 4))][(v291 + (v288 * 2))];
              float v294 = v283[(v291 + (v288 * 2))][(v290 + (v287 * 4))];
              if ((v291 + (v288 * 2)) == 0) {
                v286[(v292 + (v289 * 4))][(v290 + (v287 * 4))] = v285;
              }
              float v295 = v286[(v292 + (v289 * 4))][(v290 + (v287 * 4))];
              float v296 = v293 * v294;
              float v297 = v295 + v296;
              v286[(v292 + (v289 * 4))][(v290 + (v287 * 4))] = v297;
              if (((v291 + (v288 * 2)) - 63) == 0) {
                float v298 = v286[(v292 + (v289 * 4))][(v290 + (v287 * 4))];
                v284[v292][v290].write(v298);
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
  float v299[64][64],
  float v300[64][64],
  float v301[64][64],
  float v302[64][64],
  float v303[64][64],
  float v304[64][64],
  float v305[64][64],
  float v306[64][64],
  float v307[64][64],
  float v308[64][64],
  float v309[64][64],
  float v310[64][64],
  float v311[64][64],
  float v312[64][64],
  float v313[64][64],
  float v314[64][64],
  float v315[64][64]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v299 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v299 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v300 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v300 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v301 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v301 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v302 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v302 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v303 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v303 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v304 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v304 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v305 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v306 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v307 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v307 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v308 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v308 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v309 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v309 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v310 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v310 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v311 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v311 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v312 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v312 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v313 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v313 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v314 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v314 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v315 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v315 cyclic dim=2 factor=4

  hls::stream<float> v316[4][4];
	#pragma HLS STREAM variable=v316 depth=256
  hls::stream<float> v317[4][4];
	#pragma HLS STREAM variable=v317 depth=256
  hls::stream<float> v318[4][4];
	#pragma HLS STREAM variable=v318 depth=256
  hls::stream<float> v319[4][4];
	#pragma HLS STREAM variable=v319 depth=256
  hls::stream<float> v320[2][4];
	#pragma HLS STREAM variable=v320 depth=512
  hls::stream<float> v321[4][2];
	#pragma HLS STREAM variable=v321 depth=512
  hls::stream<float> v322[2][4];
	#pragma HLS STREAM variable=v322 depth=512
  hls::stream<float> v323[4][2];
	#pragma HLS STREAM variable=v323 depth=512
  hls::stream<float> v324[4][4];
	#pragma HLS STREAM variable=v324 depth=256
  hls::stream<float> v325[2][4];
	#pragma HLS STREAM variable=v325 depth=512
  hls::stream<float> v326[4][2];
	#pragma HLS STREAM variable=v326 depth=512
  hls::stream<float> v327[4][4];
	#pragma HLS STREAM variable=v327 depth=256
  hls::stream<float> v328[4][4];
	#pragma HLS STREAM variable=v328 depth=256
  hls::stream<float> v329[2][4];
	#pragma HLS STREAM variable=v329 depth=512
  hls::stream<float> v330[4][2];
	#pragma HLS STREAM variable=v330 depth=512
  hls::stream<float> v331[4][4];
	#pragma HLS STREAM variable=v331 depth=256
  node16(v299, v300, v331, 0.000000);
  node15(v301, v302, v330, 0.000000);
  node14(v304, v303, v329, 0.000000);
  node13(v305, v306, v328, 0.000000);
  node12(v308, v307, v327, 0.000000);
  node11(v309, v310, v326, 0.000000);
  node10(v311, v312, v325, 0.000000);
  node9(v313, v314, v324, 0.000000);
  node8(v331, v330, v323, 0.000000);
  node7(v328, v329, v322, 0.000000);
  node6(v326, v327, v321, 0.000000);
  node5(v324, v325, v320, 0.000000);
  node4(v323, v322, v319, 0.000000);
  node3(v320, v321, v318, 0.000000);
  node2(v319, v318, v317, 0.000000);
  node1(v317, v316, 1000000.000000);
  node0(v316, v315);
  return ;
}

