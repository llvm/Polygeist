
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
  hls::stream<float> v0[1][1],
  float v1[16][16]
) {
  loop0: for (int v2 = 0; v2 < 16; v2++) {
    loop1: for (int v3 = 0; v3 < 16; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 1; v4++) {
        loop3: for (int v5 = 0; v5 < 1; v5++) {
          float v6 = v0[v5][v4].read();
          v1[(v3 + v5)][(v2 + v4)] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[1][1],
  hls::stream<float> v8[1][1],
  float v9
) {
  loop4: for (int v10 = 0; v10 < 16; v10++) {
    loop5: for (int v11 = 0; v11 < 16; v11++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v12 = 0; v12 < 1; v12++) {
        loop7: for (int v13 = 0; v13 < 1; v13++) {
          float v14 = v7[v13][v12].read();
          float v15 = v14 / v9;
          v8[v13][v12].write(v15);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v16[1][1],
  hls::stream<float> v17[1][1],
  hls::stream<float> v18[1][1],
  float v19
) {
  float v20[16][16];
  float v21[16][16];
  float v22[16][16];
  loop8: for (int v23 = 0; v23 < 16; v23++) {
    loop9: for (int v24 = 0; v24 < 16; v24++) {
      loop10: for (int v25 = 0; v25 < 16; v25++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v26 = 0; v26 < 1; v26++) {
          loop12: for (int v27 = 0; v27 < 1; v27++) {
            loop13: for (int v28 = 0; v28 < 1; v28++) {
              if ((v23 + v26) == 0) {
                float v29 = v17[v28][v27].read();
                v22[(v25 + v28)][(v24 + v27)] = v29;
              }
              float v30 = v22[(v25 + v28)][(v24 + v27)];
              if ((v25 + v28) == 0) {
                float v31 = v16[v27][v26].read();
                v21[(v24 + v27)][(v23 + v26)] = v31;
              }
              float v32 = v21[(v24 + v27)][(v23 + v26)];
              if ((v24 + v27) == 0) {
                v20[(v25 + v28)][(v23 + v26)] = v19;
              }
              float v33 = v20[(v25 + v28)][(v23 + v26)];
              float v34 = v30 * v32;
              float v35 = v33 + v34;
              v20[(v25 + v28)][(v23 + v26)] = v35;
              if (((v24 + v27) - 15) == 0) {
                float v36 = v20[(v25 + v28)][(v23 + v26)];
                v18[v28][v26].write(v36);
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
  hls::stream<float> v37[1][8],
  hls::stream<float> v38[8][1],
  hls::stream<float> v39[1][1],
  float v40
) {
  float v41[16][16];
  float v42[128][16];
  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=8

  float v43[16][128];
  #pragma HLS array_partition variable=v43 cyclic dim=2 factor=8

  loop14: for (int v44 = 0; v44 < 16; v44++) {
    loop15: for (int v45 = 0; v45 < 16; v45++) {
      loop16: for (int v46 = 0; v46 < 16; v46++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop17: for (int v47 = 0; v47 < 1; v47++) {
          loop18: for (int v48 = 0; v48 < 8; v48++) {
            loop19: for (int v49 = 0; v49 < 1; v49++) {
              if ((v44 + v47) == 0) {
                float v50 = v37[v49][v48].read();
                v43[(v46 + v49)][(v48 + (v45 * 8))] = v50;
              }
              float v51 = v43[(v46 + v49)][(v48 + (v45 * 8))];
              if ((v46 + v49) == 0) {
                float v52 = v38[v48][v47].read();
                v42[(v48 + (v45 * 8))][(v44 + v47)] = v52;
              }
              float v53 = v42[(v48 + (v45 * 8))][(v44 + v47)];
              if ((v48 + (v45 * 8)) == 0) {
                v41[(v46 + v49)][(v44 + v47)] = v40;
              }
              float v54 = v41[(v46 + v49)][(v44 + v47)];
              float v55 = v51 * v53;
              float v56 = v54 + v55;
              v41[(v46 + v49)][(v44 + v47)] = v56;
              if (((v48 + (v45 * 8)) - 127) == 0) {
                float v57 = v41[(v46 + v49)][(v44 + v47)];
                v39[v49][v47].write(v57);
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
  float v58[128][16],
  hls::stream<float> v59[1][8],
  hls::stream<float> v60[1][1],
  float v61
) {
  #pragma HLS array_partition variable=v58 cyclic dim=1 factor=8

  float v62[16][16];
  float v63[128][16];
  #pragma HLS array_partition variable=v63 cyclic dim=1 factor=8

  float v64[16][128];
  #pragma HLS array_partition variable=v64 cyclic dim=2 factor=8

  loop20: for (int v65 = 0; v65 < 16; v65++) {
    loop21: for (int v66 = 0; v66 < 16; v66++) {
      loop22: for (int v67 = 0; v67 < 16; v67++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop23: for (int v68 = 0; v68 < 8; v68++) {
          loop24: for (int v69 = 0; v69 < 1; v69++) {
            loop25: for (int v70 = 0; v70 < 1; v70++) {
              if ((v66 + v69) == 0) {
                float v71 = v59[v70][v68].read();
                v64[(v67 + v70)][(v68 + (v65 * 8))] = v71;
              }
              float v72 = v64[(v67 + v70)][(v68 + (v65 * 8))];
              if ((v67 + v70) == 0) {
                float v73 = v58[(v68 + (v65 * 8))][(v66 + v69)];
                v63[(v68 + (v65 * 8))][(v66 + v69)] = v73;
              }
              float v74 = v63[(v68 + (v65 * 8))][(v66 + v69)];
              if ((v68 + (v65 * 8)) == 0) {
                v62[(v67 + v70)][(v66 + v69)] = v61;
              }
              float v75 = v62[(v67 + v70)][(v66 + v69)];
              float v76 = v72 * v74;
              float v77 = v75 + v76;
              v62[(v67 + v70)][(v66 + v69)] = v77;
              if (((v68 + (v65 * 8)) - 127) == 0) {
                float v78 = v62[(v67 + v70)][(v66 + v69)];
                v60[v70][v69].write(v78);
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
  hls::stream<float> v79[1][1],
  hls::stream<float> v80[8][1],
  hls::stream<float> v81[8][1],
  float v82
) {
  float v83[128][16];
  #pragma HLS array_partition variable=v83 cyclic dim=1 factor=8

  float v84[32][16];
  float v85[128][32];
  #pragma HLS array_partition variable=v85 cyclic dim=1 factor=8

  loop26: for (int v86 = 0; v86 < 32; v86++) {
    loop27: for (int v87 = 0; v87 < 16; v87++) {
      loop28: for (int v88 = 0; v88 < 16; v88++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop29: for (int v89 = 0; v89 < 1; v89++) {
          loop30: for (int v90 = 0; v90 < 1; v90++) {
            loop31: for (int v91 = 0; v91 < 8; v91++) {
              if ((v87 + v90) == 0) {
                float v92 = v80[v91][v89].read();
                v85[(v91 + (v88 * 8))][(v86 + v89)] = v92;
              }
              float v93 = v85[(v91 + (v88 * 8))][(v86 + v89)];
              if ((v91 + (v88 * 8)) == 0) {
                float v94 = v79[v89][v90].read();
                v84[(v86 + v89)][(v87 + v90)] = v94;
              }
              float v95 = v84[(v86 + v89)][(v87 + v90)];
              if ((v86 + v89) == 0) {
                v83[(v91 + (v88 * 8))][(v87 + v90)] = v82;
              }
              float v96 = v83[(v91 + (v88 * 8))][(v87 + v90)];
              float v97 = v93 * v95;
              float v98 = v96 + v97;
              v83[(v91 + (v88 * 8))][(v87 + v90)] = v98;
              if (((v86 + v89) - 31) == 0) {
                float v99 = v83[(v91 + (v88 * 8))][(v87 + v90)];
                v81[v91][v90].write(v99);
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
  hls::stream<float> v100[4][8],
  hls::stream<float> v101[1][4],
  hls::stream<float> v102[1][8],
  float v103
) {
  float v104[16][128];
  #pragma HLS array_partition variable=v104 cyclic dim=2 factor=8

  float v105[128][128];
  #pragma HLS array_partition variable=v105 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v105 cyclic dim=2 factor=8

  float v106[16][128];
  #pragma HLS array_partition variable=v106 cyclic dim=2 factor=4

  loop32: for (int v107 = 0; v107 < 32; v107++) {
    loop33: for (int v108 = 0; v108 < 16; v108++) {
      loop34: for (int v109 = 0; v109 < 16; v109++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop35: for (int v110 = 0; v110 < 4; v110++) {
          loop36: for (int v111 = 0; v111 < 8; v111++) {
            loop37: for (int v112 = 0; v112 < 1; v112++) {
              if ((v111 + (v108 * 8)) == 0) {
                float v113 = v101[v112][v110].read();
                v106[(v109 + v112)][(v110 + (v107 * 4))] = v113;
              }
              float v114 = v106[(v109 + v112)][(v110 + (v107 * 4))];
              if ((v109 + v112) == 0) {
                float v115 = v100[v110][v111].read();
                v105[(v110 + (v107 * 4))][(v111 + (v108 * 8))] = v115;
              }
              float v116 = v105[(v110 + (v107 * 4))][(v111 + (v108 * 8))];
              if ((v110 + (v107 * 4)) == 0) {
                v104[(v109 + v112)][(v111 + (v108 * 8))] = v103;
              }
              float v117 = v104[(v109 + v112)][(v111 + (v108 * 8))];
              float v118 = v114 * v116;
              float v119 = v117 + v118;
              v104[(v109 + v112)][(v111 + (v108 * 8))] = v119;
              if (((v110 + (v107 * 4)) - 127) == 0) {
                float v120 = v104[(v109 + v112)][(v111 + (v108 * 8))];
                v102[v112][v111].write(v120);
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
  hls::stream<float> v121[2][1],
  hls::stream<float> v122[8][2],
  float v123[128][16],
  float v124
) {
  #pragma HLS array_partition variable=v123 cyclic dim=1 factor=8

  float v125[128][16];
  #pragma HLS array_partition variable=v125 cyclic dim=1 factor=8

  float v126[32][16];
  #pragma HLS array_partition variable=v126 cyclic dim=1 factor=2

  float v127[128][32];
  #pragma HLS array_partition variable=v127 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v127 cyclic dim=2 factor=2

  loop38: for (int v128 = 0; v128 < 16; v128++) {
    loop39: for (int v129 = 0; v129 < 16; v129++) {
      loop40: for (int v130 = 0; v130 < 16; v130++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v131 = 0; v131 < 2; v131++) {
          loop42: for (int v132 = 0; v132 < 1; v132++) {
            loop43: for (int v133 = 0; v133 < 8; v133++) {
              if ((v129 + v132) == 0) {
                float v134 = v122[v133][v131].read();
                v127[(v133 + (v130 * 8))][(v131 + (v128 * 2))] = v134;
              }
              float v135 = v127[(v133 + (v130 * 8))][(v131 + (v128 * 2))];
              if ((v133 + (v130 * 8)) == 0) {
                float v136 = v121[v131][v132].read();
                v126[(v131 + (v128 * 2))][(v129 + v132)] = v136;
              }
              float v137 = v126[(v131 + (v128 * 2))][(v129 + v132)];
              if ((v131 + (v128 * 2)) == 0) {
                v125[(v133 + (v130 * 8))][(v129 + v132)] = v124;
              }
              float v138 = v125[(v133 + (v130 * 8))][(v129 + v132)];
              float v139 = v135 * v137;
              float v140 = v138 + v139;
              v125[(v133 + (v130 * 8))][(v129 + v132)] = v140;
              if (((v131 + (v128 * 2)) - 31) == 0) {
                float v141 = v125[(v133 + (v130 * 8))][(v129 + v132)];
                v123[(v133 + (v130 * 8))][(v129 + v132)] = v141;
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
  hls::stream<float> v142[8][8],
  hls::stream<float> v143[1][8],
  hls::stream<float> v144[1][8],
  float v145
) {
  float v146[16][128];
  #pragma HLS array_partition variable=v146 cyclic dim=2 factor=8

  float v147[128][128];
  #pragma HLS array_partition variable=v147 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v147 cyclic dim=2 factor=8

  float v148[16][128];
  #pragma HLS array_partition variable=v148 cyclic dim=2 factor=8

  loop44: for (int v149 = 0; v149 < 16; v149++) {
    loop45: for (int v150 = 0; v150 < 16; v150++) {
      loop46: for (int v151 = 0; v151 < 16; v151++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop47: for (int v152 = 0; v152 < 8; v152++) {
          loop48: for (int v153 = 0; v153 < 8; v153++) {
            loop49: for (int v154 = 0; v154 < 1; v154++) {
              if ((v153 + (v150 * 8)) == 0) {
                float v155 = v143[v154][v152].read();
                v148[(v151 + v154)][(v152 + (v149 * 8))] = v155;
              }
              float v156 = v148[(v151 + v154)][(v152 + (v149 * 8))];
              if ((v151 + v154) == 0) {
                float v157 = v142[v152][v153].read();
                v147[(v152 + (v149 * 8))][(v153 + (v150 * 8))] = v157;
              }
              float v158 = v147[(v152 + (v149 * 8))][(v153 + (v150 * 8))];
              if ((v152 + (v149 * 8)) == 0) {
                v146[(v151 + v154)][(v153 + (v150 * 8))] = v145;
              }
              float v159 = v146[(v151 + v154)][(v153 + (v150 * 8))];
              float v160 = v156 * v158;
              float v161 = v159 + v160;
              v146[(v151 + v154)][(v153 + (v150 * 8))] = v161;
              if (((v152 + (v149 * 8)) - 127) == 0) {
                float v162 = v146[(v151 + v154)][(v153 + (v150 * 8))];
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
  float v163[16][16],
  float v164[32][16],
  hls::stream<float> v165[1][1],
  float v166
) {
  float v167[32][16];
  loop50: for (int v168 = 0; v168 < 32; v168++) {
    loop51: for (int v169 = 0; v169 < 16; v169++) {
      loop52: for (int v170 = 0; v170 < 16; v170++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop53: for (int v171 = 0; v171 < 1; v171++) {
          loop54: for (int v172 = 0; v172 < 1; v172++) {
            loop55: for (int v173 = 0; v173 < 1; v173++) {
              float v174 = v164[(v168 + v171)][(v169 + v172)];
              float v175 = v163[(v169 + v172)][(v170 + v173)];
              if ((v169 + v172) == 0) {
                v167[(v168 + v171)][(v170 + v173)] = v166;
              }
              float v176 = v167[(v168 + v171)][(v170 + v173)];
              float v177 = v174 * v175;
              float v178 = v176 + v177;
              v167[(v168 + v171)][(v170 + v173)] = v178;
              if (((v169 + v172) - 15) == 0) {
                float v179 = v167[(v168 + v171)][(v170 + v173)];
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
  float v180[64][32],
  float v181[128][64],
  hls::stream<float> v182[8][1],
  float v183
) {
  #pragma HLS array_partition variable=v180 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v181 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v181 cyclic dim=2 factor=4

  float v184[128][32];
  #pragma HLS array_partition variable=v184 cyclic dim=1 factor=8

  loop56: for (int v185 = 0; v185 < 32; v185++) {
    loop57: for (int v186 = 0; v186 < 16; v186++) {
      loop58: for (int v187 = 0; v187 < 16; v187++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop59: for (int v188 = 0; v188 < 1; v188++) {
          loop60: for (int v189 = 0; v189 < 4; v189++) {
            loop61: for (int v190 = 0; v190 < 8; v190++) {
              float v191 = v181[(v190 + (v187 * 8))][(v189 + (v186 * 4))];
              float v192 = v180[(v189 + (v186 * 4))][(v185 + v188)];
              if ((v189 + (v186 * 4)) == 0) {
                v184[(v190 + (v187 * 8))][(v185 + v188)] = v183;
              }
              float v193 = v184[(v190 + (v187 * 8))][(v185 + v188)];
              float v194 = v191 * v192;
              float v195 = v193 + v194;
              v184[(v190 + (v187 * 8))][(v185 + v188)] = v195;
              if (((v189 + (v186 * 4)) - 63) == 0) {
                float v196 = v184[(v190 + (v187 * 8))][(v185 + v188)];
                v182[v190][v188].write(v196);
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
  float v197[128][64],
  float v198[64][128],
  hls::stream<float> v199[4][8],
  float v200
) {
  #pragma HLS array_partition variable=v197 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v197 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v198 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v198 cyclic dim=2 factor=8

  float v201[128][128];
  #pragma HLS array_partition variable=v201 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v201 cyclic dim=2 factor=8

  loop62: for (int v202 = 0; v202 < 32; v202++) {
    loop63: for (int v203 = 0; v203 < 16; v203++) {
      loop64: for (int v204 = 0; v204 < 16; v204++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop65: for (int v205 = 0; v205 < 4; v205++) {
          loop66: for (int v206 = 0; v206 < 4; v206++) {
            loop67: for (int v207 = 0; v207 < 8; v207++) {
              float v208 = v197[(v205 + (v202 * 4))][(v206 + (v203 * 4))];
              float v209 = v198[(v206 + (v203 * 4))][(v207 + (v204 * 8))];
              if ((v206 + (v203 * 4)) == 0) {
                v201[(v205 + (v202 * 4))][(v207 + (v204 * 8))] = v200;
              }
              float v210 = v201[(v205 + (v202 * 4))][(v207 + (v204 * 8))];
              float v211 = v208 * v209;
              float v212 = v210 + v211;
              v201[(v205 + (v202 * 4))][(v207 + (v204 * 8))] = v212;
              if (((v206 + (v203 * 4)) - 63) == 0) {
                float v213 = v201[(v205 + (v202 * 4))][(v207 + (v204 * 8))];
                v199[v205][v207].write(v213);
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
  float v214[16][32],
  float v215[32][128],
  hls::stream<float> v216[1][4],
  float v217
) {
  #pragma HLS array_partition variable=v214 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v215 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v215 cyclic dim=2 factor=4

  float v218[16][128];
  #pragma HLS array_partition variable=v218 cyclic dim=2 factor=4

  loop68: for (int v219 = 0; v219 < 32; v219++) {
    loop69: for (int v220 = 0; v220 < 16; v220++) {
      loop70: for (int v221 = 0; v221 < 16; v221++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v222 = 0; v222 < 4; v222++) {
          loop72: for (int v223 = 0; v223 < 2; v223++) {
            loop73: for (int v224 = 0; v224 < 1; v224++) {
              float v225 = v214[(v221 + v224)][(v223 + (v220 * 2))];
              float v226 = v215[(v223 + (v220 * 2))][(v222 + (v219 * 4))];
              if ((v223 + (v220 * 2)) == 0) {
                v218[(v221 + v224)][(v222 + (v219 * 4))] = v217;
              }
              float v227 = v218[(v221 + v224)][(v222 + (v219 * 4))];
              float v228 = v225 * v226;
              float v229 = v227 + v228;
              v218[(v221 + v224)][(v222 + (v219 * 4))] = v229;
              if (((v223 + (v220 * 2)) - 31) == 0) {
                float v230 = v218[(v221 + v224)][(v222 + (v219 * 4))];
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
  float v231[32][16],
  float v232[16][16],
  hls::stream<float> v233[2][1],
  float v234
) {
  #pragma HLS array_partition variable=v231 cyclic dim=1 factor=2

  float v235[32][16];
  #pragma HLS array_partition variable=v235 cyclic dim=1 factor=2

  loop74: for (int v236 = 0; v236 < 16; v236++) {
    loop75: for (int v237 = 0; v237 < 16; v237++) {
      loop76: for (int v238 = 0; v238 < 16; v238++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop77: for (int v239 = 0; v239 < 2; v239++) {
          loop78: for (int v240 = 0; v240 < 1; v240++) {
            loop79: for (int v241 = 0; v241 < 1; v241++) {
              float v242 = v231[(v239 + (v236 * 2))][(v237 + v240)];
              float v243 = v232[(v237 + v240)][(v238 + v241)];
              if ((v237 + v240) == 0) {
                v235[(v239 + (v236 * 2))][(v238 + v241)] = v234;
              }
              float v244 = v235[(v239 + (v236 * 2))][(v238 + v241)];
              float v245 = v242 * v243;
              float v246 = v244 + v245;
              v235[(v239 + (v236 * 2))][(v238 + v241)] = v246;
              if (((v237 + v240) - 15) == 0) {
                float v247 = v235[(v239 + (v236 * 2))][(v238 + v241)];
                v233[v239][v241].write(v247);
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
  float v248[64][32],
  float v249[128][64],
  hls::stream<float> v250[8][2],
  float v251
) {
  #pragma HLS array_partition variable=v248 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v248 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v249 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v249 cyclic dim=2 factor=4

  float v252[128][32];
  #pragma HLS array_partition variable=v252 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v252 cyclic dim=2 factor=2

  loop80: for (int v253 = 0; v253 < 16; v253++) {
    loop81: for (int v254 = 0; v254 < 16; v254++) {
      loop82: for (int v255 = 0; v255 < 16; v255++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop83: for (int v256 = 0; v256 < 2; v256++) {
          loop84: for (int v257 = 0; v257 < 4; v257++) {
            loop85: for (int v258 = 0; v258 < 8; v258++) {
              float v259 = v249[(v258 + (v255 * 8))][(v257 + (v254 * 4))];
              float v260 = v248[(v257 + (v254 * 4))][(v256 + (v253 * 2))];
              if ((v257 + (v254 * 4)) == 0) {
                v252[(v258 + (v255 * 8))][(v256 + (v253 * 2))] = v251;
              }
              float v261 = v252[(v258 + (v255 * 8))][(v256 + (v253 * 2))];
              float v262 = v259 * v260;
              float v263 = v261 + v262;
              v252[(v258 + (v255 * 8))][(v256 + (v253 * 2))] = v263;
              if (((v257 + (v254 * 4)) - 63) == 0) {
                float v264 = v252[(v258 + (v255 * 8))][(v256 + (v253 * 2))];
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
  float v265[128][64],
  float v266[64][128],
  hls::stream<float> v267[8][8],
  float v268
) {
  #pragma HLS array_partition variable=v265 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v265 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v266 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v266 cyclic dim=2 factor=8

  float v269[128][128];
  #pragma HLS array_partition variable=v269 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v269 cyclic dim=2 factor=8

  loop86: for (int v270 = 0; v270 < 16; v270++) {
    loop87: for (int v271 = 0; v271 < 32; v271++) {
      loop88: for (int v272 = 0; v272 < 16; v272++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop89: for (int v273 = 0; v273 < 8; v273++) {
          loop90: for (int v274 = 0; v274 < 2; v274++) {
            loop91: for (int v275 = 0; v275 < 8; v275++) {
              float v276 = v265[(v273 + (v270 * 8))][(v274 + (v271 * 2))];
              float v277 = v266[(v274 + (v271 * 2))][(v275 + (v272 * 8))];
              if ((v274 + (v271 * 2)) == 0) {
                v269[(v273 + (v270 * 8))][(v275 + (v272 * 8))] = v268;
              }
              float v278 = v269[(v273 + (v270 * 8))][(v275 + (v272 * 8))];
              float v279 = v276 * v277;
              float v280 = v278 + v279;
              v269[(v273 + (v270 * 8))][(v275 + (v272 * 8))] = v280;
              if (((v274 + (v271 * 2)) - 63) == 0) {
                float v281 = v269[(v273 + (v270 * 8))][(v275 + (v272 * 8))];
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
  float v282[32][128],
  float v283[16][32],
  hls::stream<float> v284[1][8],
  float v285
) {
  #pragma HLS array_partition variable=v282 cyclic dim=2 factor=8

  float v286[16][128];
  #pragma HLS array_partition variable=v286 cyclic dim=2 factor=8

  loop92: for (int v287 = 0; v287 < 16; v287++) {
    loop93: for (int v288 = 0; v288 < 32; v288++) {
      loop94: for (int v289 = 0; v289 < 16; v289++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop95: for (int v290 = 0; v290 < 8; v290++) {
          loop96: for (int v291 = 0; v291 < 1; v291++) {
            loop97: for (int v292 = 0; v292 < 1; v292++) {
              float v293 = v283[(v289 + v292)][(v288 + v291)];
              float v294 = v282[(v288 + v291)][(v290 + (v287 * 8))];
              if ((v288 + v291) == 0) {
                v286[(v289 + v292)][(v290 + (v287 * 8))] = v285;
              }
              float v295 = v286[(v289 + v292)][(v290 + (v287 * 8))];
              float v296 = v293 * v294;
              float v297 = v295 + v296;
              v286[(v289 + v292)][(v290 + (v287 * 8))] = v297;
              if (((v288 + v291) - 31) == 0) {
                float v298 = v286[(v289 + v292)][(v290 + (v287 * 8))];
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
  float v299[16][32],
  float v300[32][128],
  float v301[128][64],
  float v302[64][128],
  float v303[128][64],
  float v304[64][32],
  float v305[32][16],
  float v306[16][16],
  float v307[16][32],
  float v308[32][128],
  float v309[128][64],
  float v310[64][128],
  float v311[128][64],
  float v312[64][32],
  float v313[32][16],
  float v314[16][16],
  float v315[16][16]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v300 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v301 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v301 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v302 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v302 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v303 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v303 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v304 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v304 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v305 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v307 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v308 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v308 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v309 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v309 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v310 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v310 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v311 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v311 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v312 cyclic dim=1 factor=4

  hls::stream<float> v316[1][1];
	#pragma HLS STREAM variable=v316 depth=256
  hls::stream<float> v317[1][1];
	#pragma HLS STREAM variable=v317 depth=256
  hls::stream<float> v318[1][1];
	#pragma HLS STREAM variable=v318 depth=256
  hls::stream<float> v319[1][1];
	#pragma HLS STREAM variable=v319 depth=256
  hls::stream<float> v320[8][1];
	#pragma HLS STREAM variable=v320 depth=256
  hls::stream<float> v321[1][8];
	#pragma HLS STREAM variable=v321 depth=256
  float v322[128][16];
  #pragma HLS array_partition variable=v322 cyclic dim=1 factor=8

  hls::stream<float> v323[1][8];
	#pragma HLS STREAM variable=v323 depth=256
  hls::stream<float> v324[1][1];
	#pragma HLS STREAM variable=v324 depth=512
  hls::stream<float> v325[8][1];
	#pragma HLS STREAM variable=v325 depth=512
  hls::stream<float> v326[4][8];
	#pragma HLS STREAM variable=v326 depth=512
  hls::stream<float> v327[1][4];
	#pragma HLS STREAM variable=v327 depth=512
  hls::stream<float> v328[2][1];
	#pragma HLS STREAM variable=v328 depth=256
  hls::stream<float> v329[8][2];
	#pragma HLS STREAM variable=v329 depth=256
  hls::stream<float> v330[8][8];
	#pragma HLS STREAM variable=v330 depth=256
  hls::stream<float> v331[1][8];
	#pragma HLS STREAM variable=v331 depth=256
  node16(v300, v299, v331, 0.000000);
  node15(v301, v302, v330, 0.000000);
  node14(v304, v303, v329, 0.000000);
  node13(v305, v306, v328, 0.000000);
  node12(v307, v308, v327, 0.000000);
  node11(v309, v310, v326, 0.000000);
  node10(v312, v311, v325, 0.000000);
  node9(v314, v313, v324, 0.000000);
  node8(v330, v331, v323, 0.000000);
  node7(v328, v329, v322, 0.000000);
  node6(v326, v327, v321, 0.000000);
  node5(v324, v325, v320, 0.000000);
  node4(v322, v323, v319, 0.000000);
  node3(v321, v320, v318, 0.000000);
  node2(v318, v319, v317, 0.000000);
  node1(v317, v316, 1000000.000000);
  node0(v316, v315);
  return ;
}

