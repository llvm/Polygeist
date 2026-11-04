
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
          float v6 = v0[v4][v5].read();
          v1[(v2 + v4)][(v3 + v5)] = v6;
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
  float v16[16][16],
  hls::stream<float> v17[1][1],
  hls::stream<float> v18[1][1],
  float v19
) {
  float v20[16][16];
  float v21[16][16];
  loop8: for (int v22 = 0; v22 < 16; v22++) {
    loop9: for (int v23 = 0; v23 < 16; v23++) {
      loop10: for (int v24 = 0; v24 < 16; v24++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v25 = 0; v25 < 1; v25++) {
          loop12: for (int v26 = 0; v26 < 1; v26++) {
            loop13: for (int v27 = 0; v27 < 1; v27++) {
              if ((v24 + v27) == 0) {
                float v28 = v17[v25][v26].read();
                v21[(v22 + v25)][(v23 + v26)] = v28;
              }
              float v29 = v21[(v22 + v25)][(v23 + v26)];
              float v30 = v16[(v23 + v26)][(v24 + v27)];
              if ((v23 + v26) == 0) {
                v20[(v22 + v25)][(v24 + v27)] = v19;
              }
              float v31 = v20[(v22 + v25)][(v24 + v27)];
              float v32 = v29 * v30;
              float v33 = v31 + v32;
              v20[(v22 + v25)][(v24 + v27)] = v33;
              if (((v23 + v26) - 15) == 0) {
                float v34 = v20[(v22 + v25)][(v24 + v27)];
                v18[v25][v27].write(v34);
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
  hls::stream<float> v35[1][2],
  float v36[32][16],
  hls::stream<float> v37[1][1],
  float v38
) {
  #pragma HLS array_partition variable=v36 cyclic dim=1 factor=2

  float v39[16][16];
  float v40[16][32];
  #pragma HLS array_partition variable=v40 cyclic dim=2 factor=2

  loop14: for (int v41 = 0; v41 < 16; v41++) {
    loop15: for (int v42 = 0; v42 < 16; v42++) {
      loop16: for (int v43 = 0; v43 < 16; v43++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop17: for (int v44 = 0; v44 < 1; v44++) {
          loop18: for (int v45 = 0; v45 < 2; v45++) {
            loop19: for (int v46 = 0; v46 < 1; v46++) {
              if ((v43 + v46) == 0) {
                float v47 = v35[v44][v45].read();
                v40[(v41 + v44)][(v45 + (v42 * 2))] = v47;
              }
              float v48 = v40[(v41 + v44)][(v45 + (v42 * 2))];
              float v49 = v36[(v45 + (v42 * 2))][(v43 + v46)];
              if ((v45 + (v42 * 2)) == 0) {
                v39[(v41 + v44)][(v43 + v46)] = v38;
              }
              float v50 = v39[(v41 + v44)][(v43 + v46)];
              float v51 = v48 * v49;
              float v52 = v50 + v51;
              v39[(v41 + v44)][(v43 + v46)] = v52;
              if (((v45 + (v42 * 2)) - 31) == 0) {
                float v53 = v39[(v41 + v44)][(v43 + v46)];
                v37[v44][v46].write(v53);
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
  hls::stream<float> v54[1][4],
  float v55[64][32],
  hls::stream<float> v56[1][2],
  float v57
) {
  #pragma HLS array_partition variable=v55 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v55 cyclic dim=2 factor=2

  float v58[16][32];
  #pragma HLS array_partition variable=v58 cyclic dim=2 factor=2

  float v59[16][64];
  #pragma HLS array_partition variable=v59 cyclic dim=2 factor=4

  loop20: for (int v60 = 0; v60 < 16; v60++) {
    loop21: for (int v61 = 0; v61 < 16; v61++) {
      loop22: for (int v62 = 0; v62 < 16; v62++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop23: for (int v63 = 0; v63 < 1; v63++) {
          loop24: for (int v64 = 0; v64 < 4; v64++) {
            loop25: for (int v65 = 0; v65 < 2; v65++) {
              if ((v65 + (v62 * 2)) == 0) {
                float v66 = v54[v63][v64].read();
                v59[(v60 + v63)][(v64 + (v61 * 4))] = v66;
              }
              float v67 = v59[(v60 + v63)][(v64 + (v61 * 4))];
              float v68 = v55[(v64 + (v61 * 4))][(v65 + (v62 * 2))];
              if ((v64 + (v61 * 4)) == 0) {
                v58[(v60 + v63)][(v65 + (v62 * 2))] = v57;
              }
              float v69 = v58[(v60 + v63)][(v65 + (v62 * 2))];
              float v70 = v67 * v68;
              float v71 = v69 + v70;
              v58[(v60 + v63)][(v65 + (v62 * 2))] = v71;
              if (((v64 + (v61 * 4)) - 63) == 0) {
                float v72 = v58[(v60 + v63)][(v65 + (v62 * 2))];
                v56[v63][v65].write(v72);
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
  float v73[128][64],
  hls::stream<float> v74[1][8],
  hls::stream<float> v75[1][4],
  float v76
) {
  #pragma HLS array_partition variable=v73 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v73 cyclic dim=2 factor=4

  float v77[16][64];
  #pragma HLS array_partition variable=v77 cyclic dim=2 factor=4

  float v78[16][128];
  #pragma HLS array_partition variable=v78 cyclic dim=2 factor=8

  loop26: for (int v79 = 0; v79 < 16; v79++) {
    loop27: for (int v80 = 0; v80 < 16; v80++) {
      loop28: for (int v81 = 0; v81 < 16; v81++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop29: for (int v82 = 0; v82 < 1; v82++) {
          loop30: for (int v83 = 0; v83 < 8; v83++) {
            loop31: for (int v84 = 0; v84 < 4; v84++) {
              if ((v84 + (v81 * 4)) == 0) {
                float v85 = v74[v82][v83].read();
                v78[(v79 + v82)][(v83 + (v80 * 8))] = v85;
              }
              float v86 = v78[(v79 + v82)][(v83 + (v80 * 8))];
              float v87 = v73[(v83 + (v80 * 8))][(v84 + (v81 * 4))];
              if ((v83 + (v80 * 8)) == 0) {
                v77[(v79 + v82)][(v84 + (v81 * 4))] = v76;
              }
              float v88 = v77[(v79 + v82)][(v84 + (v81 * 4))];
              float v89 = v86 * v87;
              float v90 = v88 + v89;
              v77[(v79 + v82)][(v84 + (v81 * 4))] = v90;
              if (((v83 + (v80 * 8)) - 127) == 0) {
                float v91 = v77[(v79 + v82)][(v84 + (v81 * 4))];
                v75[v82][v84].write(v91);
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
  hls::stream<float> v92[1][4],
  float v93[64][128],
  hls::stream<float> v94[1][8],
  float v95
) {
  #pragma HLS array_partition variable=v93 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v93 cyclic dim=2 factor=8

  float v96[16][128];
  #pragma HLS array_partition variable=v96 cyclic dim=2 factor=8

  float v97[16][64];
  #pragma HLS array_partition variable=v97 cyclic dim=2 factor=4

  loop32: for (int v98 = 0; v98 < 16; v98++) {
    loop33: for (int v99 = 0; v99 < 16; v99++) {
      loop34: for (int v100 = 0; v100 < 16; v100++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop35: for (int v101 = 0; v101 < 1; v101++) {
          loop36: for (int v102 = 0; v102 < 4; v102++) {
            loop37: for (int v103 = 0; v103 < 8; v103++) {
              if ((v103 + (v100 * 8)) == 0) {
                float v104 = v92[v101][v102].read();
                v97[(v98 + v101)][(v102 + (v99 * 4))] = v104;
              }
              float v105 = v97[(v98 + v101)][(v102 + (v99 * 4))];
              float v106 = v93[(v102 + (v99 * 4))][(v103 + (v100 * 8))];
              if ((v102 + (v99 * 4)) == 0) {
                v96[(v98 + v101)][(v103 + (v100 * 8))] = v95;
              }
              float v107 = v96[(v98 + v101)][(v103 + (v100 * 8))];
              float v108 = v105 * v106;
              float v109 = v107 + v108;
              v96[(v98 + v101)][(v103 + (v100 * 8))] = v109;
              if (((v102 + (v99 * 4)) - 63) == 0) {
                float v110 = v96[(v98 + v101)][(v103 + (v100 * 8))];
                v94[v101][v103].write(v110);
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
  float v111[128][64],
  hls::stream<float> v112[1][8],
  hls::stream<float> v113[1][4],
  float v114
) {
  #pragma HLS array_partition variable=v111 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v111 cyclic dim=2 factor=4

  float v115[16][64];
  #pragma HLS array_partition variable=v115 cyclic dim=2 factor=4

  float v116[16][128];
  #pragma HLS array_partition variable=v116 cyclic dim=2 factor=8

  loop38: for (int v117 = 0; v117 < 16; v117++) {
    loop39: for (int v118 = 0; v118 < 16; v118++) {
      loop40: for (int v119 = 0; v119 < 16; v119++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v120 = 0; v120 < 1; v120++) {
          loop42: for (int v121 = 0; v121 < 8; v121++) {
            loop43: for (int v122 = 0; v122 < 4; v122++) {
              if ((v122 + (v119 * 4)) == 0) {
                float v123 = v112[v120][v121].read();
                v116[(v117 + v120)][(v121 + (v118 * 8))] = v123;
              }
              float v124 = v116[(v117 + v120)][(v121 + (v118 * 8))];
              float v125 = v111[(v121 + (v118 * 8))][(v122 + (v119 * 4))];
              if ((v121 + (v118 * 8)) == 0) {
                v115[(v117 + v120)][(v122 + (v119 * 4))] = v114;
              }
              float v126 = v115[(v117 + v120)][(v122 + (v119 * 4))];
              float v127 = v124 * v125;
              float v128 = v126 + v127;
              v115[(v117 + v120)][(v122 + (v119 * 4))] = v128;
              if (((v121 + (v118 * 8)) - 127) == 0) {
                float v129 = v115[(v117 + v120)][(v122 + (v119 * 4))];
                v113[v120][v122].write(v129);
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
  float v130[32][128],
  hls::stream<float> v131[1][2],
  hls::stream<float> v132[1][8],
  float v133
) {
  #pragma HLS array_partition variable=v130 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v130 cyclic dim=2 factor=8

  float v134[16][128];
  #pragma HLS array_partition variable=v134 cyclic dim=2 factor=8

  float v135[16][32];
  #pragma HLS array_partition variable=v135 cyclic dim=2 factor=2

  loop44: for (int v136 = 0; v136 < 16; v136++) {
    loop45: for (int v137 = 0; v137 < 16; v137++) {
      loop46: for (int v138 = 0; v138 < 16; v138++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop47: for (int v139 = 0; v139 < 1; v139++) {
          loop48: for (int v140 = 0; v140 < 2; v140++) {
            loop49: for (int v141 = 0; v141 < 8; v141++) {
              if ((v141 + (v138 * 8)) == 0) {
                float v142 = v131[v139][v140].read();
                v135[(v136 + v139)][(v140 + (v137 * 2))] = v142;
              }
              float v143 = v135[(v136 + v139)][(v140 + (v137 * 2))];
              float v144 = v130[(v140 + (v137 * 2))][(v141 + (v138 * 8))];
              if ((v140 + (v137 * 2)) == 0) {
                v134[(v136 + v139)][(v141 + (v138 * 8))] = v133;
              }
              float v145 = v134[(v136 + v139)][(v141 + (v138 * 8))];
              float v146 = v143 * v144;
              float v147 = v145 + v146;
              v134[(v136 + v139)][(v141 + (v138 * 8))] = v147;
              if (((v140 + (v137 * 2)) - 31) == 0) {
                float v148 = v134[(v136 + v139)][(v141 + (v138 * 8))];
                v132[v139][v141].write(v148);
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
  float v149[16][32],
  hls::stream<float> v150[1][1],
  hls::stream<float> v151[1][2],
  float v152
) {
  #pragma HLS array_partition variable=v149 cyclic dim=2 factor=2

  float v153[16][32];
  #pragma HLS array_partition variable=v153 cyclic dim=2 factor=2

  float v154[16][16];
  loop50: for (int v155 = 0; v155 < 16; v155++) {
    loop51: for (int v156 = 0; v156 < 16; v156++) {
      loop52: for (int v157 = 0; v157 < 16; v157++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop53: for (int v158 = 0; v158 < 1; v158++) {
          loop54: for (int v159 = 0; v159 < 1; v159++) {
            loop55: for (int v160 = 0; v160 < 2; v160++) {
              if ((v160 + (v157 * 2)) == 0) {
                float v161 = v150[v158][v159].read();
                v154[(v155 + v158)][(v156 + v159)] = v161;
              }
              float v162 = v154[(v155 + v158)][(v156 + v159)];
              float v163 = v149[(v156 + v159)][(v160 + (v157 * 2))];
              if ((v156 + v159) == 0) {
                v153[(v155 + v158)][(v160 + (v157 * 2))] = v152;
              }
              float v164 = v153[(v155 + v158)][(v160 + (v157 * 2))];
              float v165 = v162 * v163;
              float v166 = v164 + v165;
              v153[(v155 + v158)][(v160 + (v157 * 2))] = v166;
              if (((v156 + v159) - 15) == 0) {
                float v167 = v153[(v155 + v158)][(v160 + (v157 * 2))];
                v151[v158][v160].write(v167);
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
  hls::stream<float> v168[1][1],
  float v169[16][16],
  hls::stream<float> v170[1][1],
  float v171
) {
  float v172[16][16];
  float v173[16][16];
  loop56: for (int v174 = 0; v174 < 16; v174++) {
    loop57: for (int v175 = 0; v175 < 16; v175++) {
      loop58: for (int v176 = 0; v176 < 16; v176++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop59: for (int v177 = 0; v177 < 1; v177++) {
          loop60: for (int v178 = 0; v178 < 1; v178++) {
            loop61: for (int v179 = 0; v179 < 1; v179++) {
              if ((v176 + v179) == 0) {
                float v180 = v168[v177][v178].read();
                v173[(v174 + v177)][(v175 + v178)] = v180;
              }
              float v181 = v173[(v174 + v177)][(v175 + v178)];
              float v182 = v169[(v175 + v178)][(v176 + v179)];
              if ((v175 + v178) == 0) {
                v172[(v174 + v177)][(v176 + v179)] = v171;
              }
              float v183 = v172[(v174 + v177)][(v176 + v179)];
              float v184 = v181 * v182;
              float v185 = v183 + v184;
              v172[(v174 + v177)][(v176 + v179)] = v185;
              if (((v175 + v178) - 15) == 0) {
                float v186 = v172[(v174 + v177)][(v176 + v179)];
                v170[v177][v179].write(v186);
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
  hls::stream<float> v187[1][2],
  float v188[32][16],
  hls::stream<float> v189[1][1],
  float v190
) {
  #pragma HLS array_partition variable=v188 cyclic dim=1 factor=2

  float v191[16][16];
  float v192[16][32];
  #pragma HLS array_partition variable=v192 cyclic dim=2 factor=2

  loop62: for (int v193 = 0; v193 < 16; v193++) {
    loop63: for (int v194 = 0; v194 < 16; v194++) {
      loop64: for (int v195 = 0; v195 < 16; v195++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop65: for (int v196 = 0; v196 < 1; v196++) {
          loop66: for (int v197 = 0; v197 < 2; v197++) {
            loop67: for (int v198 = 0; v198 < 1; v198++) {
              if ((v195 + v198) == 0) {
                float v199 = v187[v196][v197].read();
                v192[(v193 + v196)][(v197 + (v194 * 2))] = v199;
              }
              float v200 = v192[(v193 + v196)][(v197 + (v194 * 2))];
              float v201 = v188[(v197 + (v194 * 2))][(v195 + v198)];
              if ((v197 + (v194 * 2)) == 0) {
                v191[(v193 + v196)][(v195 + v198)] = v190;
              }
              float v202 = v191[(v193 + v196)][(v195 + v198)];
              float v203 = v200 * v201;
              float v204 = v202 + v203;
              v191[(v193 + v196)][(v195 + v198)] = v204;
              if (((v197 + (v194 * 2)) - 31) == 0) {
                float v205 = v191[(v193 + v196)][(v195 + v198)];
                v189[v196][v198].write(v205);
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
  float v206[64][32],
  hls::stream<float> v207[1][4],
  hls::stream<float> v208[1][2],
  float v209
) {
  #pragma HLS array_partition variable=v206 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v206 cyclic dim=2 factor=2

  float v210[16][32];
  #pragma HLS array_partition variable=v210 cyclic dim=2 factor=2

  float v211[16][64];
  #pragma HLS array_partition variable=v211 cyclic dim=2 factor=4

  loop68: for (int v212 = 0; v212 < 16; v212++) {
    loop69: for (int v213 = 0; v213 < 16; v213++) {
      loop70: for (int v214 = 0; v214 < 16; v214++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v215 = 0; v215 < 1; v215++) {
          loop72: for (int v216 = 0; v216 < 4; v216++) {
            loop73: for (int v217 = 0; v217 < 2; v217++) {
              if ((v217 + (v214 * 2)) == 0) {
                float v218 = v207[v215][v216].read();
                v211[(v212 + v215)][(v216 + (v213 * 4))] = v218;
              }
              float v219 = v211[(v212 + v215)][(v216 + (v213 * 4))];
              float v220 = v206[(v216 + (v213 * 4))][(v217 + (v214 * 2))];
              if ((v216 + (v213 * 4)) == 0) {
                v210[(v212 + v215)][(v217 + (v214 * 2))] = v209;
              }
              float v221 = v210[(v212 + v215)][(v217 + (v214 * 2))];
              float v222 = v219 * v220;
              float v223 = v221 + v222;
              v210[(v212 + v215)][(v217 + (v214 * 2))] = v223;
              if (((v216 + (v213 * 4)) - 63) == 0) {
                float v224 = v210[(v212 + v215)][(v217 + (v214 * 2))];
                v208[v215][v217].write(v224);
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
  float v225[128][64],
  hls::stream<float> v226[1][8],
  hls::stream<float> v227[1][4],
  float v228
) {
  #pragma HLS array_partition variable=v225 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v225 cyclic dim=2 factor=4

  float v229[16][64];
  #pragma HLS array_partition variable=v229 cyclic dim=2 factor=4

  float v230[16][128];
  #pragma HLS array_partition variable=v230 cyclic dim=2 factor=8

  loop74: for (int v231 = 0; v231 < 16; v231++) {
    loop75: for (int v232 = 0; v232 < 16; v232++) {
      loop76: for (int v233 = 0; v233 < 16; v233++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop77: for (int v234 = 0; v234 < 1; v234++) {
          loop78: for (int v235 = 0; v235 < 8; v235++) {
            loop79: for (int v236 = 0; v236 < 4; v236++) {
              if ((v236 + (v233 * 4)) == 0) {
                float v237 = v226[v234][v235].read();
                v230[(v231 + v234)][(v235 + (v232 * 8))] = v237;
              }
              float v238 = v230[(v231 + v234)][(v235 + (v232 * 8))];
              float v239 = v225[(v235 + (v232 * 8))][(v236 + (v233 * 4))];
              if ((v235 + (v232 * 8)) == 0) {
                v229[(v231 + v234)][(v236 + (v233 * 4))] = v228;
              }
              float v240 = v229[(v231 + v234)][(v236 + (v233 * 4))];
              float v241 = v238 * v239;
              float v242 = v240 + v241;
              v229[(v231 + v234)][(v236 + (v233 * 4))] = v242;
              if (((v235 + (v232 * 8)) - 127) == 0) {
                float v243 = v229[(v231 + v234)][(v236 + (v233 * 4))];
                v227[v234][v236].write(v243);
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
  float v244[64][128],
  hls::stream<float> v245[1][4],
  hls::stream<float> v246[1][8],
  float v247
) {
  #pragma HLS array_partition variable=v244 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v244 cyclic dim=2 factor=8

  float v248[16][128];
  #pragma HLS array_partition variable=v248 cyclic dim=2 factor=8

  float v249[16][64];
  #pragma HLS array_partition variable=v249 cyclic dim=2 factor=4

  loop80: for (int v250 = 0; v250 < 16; v250++) {
    loop81: for (int v251 = 0; v251 < 16; v251++) {
      loop82: for (int v252 = 0; v252 < 16; v252++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop83: for (int v253 = 0; v253 < 1; v253++) {
          loop84: for (int v254 = 0; v254 < 4; v254++) {
            loop85: for (int v255 = 0; v255 < 8; v255++) {
              if ((v255 + (v252 * 8)) == 0) {
                float v256 = v245[v253][v254].read();
                v249[(v250 + v253)][(v254 + (v251 * 4))] = v256;
              }
              float v257 = v249[(v250 + v253)][(v254 + (v251 * 4))];
              float v258 = v244[(v254 + (v251 * 4))][(v255 + (v252 * 8))];
              if ((v254 + (v251 * 4)) == 0) {
                v248[(v250 + v253)][(v255 + (v252 * 8))] = v247;
              }
              float v259 = v248[(v250 + v253)][(v255 + (v252 * 8))];
              float v260 = v257 * v258;
              float v261 = v259 + v260;
              v248[(v250 + v253)][(v255 + (v252 * 8))] = v261;
              if (((v254 + (v251 * 4)) - 63) == 0) {
                float v262 = v248[(v250 + v253)][(v255 + (v252 * 8))];
                v246[v253][v255].write(v262);
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
  hls::stream<float> v263[1][8],
  float v264[128][64],
  hls::stream<float> v265[1][4],
  float v266
) {
  #pragma HLS array_partition variable=v264 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v264 cyclic dim=2 factor=4

  float v267[16][64];
  #pragma HLS array_partition variable=v267 cyclic dim=2 factor=4

  float v268[16][128];
  #pragma HLS array_partition variable=v268 cyclic dim=2 factor=8

  loop86: for (int v269 = 0; v269 < 16; v269++) {
    loop87: for (int v270 = 0; v270 < 16; v270++) {
      loop88: for (int v271 = 0; v271 < 16; v271++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop89: for (int v272 = 0; v272 < 1; v272++) {
          loop90: for (int v273 = 0; v273 < 8; v273++) {
            loop91: for (int v274 = 0; v274 < 4; v274++) {
              if ((v274 + (v271 * 4)) == 0) {
                float v275 = v263[v272][v273].read();
                v268[(v269 + v272)][(v273 + (v270 * 8))] = v275;
              }
              float v276 = v268[(v269 + v272)][(v273 + (v270 * 8))];
              float v277 = v264[(v273 + (v270 * 8))][(v274 + (v271 * 4))];
              if ((v273 + (v270 * 8)) == 0) {
                v267[(v269 + v272)][(v274 + (v271 * 4))] = v266;
              }
              float v278 = v267[(v269 + v272)][(v274 + (v271 * 4))];
              float v279 = v276 * v277;
              float v280 = v278 + v279;
              v267[(v269 + v272)][(v274 + (v271 * 4))] = v280;
              if (((v273 + (v270 * 8)) - 127) == 0) {
                float v281 = v267[(v269 + v272)][(v274 + (v271 * 4))];
                v265[v272][v274].write(v281);
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
  #pragma HLS array_partition variable=v282 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v282 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v283 cyclic dim=2 factor=2

  float v286[16][128];
  #pragma HLS array_partition variable=v286 cyclic dim=2 factor=8

  loop92: for (int v287 = 0; v287 < 16; v287++) {
    loop93: for (int v288 = 0; v288 < 16; v288++) {
      loop94: for (int v289 = 0; v289 < 16; v289++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop95: for (int v290 = 0; v290 < 1; v290++) {
          loop96: for (int v291 = 0; v291 < 2; v291++) {
            loop97: for (int v292 = 0; v292 < 8; v292++) {
              float v293 = v283[(v287 + v290)][(v291 + (v288 * 2))];
              float v294 = v282[(v291 + (v288 * 2))][(v292 + (v289 * 8))];
              if ((v291 + (v288 * 2)) == 0) {
                v286[(v287 + v290)][(v292 + (v289 * 8))] = v285;
              }
              float v295 = v286[(v287 + v290)][(v292 + (v289 * 8))];
              float v296 = v293 * v294;
              float v297 = v295 + v296;
              v286[(v287 + v290)][(v292 + (v289 * 8))] = v297;
              if (((v291 + (v288 * 2)) - 31) == 0) {
                float v298 = v286[(v287 + v290)][(v292 + (v289 * 8))];
                v284[v290][v292].write(v298);
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
  #pragma HLS array_partition variable=v299 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v300 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v300 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v301 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v301 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v302 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v302 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v303 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v303 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v304 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v304 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v305 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v307 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v308 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v308 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v309 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v309 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v310 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v310 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v311 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v311 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v312 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v312 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v313 cyclic dim=1 factor=2

  hls::stream<float> v316[1][1];
	#pragma HLS STREAM variable=v316 depth=256
  hls::stream<float> v317[1][1];
	#pragma HLS STREAM variable=v317 depth=256
  hls::stream<float> v318[1][1];
	#pragma HLS STREAM variable=v318 depth=256
  hls::stream<float> v319[1][2];
	#pragma HLS STREAM variable=v319 depth=256
  hls::stream<float> v320[1][4];
	#pragma HLS STREAM variable=v320 depth=256
  hls::stream<float> v321[1][8];
	#pragma HLS STREAM variable=v321 depth=256
  hls::stream<float> v322[1][4];
	#pragma HLS STREAM variable=v322 depth=256
  hls::stream<float> v323[1][8];
	#pragma HLS STREAM variable=v323 depth=256
  hls::stream<float> v324[1][2];
	#pragma HLS STREAM variable=v324 depth=256
  hls::stream<float> v325[1][1];
	#pragma HLS STREAM variable=v325 depth=256
  hls::stream<float> v326[1][1];
	#pragma HLS STREAM variable=v326 depth=256
  hls::stream<float> v327[1][2];
	#pragma HLS STREAM variable=v327 depth=256
  hls::stream<float> v328[1][4];
	#pragma HLS STREAM variable=v328 depth=256
  hls::stream<float> v329[1][8];
	#pragma HLS STREAM variable=v329 depth=256
  hls::stream<float> v330[1][4];
	#pragma HLS STREAM variable=v330 depth=256
  hls::stream<float> v331[1][8];
	#pragma HLS STREAM variable=v331 depth=256
  node16(v300, v299, v331, 0.000000);
  node15(v331, v301, v330, 0.000000);
  node14(v302, v330, v329, 0.000000);
  node13(v303, v329, v328, 0.000000);
  node12(v304, v328, v327, 0.000000);
  node11(v327, v305, v326, 0.000000);
  node10(v326, v306, v325, 0.000000);
  node9(v307, v325, v324, 0.000000);
  node8(v308, v324, v323, 0.000000);
  node7(v309, v323, v322, 0.000000);
  node6(v322, v310, v321, 0.000000);
  node5(v311, v321, v320, 0.000000);
  node4(v320, v312, v319, 0.000000);
  node3(v319, v313, v318, 0.000000);
  node2(v314, v318, v317, 0.000000);
  node1(v317, v316, 1000000.000000);
  node0(v316, v315);
  return ;
}

