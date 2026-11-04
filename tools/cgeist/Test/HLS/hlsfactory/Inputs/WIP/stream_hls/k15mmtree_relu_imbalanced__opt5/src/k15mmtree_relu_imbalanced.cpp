
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
  float v9,
  float v10
) {
  loop4: for (int v11 = 0; v11 < 16; v11++) {
    loop5: for (int v12 = 0; v12 < 16; v12++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v13 = 0; v13 < 1; v13++) {
        loop7: for (int v14 = 0; v14 < 1; v14++) {
          float v15 = v7[v13][v14].read();
          bool v16 = v15 > v9;
          float v17 = v16 ? (float)v15 : (float)v9;
          float v18 = v17 / v10;
          v8[v13][v14].write(v18);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v19[1][1],
  hls::stream<float> v20[1][1],
  hls::stream<float> v21[1][1],
  float v22
) {
  float v23[16][16];
  float v24[16][16];
  float v25[16][16];
  loop8: for (int v26 = 0; v26 < 16; v26++) {
    loop9: for (int v27 = 0; v27 < 16; v27++) {
      loop10: for (int v28 = 0; v28 < 16; v28++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v29 = 0; v29 < 1; v29++) {
          loop12: for (int v30 = 0; v30 < 1; v30++) {
            loop13: for (int v31 = 0; v31 < 1; v31++) {
              if ((v28 + v31) == 0) {
                float v32 = v20[v29][v30].read();
                v25[(v26 + v29)][(v27 + v30)] = v32;
              }
              float v33 = v25[(v26 + v29)][(v27 + v30)];
              if ((v26 + v29) == 0) {
                float v34 = v19[v30][v31].read();
                v24[(v27 + v30)][(v28 + v31)] = v34;
              }
              float v35 = v24[(v27 + v30)][(v28 + v31)];
              if ((v27 + v30) == 0) {
                v23[(v26 + v29)][(v28 + v31)] = v22;
              }
              float v36 = v23[(v26 + v29)][(v28 + v31)];
              float v37 = v33 * v35;
              float v38 = v36 + v37;
              v23[(v26 + v29)][(v28 + v31)] = v38;
              if (((v27 + v30) - 15) == 0) {
                float v39 = v23[(v26 + v29)][(v28 + v31)];
                v21[v29][v31].write(v39);
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
  hls::stream<float> v40[1][1],
  hls::stream<float> v41[1][1],
  float v42
) {
  loop14: for (int v43 = 0; v43 < 16; v43++) {
    loop15: for (int v44 = 0; v44 < 16; v44++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v45 = 0; v45 < 1; v45++) {
        loop17: for (int v46 = 0; v46 < 1; v46++) {
          float v47 = v40[v45][v46].read();
          bool v48 = v47 > v42;
          float v49 = v48 ? (float)v47 : (float)v42;
          v41[v45][v46].write(v49);
        }
      }
    }
  }
  return ;
}

void node4(
  hls::stream<float> v50[1][8],
  hls::stream<float> v51[8][1],
  hls::stream<float> v52[1][1],
  float v53
) {
  float v54[16][16];
  float v55[128][16];
  #pragma HLS array_partition variable=v55 cyclic dim=1 factor=8

  float v56[16][128];
  #pragma HLS array_partition variable=v56 cyclic dim=2 factor=8

  loop18: for (int v57 = 0; v57 < 16; v57++) {
    loop19: for (int v58 = 0; v58 < 16; v58++) {
      loop20: for (int v59 = 0; v59 < 16; v59++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop21: for (int v60 = 0; v60 < 1; v60++) {
          loop22: for (int v61 = 0; v61 < 8; v61++) {
            loop23: for (int v62 = 0; v62 < 1; v62++) {
              if ((v59 + v62) == 0) {
                float v63 = v50[v60][v61].read();
                v56[(v57 + v60)][(v61 + (v58 * 8))] = v63;
              }
              float v64 = v56[(v57 + v60)][(v61 + (v58 * 8))];
              if ((v57 + v60) == 0) {
                float v65 = v51[v61][v62].read();
                v55[(v61 + (v58 * 8))][(v59 + v62)] = v65;
              }
              float v66 = v55[(v61 + (v58 * 8))][(v59 + v62)];
              if ((v61 + (v58 * 8)) == 0) {
                v54[(v57 + v60)][(v59 + v62)] = v53;
              }
              float v67 = v54[(v57 + v60)][(v59 + v62)];
              float v68 = v64 * v66;
              float v69 = v67 + v68;
              v54[(v57 + v60)][(v59 + v62)] = v69;
              if (((v61 + (v58 * 8)) - 127) == 0) {
                float v70 = v54[(v57 + v60)][(v59 + v62)];
                v52[v60][v62].write(v70);
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
  hls::stream<float> v71[1][1],
  hls::stream<float> v72[1][1],
  float v73
) {
  loop24: for (int v74 = 0; v74 < 16; v74++) {
    loop25: for (int v75 = 0; v75 < 16; v75++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop26: for (int v76 = 0; v76 < 1; v76++) {
        loop27: for (int v77 = 0; v77 < 1; v77++) {
          float v78 = v71[v76][v77].read();
          bool v79 = v78 > v73;
          float v80 = v79 ? (float)v78 : (float)v73;
          v72[v76][v77].write(v80);
        }
      }
    }
  }
  return ;
}

void node6(
  hls::stream<float> v81[8][1],
  hls::stream<float> v82[1][8],
  hls::stream<float> v83[1][1],
  float v84
) {
  float v85[16][16];
  float v86[128][16];
  #pragma HLS array_partition variable=v86 cyclic dim=1 factor=8

  float v87[16][128];
  #pragma HLS array_partition variable=v87 cyclic dim=2 factor=8

  loop28: for (int v88 = 0; v88 < 16; v88++) {
    loop29: for (int v89 = 0; v89 < 16; v89++) {
      loop30: for (int v90 = 0; v90 < 16; v90++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop31: for (int v91 = 0; v91 < 1; v91++) {
          loop32: for (int v92 = 0; v92 < 8; v92++) {
            loop33: for (int v93 = 0; v93 < 1; v93++) {
              if ((v90 + v93) == 0) {
                float v94 = v82[v91][v92].read();
                v87[(v88 + v91)][(v92 + (v89 * 8))] = v94;
              }
              float v95 = v87[(v88 + v91)][(v92 + (v89 * 8))];
              if ((v88 + v91) == 0) {
                float v96 = v81[v92][v93].read();
                v86[(v92 + (v89 * 8))][(v90 + v93)] = v96;
              }
              float v97 = v86[(v92 + (v89 * 8))][(v90 + v93)];
              if ((v92 + (v89 * 8)) == 0) {
                v85[(v88 + v91)][(v90 + v93)] = v84;
              }
              float v98 = v85[(v88 + v91)][(v90 + v93)];
              float v99 = v95 * v97;
              float v100 = v98 + v99;
              v85[(v88 + v91)][(v90 + v93)] = v100;
              if (((v92 + (v89 * 8)) - 127) == 0) {
                float v101 = v85[(v88 + v91)][(v90 + v93)];
                v83[v91][v93].write(v101);
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
  hls::stream<float> v102[8][1],
  hls::stream<float> v103[8][1],
  float v104
) {
  loop34: for (int v105 = 0; v105 < 16; v105++) {
    loop35: for (int v106 = 0; v106 < 16; v106++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop36: for (int v107 = 0; v107 < 8; v107++) {
        loop37: for (int v108 = 0; v108 < 1; v108++) {
          float v109 = v102[v107][v108].read();
          bool v110 = v109 > v104;
          float v111 = v110 ? (float)v109 : (float)v104;
          v103[v107][v108].write(v111);
        }
      }
    }
  }
  return ;
}

void node8(
  hls::stream<float> v112[8][2],
  hls::stream<float> v113[2][1],
  hls::stream<float> v114[8][1],
  float v115
) {
  float v116[128][16];
  #pragma HLS array_partition variable=v116 cyclic dim=1 factor=8

  float v117[32][16];
  #pragma HLS array_partition variable=v117 cyclic dim=1 factor=2

  float v118[128][32];
  #pragma HLS array_partition variable=v118 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v118 cyclic dim=2 factor=2

  loop38: for (int v119 = 0; v119 < 16; v119++) {
    loop39: for (int v120 = 0; v120 < 16; v120++) {
      loop40: for (int v121 = 0; v121 < 16; v121++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v122 = 0; v122 < 2; v122++) {
          loop42: for (int v123 = 0; v123 < 8; v123++) {
            loop43: for (int v124 = 0; v124 < 1; v124++) {
              if ((v121 + v124) == 0) {
                float v125 = v112[v123][v122].read();
                v118[(v123 + (v120 * 8))][(v122 + (v119 * 2))] = v125;
              }
              float v126 = v118[(v123 + (v120 * 8))][(v122 + (v119 * 2))];
              if ((v123 + (v120 * 8)) == 0) {
                float v127 = v113[v122][v124].read();
                v117[(v122 + (v119 * 2))][(v121 + v124)] = v127;
              }
              float v128 = v117[(v122 + (v119 * 2))][(v121 + v124)];
              if ((v122 + (v119 * 2)) == 0) {
                v116[(v123 + (v120 * 8))][(v121 + v124)] = v115;
              }
              float v129 = v116[(v123 + (v120 * 8))][(v121 + v124)];
              float v130 = v126 * v128;
              float v131 = v129 + v130;
              v116[(v123 + (v120 * 8))][(v121 + v124)] = v131;
              if (((v122 + (v119 * 2)) - 31) == 0) {
                float v132 = v116[(v123 + (v120 * 8))][(v121 + v124)];
                v114[v123][v124].write(v132);
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
  hls::stream<float> v133[1][8],
  hls::stream<float> v134[1][8],
  float v135
) {
  loop44: for (int v136 = 0; v136 < 16; v136++) {
    loop45: for (int v137 = 0; v137 < 16; v137++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop46: for (int v138 = 0; v138 < 1; v138++) {
        loop47: for (int v139 = 0; v139 < 8; v139++) {
          float v140 = v133[v138][v139].read();
          bool v141 = v140 > v135;
          float v142 = v141 ? (float)v140 : (float)v135;
          v134[v138][v139].write(v142);
        }
      }
    }
  }
  return ;
}

void node10(
  hls::stream<float> v143[8][8],
  hls::stream<float> v144[1][8],
  hls::stream<float> v145[1][8],
  float v146
) {
  float v147[16][128];
  #pragma HLS array_partition variable=v147 cyclic dim=2 factor=8

  float v148[128][128];
  #pragma HLS array_partition variable=v148 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v148 cyclic dim=2 factor=8

  float v149[16][128];
  #pragma HLS array_partition variable=v149 cyclic dim=2 factor=8

  loop48: for (int v150 = 0; v150 < 16; v150++) {
    loop49: for (int v151 = 0; v151 < 16; v151++) {
      loop50: for (int v152 = 0; v152 < 16; v152++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop51: for (int v153 = 0; v153 < 8; v153++) {
          loop52: for (int v154 = 0; v154 < 1; v154++) {
            loop53: for (int v155 = 0; v155 < 8; v155++) {
              if ((v155 + (v152 * 8)) == 0) {
                float v156 = v144[v154][v153].read();
                v149[(v151 + v154)][(v153 + (v150 * 8))] = v156;
              }
              float v157 = v149[(v151 + v154)][(v153 + (v150 * 8))];
              if ((v151 + v154) == 0) {
                float v158 = v143[v153][v155].read();
                v148[(v153 + (v150 * 8))][(v155 + (v152 * 8))] = v158;
              }
              float v159 = v148[(v153 + (v150 * 8))][(v155 + (v152 * 8))];
              if ((v153 + (v150 * 8)) == 0) {
                v147[(v151 + v154)][(v155 + (v152 * 8))] = v146;
              }
              float v160 = v147[(v151 + v154)][(v155 + (v152 * 8))];
              float v161 = v157 * v159;
              float v162 = v160 + v161;
              v147[(v151 + v154)][(v155 + (v152 * 8))] = v162;
              if (((v153 + (v150 * 8)) - 127) == 0) {
                float v163 = v147[(v151 + v154)][(v155 + (v152 * 8))];
                v145[v154][v155].write(v163);
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
  hls::stream<float> v164[8][1],
  hls::stream<float> v165[8][1],
  float v166
) {
  loop54: for (int v167 = 0; v167 < 16; v167++) {
    loop55: for (int v168 = 0; v168 < 16; v168++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop56: for (int v169 = 0; v169 < 8; v169++) {
        loop57: for (int v170 = 0; v170 < 1; v170++) {
          float v171 = v164[v169][v170].read();
          bool v172 = v171 > v166;
          float v173 = v172 ? (float)v171 : (float)v166;
          v165[v169][v170].write(v173);
        }
      }
    }
  }
  return ;
}

void node12(
  hls::stream<float> v174[8][1],
  hls::stream<float> v175[1][1],
  hls::stream<float> v176[8][1],
  float v177
) {
  float v178[128][16];
  #pragma HLS array_partition variable=v178 cyclic dim=1 factor=8

  float v179[32][16];
  float v180[128][32];
  #pragma HLS array_partition variable=v180 cyclic dim=1 factor=8

  loop58: for (int v181 = 0; v181 < 32; v181++) {
    loop59: for (int v182 = 0; v182 < 16; v182++) {
      loop60: for (int v183 = 0; v183 < 16; v183++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop61: for (int v184 = 0; v184 < 1; v184++) {
          loop62: for (int v185 = 0; v185 < 8; v185++) {
            loop63: for (int v186 = 0; v186 < 1; v186++) {
              if ((v183 + v186) == 0) {
                float v187 = v174[v185][v184].read();
                v180[(v185 + (v182 * 8))][(v181 + v184)] = v187;
              }
              float v188 = v180[(v185 + (v182 * 8))][(v181 + v184)];
              if ((v185 + (v182 * 8)) == 0) {
                float v189 = v175[v184][v186].read();
                v179[(v181 + v184)][(v183 + v186)] = v189;
              }
              float v190 = v179[(v181 + v184)][(v183 + v186)];
              if ((v181 + v184) == 0) {
                v178[(v185 + (v182 * 8))][(v183 + v186)] = v177;
              }
              float v191 = v178[(v185 + (v182 * 8))][(v183 + v186)];
              float v192 = v188 * v190;
              float v193 = v191 + v192;
              v178[(v185 + (v182 * 8))][(v183 + v186)] = v193;
              if (((v181 + v184) - 31) == 0) {
                float v194 = v178[(v185 + (v182 * 8))][(v183 + v186)];
                v176[v185][v186].write(v194);
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
  hls::stream<float> v195[1][8],
  hls::stream<float> v196[1][8],
  float v197
) {
  loop64: for (int v198 = 0; v198 < 16; v198++) {
    loop65: for (int v199 = 0; v199 < 16; v199++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop66: for (int v200 = 0; v200 < 1; v200++) {
        loop67: for (int v201 = 0; v201 < 8; v201++) {
          float v202 = v195[v200][v201].read();
          bool v203 = v202 > v197;
          float v204 = v203 ? (float)v202 : (float)v197;
          v196[v200][v201].write(v204);
        }
      }
    }
  }
  return ;
}

void node14(
  hls::stream<float> v205[1][4],
  hls::stream<float> v206[4][8],
  hls::stream<float> v207[1][8],
  float v208
) {
  float v209[16][128];
  #pragma HLS array_partition variable=v209 cyclic dim=2 factor=8

  float v210[128][128];
  #pragma HLS array_partition variable=v210 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v210 cyclic dim=2 factor=8

  float v211[16][128];
  #pragma HLS array_partition variable=v211 cyclic dim=2 factor=4

  loop68: for (int v212 = 0; v212 < 32; v212++) {
    loop69: for (int v213 = 0; v213 < 16; v213++) {
      loop70: for (int v214 = 0; v214 < 16; v214++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v215 = 0; v215 < 4; v215++) {
          loop72: for (int v216 = 0; v216 < 1; v216++) {
            loop73: for (int v217 = 0; v217 < 8; v217++) {
              if ((v217 + (v214 * 8)) == 0) {
                float v218 = v205[v216][v215].read();
                v211[(v213 + v216)][(v215 + (v212 * 4))] = v218;
              }
              float v219 = v211[(v213 + v216)][(v215 + (v212 * 4))];
              if ((v213 + v216) == 0) {
                float v220 = v206[v215][v217].read();
                v210[(v215 + (v212 * 4))][(v217 + (v214 * 8))] = v220;
              }
              float v221 = v210[(v215 + (v212 * 4))][(v217 + (v214 * 8))];
              if ((v215 + (v212 * 4)) == 0) {
                v209[(v213 + v216)][(v217 + (v214 * 8))] = v208;
              }
              float v222 = v209[(v213 + v216)][(v217 + (v214 * 8))];
              float v223 = v219 * v221;
              float v224 = v222 + v223;
              v209[(v213 + v216)][(v217 + (v214 * 8))] = v224;
              if (((v215 + (v212 * 4)) - 127) == 0) {
                float v225 = v209[(v213 + v216)][(v217 + (v214 * 8))];
                v207[v216][v217].write(v225);
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
  hls::stream<float> v226[2][1],
  hls::stream<float> v227[2][1],
  float v228
) {
  loop74: for (int v229 = 0; v229 < 16; v229++) {
    loop75: for (int v230 = 0; v230 < 16; v230++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop76: for (int v231 = 0; v231 < 2; v231++) {
        loop77: for (int v232 = 0; v232 < 1; v232++) {
          float v233 = v226[v231][v232].read();
          bool v234 = v233 > v228;
          float v235 = v234 ? (float)v233 : (float)v228;
          v227[v231][v232].write(v235);
        }
      }
    }
  }
  return ;
}

void node16(
  float v236[16][16],
  float v237[32][16],
  hls::stream<float> v238[2][1],
  float v239
) {
  #pragma HLS array_partition variable=v237 cyclic dim=1 factor=2

  float v240[32][16];
  #pragma HLS array_partition variable=v240 cyclic dim=1 factor=2

  loop78: for (int v241 = 0; v241 < 16; v241++) {
    loop79: for (int v242 = 0; v242 < 16; v242++) {
      loop80: for (int v243 = 0; v243 < 16; v243++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop81: for (int v244 = 0; v244 < 2; v244++) {
          loop82: for (int v245 = 0; v245 < 1; v245++) {
            loop83: for (int v246 = 0; v246 < 1; v246++) {
              float v247 = v237[(v244 + (v241 * 2))][(v242 + v245)];
              float v248 = v236[(v242 + v245)][(v243 + v246)];
              if ((v242 + v245) == 0) {
                v240[(v244 + (v241 * 2))][(v243 + v246)] = v239;
              }
              float v249 = v240[(v244 + (v241 * 2))][(v243 + v246)];
              float v250 = v247 * v248;
              float v251 = v249 + v250;
              v240[(v244 + (v241 * 2))][(v243 + v246)] = v251;
              if (((v242 + v245) - 15) == 0) {
                float v252 = v240[(v244 + (v241 * 2))][(v243 + v246)];
                v238[v244][v246].write(v252);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node17(
  hls::stream<float> v253[8][2],
  hls::stream<float> v254[8][2],
  float v255
) {
  loop84: for (int v256 = 0; v256 < 16; v256++) {
    loop85: for (int v257 = 0; v257 < 16; v257++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop86: for (int v258 = 0; v258 < 2; v258++) {
        loop87: for (int v259 = 0; v259 < 8; v259++) {
          float v260 = v253[v259][v258].read();
          bool v261 = v260 > v255;
          float v262 = v261 ? (float)v260 : (float)v255;
          v254[v259][v258].write(v262);
        }
      }
    }
  }
  return ;
}

void node18(
  float v263[128][64],
  float v264[64][32],
  hls::stream<float> v265[8][2],
  float v266
) {
  #pragma HLS array_partition variable=v263 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v263 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v264 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v264 cyclic dim=2 factor=2

  float v267[128][32];
  #pragma HLS array_partition variable=v267 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v267 cyclic dim=2 factor=2

  loop88: for (int v268 = 0; v268 < 16; v268++) {
    loop89: for (int v269 = 0; v269 < 16; v269++) {
      loop90: for (int v270 = 0; v270 < 16; v270++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop91: for (int v271 = 0; v271 < 2; v271++) {
          loop92: for (int v272 = 0; v272 < 4; v272++) {
            loop93: for (int v273 = 0; v273 < 8; v273++) {
              float v274 = v263[(v273 + (v270 * 8))][(v272 + (v269 * 4))];
              float v275 = v264[(v272 + (v269 * 4))][(v271 + (v268 * 2))];
              if ((v272 + (v269 * 4)) == 0) {
                v267[(v273 + (v270 * 8))][(v271 + (v268 * 2))] = v266;
              }
              float v276 = v267[(v273 + (v270 * 8))][(v271 + (v268 * 2))];
              float v277 = v274 * v275;
              float v278 = v276 + v277;
              v267[(v273 + (v270 * 8))][(v271 + (v268 * 2))] = v278;
              if (((v272 + (v269 * 4)) - 63) == 0) {
                float v279 = v267[(v273 + (v270 * 8))][(v271 + (v268 * 2))];
                v265[v273][v271].write(v279);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node19(
  hls::stream<float> v280[8][8],
  hls::stream<float> v281[8][8],
  float v282
) {
  loop94: for (int v283 = 0; v283 < 16; v283++) {
    loop95: for (int v284 = 0; v284 < 16; v284++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop96: for (int v285 = 0; v285 < 8; v285++) {
        loop97: for (int v286 = 0; v286 < 8; v286++) {
          float v287 = v280[v285][v286].read();
          bool v288 = v287 > v282;
          float v289 = v288 ? (float)v287 : (float)v282;
          v281[v285][v286].write(v289);
        }
      }
    }
  }
  return ;
}

void node20(
  float v290[64][128],
  float v291[128][64],
  hls::stream<float> v292[8][8],
  float v293
) {
  #pragma HLS array_partition variable=v290 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v290 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v291 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v291 cyclic dim=2 factor=2

  float v294[128][128];
  #pragma HLS array_partition variable=v294 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v294 cyclic dim=2 factor=8

  loop98: for (int v295 = 0; v295 < 16; v295++) {
    loop99: for (int v296 = 0; v296 < 32; v296++) {
      loop100: for (int v297 = 0; v297 < 16; v297++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop101: for (int v298 = 0; v298 < 8; v298++) {
          loop102: for (int v299 = 0; v299 < 2; v299++) {
            loop103: for (int v300 = 0; v300 < 8; v300++) {
              float v301 = v291[(v298 + (v295 * 8))][(v299 + (v296 * 2))];
              float v302 = v290[(v299 + (v296 * 2))][(v300 + (v297 * 8))];
              if ((v299 + (v296 * 2)) == 0) {
                v294[(v298 + (v295 * 8))][(v300 + (v297 * 8))] = v293;
              }
              float v303 = v294[(v298 + (v295 * 8))][(v300 + (v297 * 8))];
              float v304 = v301 * v302;
              float v305 = v303 + v304;
              v294[(v298 + (v295 * 8))][(v300 + (v297 * 8))] = v305;
              if (((v299 + (v296 * 2)) - 63) == 0) {
                float v306 = v294[(v298 + (v295 * 8))][(v300 + (v297 * 8))];
                v292[v298][v300].write(v306);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node21(
  hls::stream<float> v307[1][8],
  hls::stream<float> v308[1][8],
  float v309
) {
  loop104: for (int v310 = 0; v310 < 16; v310++) {
    loop105: for (int v311 = 0; v311 < 16; v311++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop106: for (int v312 = 0; v312 < 8; v312++) {
        loop107: for (int v313 = 0; v313 < 1; v313++) {
          float v314 = v307[v313][v312].read();
          bool v315 = v314 > v309;
          float v316 = v315 ? (float)v314 : (float)v309;
          v308[v313][v312].write(v316);
        }
      }
    }
  }
  return ;
}

void node22(
  float v317[32][128],
  float v318[16][32],
  hls::stream<float> v319[1][8],
  float v320
) {
  #pragma HLS array_partition variable=v317 cyclic dim=2 factor=8

  float v321[16][128];
  #pragma HLS array_partition variable=v321 cyclic dim=2 factor=8

  loop108: for (int v322 = 0; v322 < 16; v322++) {
    loop109: for (int v323 = 0; v323 < 32; v323++) {
      loop110: for (int v324 = 0; v324 < 16; v324++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop111: for (int v325 = 0; v325 < 8; v325++) {
          loop112: for (int v326 = 0; v326 < 1; v326++) {
            loop113: for (int v327 = 0; v327 < 1; v327++) {
              float v328 = v318[(v324 + v327)][(v323 + v326)];
              float v329 = v317[(v323 + v326)][(v325 + (v322 * 8))];
              if ((v323 + v326) == 0) {
                v321[(v324 + v327)][(v325 + (v322 * 8))] = v320;
              }
              float v330 = v321[(v324 + v327)][(v325 + (v322 * 8))];
              float v331 = v328 * v329;
              float v332 = v330 + v331;
              v321[(v324 + v327)][(v325 + (v322 * 8))] = v332;
              if (((v323 + v326) - 31) == 0) {
                float v333 = v321[(v324 + v327)][(v325 + (v322 * 8))];
                v319[v327][v325].write(v333);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node23(
  hls::stream<float> v334[1][1],
  hls::stream<float> v335[1][1],
  float v336
) {
  loop114: for (int v337 = 0; v337 < 32; v337++) {
    loop115: for (int v338 = 0; v338 < 16; v338++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop116: for (int v339 = 0; v339 < 1; v339++) {
        loop117: for (int v340 = 0; v340 < 1; v340++) {
          float v341 = v334[v339][v340].read();
          bool v342 = v341 > v336;
          float v343 = v342 ? (float)v341 : (float)v336;
          v335[v339][v340].write(v343);
        }
      }
    }
  }
  return ;
}

void node24(
  float v344[32][16],
  float v345[16][16],
  hls::stream<float> v346[1][1],
  float v347
) {
  float v348[32][16];
  loop118: for (int v349 = 0; v349 < 32; v349++) {
    loop119: for (int v350 = 0; v350 < 16; v350++) {
      loop120: for (int v351 = 0; v351 < 16; v351++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop121: for (int v352 = 0; v352 < 1; v352++) {
          loop122: for (int v353 = 0; v353 < 1; v353++) {
            loop123: for (int v354 = 0; v354 < 1; v354++) {
              float v355 = v344[(v349 + v352)][(v350 + v353)];
              float v356 = v345[(v350 + v353)][(v351 + v354)];
              if ((v350 + v353) == 0) {
                v348[(v349 + v352)][(v351 + v354)] = v347;
              }
              float v357 = v348[(v349 + v352)][(v351 + v354)];
              float v358 = v355 * v356;
              float v359 = v357 + v358;
              v348[(v349 + v352)][(v351 + v354)] = v359;
              if (((v350 + v353) - 15) == 0) {
                float v360 = v348[(v349 + v352)][(v351 + v354)];
                v346[v352][v354].write(v360);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node25(
  hls::stream<float> v361[8][1],
  hls::stream<float> v362[8][1],
  float v363
) {
  loop124: for (int v364 = 0; v364 < 32; v364++) {
    loop125: for (int v365 = 0; v365 < 16; v365++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop126: for (int v366 = 0; v366 < 1; v366++) {
        loop127: for (int v367 = 0; v367 < 8; v367++) {
          float v368 = v361[v367][v366].read();
          bool v369 = v368 > v363;
          float v370 = v369 ? (float)v368 : (float)v363;
          v362[v367][v366].write(v370);
        }
      }
    }
  }
  return ;
}

void node26(
  float v371[128][64],
  float v372[64][32],
  hls::stream<float> v373[8][1],
  float v374
) {
  #pragma HLS array_partition variable=v371 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v371 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v372 cyclic dim=1 factor=4

  float v375[128][32];
  #pragma HLS array_partition variable=v375 cyclic dim=1 factor=8

  loop128: for (int v376 = 0; v376 < 32; v376++) {
    loop129: for (int v377 = 0; v377 < 16; v377++) {
      loop130: for (int v378 = 0; v378 < 16; v378++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop131: for (int v379 = 0; v379 < 1; v379++) {
          loop132: for (int v380 = 0; v380 < 4; v380++) {
            loop133: for (int v381 = 0; v381 < 8; v381++) {
              float v382 = v371[(v381 + (v378 * 8))][(v380 + (v377 * 4))];
              float v383 = v372[(v380 + (v377 * 4))][(v376 + v379)];
              if ((v380 + (v377 * 4)) == 0) {
                v375[(v381 + (v378 * 8))][(v376 + v379)] = v374;
              }
              float v384 = v375[(v381 + (v378 * 8))][(v376 + v379)];
              float v385 = v382 * v383;
              float v386 = v384 + v385;
              v375[(v381 + (v378 * 8))][(v376 + v379)] = v386;
              if (((v380 + (v377 * 4)) - 63) == 0) {
                float v387 = v375[(v381 + (v378 * 8))][(v376 + v379)];
                v373[v381][v379].write(v387);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node27(
  hls::stream<float> v388[4][8],
  hls::stream<float> v389[4][8],
  float v390
) {
  loop134: for (int v391 = 0; v391 < 32; v391++) {
    loop135: for (int v392 = 0; v392 < 16; v392++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop136: for (int v393 = 0; v393 < 4; v393++) {
        loop137: for (int v394 = 0; v394 < 8; v394++) {
          float v395 = v388[v393][v394].read();
          bool v396 = v395 > v390;
          float v397 = v396 ? (float)v395 : (float)v390;
          v389[v393][v394].write(v397);
        }
      }
    }
  }
  return ;
}

void node28(
  float v398[64][128],
  float v399[128][64],
  hls::stream<float> v400[4][8],
  float v401
) {
  #pragma HLS array_partition variable=v398 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v398 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v399 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v399 cyclic dim=2 factor=4

  float v402[128][128];
  #pragma HLS array_partition variable=v402 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v402 cyclic dim=2 factor=8

  loop138: for (int v403 = 0; v403 < 32; v403++) {
    loop139: for (int v404 = 0; v404 < 16; v404++) {
      loop140: for (int v405 = 0; v405 < 16; v405++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop141: for (int v406 = 0; v406 < 4; v406++) {
          loop142: for (int v407 = 0; v407 < 4; v407++) {
            loop143: for (int v408 = 0; v408 < 8; v408++) {
              float v409 = v399[(v406 + (v403 * 4))][(v407 + (v404 * 4))];
              float v410 = v398[(v407 + (v404 * 4))][(v408 + (v405 * 8))];
              if ((v407 + (v404 * 4)) == 0) {
                v402[(v406 + (v403 * 4))][(v408 + (v405 * 8))] = v401;
              }
              float v411 = v402[(v406 + (v403 * 4))][(v408 + (v405 * 8))];
              float v412 = v409 * v410;
              float v413 = v411 + v412;
              v402[(v406 + (v403 * 4))][(v408 + (v405 * 8))] = v413;
              if (((v407 + (v404 * 4)) - 63) == 0) {
                float v414 = v402[(v406 + (v403 * 4))][(v408 + (v405 * 8))];
                v400[v406][v408].write(v414);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node29(
  hls::stream<float> v415[1][4],
  hls::stream<float> v416[1][4],
  float v417
) {
  loop144: for (int v418 = 0; v418 < 32; v418++) {
    loop145: for (int v419 = 0; v419 < 16; v419++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop146: for (int v420 = 0; v420 < 4; v420++) {
        loop147: for (int v421 = 0; v421 < 1; v421++) {
          float v422 = v415[v421][v420].read();
          bool v423 = v422 > v417;
          float v424 = v423 ? (float)v422 : (float)v417;
          v416[v421][v420].write(v424);
        }
      }
    }
  }
  return ;
}

void node30(
  float v425[16][32],
  float v426[32][128],
  hls::stream<float> v427[1][4],
  float v428
) {
  #pragma HLS array_partition variable=v425 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v426 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v426 cyclic dim=2 factor=4

  float v429[16][128];
  #pragma HLS array_partition variable=v429 cyclic dim=2 factor=4

  loop148: for (int v430 = 0; v430 < 32; v430++) {
    loop149: for (int v431 = 0; v431 < 16; v431++) {
      loop150: for (int v432 = 0; v432 < 16; v432++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop151: for (int v433 = 0; v433 < 4; v433++) {
          loop152: for (int v434 = 0; v434 < 2; v434++) {
            loop153: for (int v435 = 0; v435 < 1; v435++) {
              float v436 = v425[(v432 + v435)][(v434 + (v431 * 2))];
              float v437 = v426[(v434 + (v431 * 2))][(v433 + (v430 * 4))];
              if ((v434 + (v431 * 2)) == 0) {
                v429[(v432 + v435)][(v433 + (v430 * 4))] = v428;
              }
              float v438 = v429[(v432 + v435)][(v433 + (v430 * 4))];
              float v439 = v436 * v437;
              float v440 = v438 + v439;
              v429[(v432 + v435)][(v433 + (v430 * 4))] = v440;
              if (((v434 + (v431 * 2)) - 31) == 0) {
                float v441 = v429[(v432 + v435)][(v433 + (v430 * 4))];
                v427[v435][v433].write(v441);
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
  float v442[16][32],
  float v443[32][128],
  float v444[128][64],
  float v445[64][128],
  float v446[128][64],
  float v447[64][32],
  float v448[32][16],
  float v449[16][16],
  float v450[16][32],
  float v451[32][128],
  float v452[128][64],
  float v453[64][128],
  float v454[128][64],
  float v455[64][32],
  float v456[32][16],
  float v457[16][16],
  float v458[16][16]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v442 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v443 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v443 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v444 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v444 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v445 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v445 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v446 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v446 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v447 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v451 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v452 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v452 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v453 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v453 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v454 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v454 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v455 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v455 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v456 cyclic dim=1 factor=2

  hls::stream<float> v459[1][1];
	#pragma HLS STREAM variable=v459 depth=256
  hls::stream<float> v460[1][1];
	#pragma HLS STREAM variable=v460 depth=256
  hls::stream<float> v461[1][1];
	#pragma HLS STREAM variable=v461 depth=256
  hls::stream<float> v462[1][1];
	#pragma HLS STREAM variable=v462 depth=256
  hls::stream<float> v463[1][1];
	#pragma HLS STREAM variable=v463 depth=256
  hls::stream<float> v464[1][1];
	#pragma HLS STREAM variable=v464 depth=256
  hls::stream<float> v465[8][1];
	#pragma HLS STREAM variable=v465 depth=256
  hls::stream<float> v466[8][1];
	#pragma HLS STREAM variable=v466 depth=256
  hls::stream<float> v467[1][8];
	#pragma HLS STREAM variable=v467 depth=256
  hls::stream<float> v468[1][8];
	#pragma HLS STREAM variable=v468 depth=256
  hls::stream<float> v469[8][1];
	#pragma HLS STREAM variable=v469 depth=256
  hls::stream<float> v470[8][1];
	#pragma HLS STREAM variable=v470 depth=256
  hls::stream<float> v471[1][8];
	#pragma HLS STREAM variable=v471 depth=256
  hls::stream<float> v472[1][8];
	#pragma HLS STREAM variable=v472 depth=256
  hls::stream<float> v473[2][1];
	#pragma HLS STREAM variable=v473 depth=256
  hls::stream<float> v474[2][1];
	#pragma HLS STREAM variable=v474 depth=256
  hls::stream<float> v475[8][2];
	#pragma HLS STREAM variable=v475 depth=256
  hls::stream<float> v476[8][2];
	#pragma HLS STREAM variable=v476 depth=256
  hls::stream<float> v477[8][8];
	#pragma HLS STREAM variable=v477 depth=256
  hls::stream<float> v478[8][8];
	#pragma HLS STREAM variable=v478 depth=256
  hls::stream<float> v479[1][8];
	#pragma HLS STREAM variable=v479 depth=256
  hls::stream<float> v480[1][8];
	#pragma HLS STREAM variable=v480 depth=256
  hls::stream<float> v481[1][1];
	#pragma HLS STREAM variable=v481 depth=512
  hls::stream<float> v482[1][1];
	#pragma HLS STREAM variable=v482 depth=512
  hls::stream<float> v483[8][1];
	#pragma HLS STREAM variable=v483 depth=512
  hls::stream<float> v484[8][1];
	#pragma HLS STREAM variable=v484 depth=512
  hls::stream<float> v485[4][8];
	#pragma HLS STREAM variable=v485 depth=512
  hls::stream<float> v486[4][8];
	#pragma HLS STREAM variable=v486 depth=512
  hls::stream<float> v487[1][4];
	#pragma HLS STREAM variable=v487 depth=512
  hls::stream<float> v488[1][4];
	#pragma HLS STREAM variable=v488 depth=512
  node30(v442, v443, v488, 0.000000);
  node29(v488, v487, 0.000000);
  node28(v445, v444, v486, 0.000000);
  node27(v486, v485, 0.000000);
  node26(v446, v447, v484, 0.000000);
  node25(v484, v483, 0.000000);
  node24(v448, v449, v482, 0.000000);
  node23(v482, v481, 0.000000);
  node22(v451, v450, v480, 0.000000);
  node21(v480, v479, 0.000000);
  node20(v453, v452, v478, 0.000000);
  node19(v478, v477, 0.000000);
  node18(v454, v455, v476, 0.000000);
  node17(v476, v475, 0.000000);
  node16(v457, v456, v474, 0.000000);
  node15(v474, v473, 0.000000);
  node14(v487, v485, v472, 0.000000);
  node13(v472, v471, 0.000000);
  node12(v483, v481, v470, 0.000000);
  node11(v470, v469, 0.000000);
  node10(v477, v479, v468, 0.000000);
  node9(v468, v467, 0.000000);
  node8(v475, v473, v466, 0.000000);
  node7(v466, v465, 0.000000);
  node6(v469, v471, v464, 0.000000);
  node5(v464, v463, 0.000000);
  node4(v467, v465, v462, 0.000000);
  node3(v462, v461, 0.000000);
  node2(v461, v463, v460, 0.000000);
  node1(v460, v459, 0.000000, 100000000.000000);
  node0(v459, v458);
  return ;
}

