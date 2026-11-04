
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
  float v1[1][64][128]
) {
  loop0: for (int v2 = 0; v2 < 64; v2++) {
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
  loop2: for (int v8 = 0; v8 < 64; v8++) {
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
  float v13[128][128],
  hls::stream<float> &v14,
  hls::stream<float> &v15,
  float v16
) {
  float v17[64][128];
  float v18[128][128];
  float v19[64][128];
  loop4: for (int v20 = 0; v20 < 64; v20++) {
    loop5: for (int v21 = 0; v21 < 128; v21++) {
      loop6: for (int v22 = 0; v22 < 128; v22++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v21 == 0) {
          float v23 = v14.read();
          v19[v20][v22] = v23;
        }
        float v24 = v19[v20][v22];
        if (v20 == 0) {
          float v25 = v13[v22][v21];
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
        if ((v22 - 127) == 0) {
          float v30 = v17[v20][v21];
          v15.write(v30);
        }
      }
    }
  }
  return ;
}

void node3(
  float v31[128][128],
  float v32[128][128]
) {
  loop7: for (int v33 = 0; v33 < 128; v33++) {
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
  float v36[64][8][16],
  hls::stream<float> &v37
) {
  loop9: for (int v38 = 0; v38 < 64; v38++) {
    loop10: for (int v39 = 0; v39 < 128; v39++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v40 = v36[v38][(v39 / 16)][(v39 % 16)];
      v37.write(v40);
    }
  }
  return ;
}

void node5(
  float v41[8][64][16],
  float v42[64][8][16]
) {
  loop11: for (int v43 = 0; v43 < 8; v43++) {
    loop12: for (int v44 = 0; v44 < 64; v44++) {
      loop13: for (int v45 = 0; v45 < 16; v45++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v46 = v41[v43][v44][v45];
        v42[v44][v43][v45] = v46;
      }
    }
  }
  return ;
}

void node6(
  float v47[8][64][16],
  hls::stream<float> &v48,
  float v49[8][64][16],
  float v50
) {
  float v51[8][64][16];
  float v52[8][64][16];
  float v53[8][64][64];
  loop14: for (int v54 = 0; v54 < 8; v54++) {
    loop15: for (int v55 = 0; v55 < 64; v55++) {
      loop16: for (int v56 = 0; v56 < 16; v56++) {
        loop17: for (int v57 = 0; v57 < 64; v57++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          if (v56 == 0) {
            float v58 = v48.read();
            v53[v54][v55][v57] = v58;
          }
          float v59 = v53[v54][v55][v57];
          if (v55 == 0) {
            float v60 = v47[v54][v57][v56];
            v52[v54][v57][v56] = v60;
          }
          float v61 = v52[v54][v57][v56];
          if (v57 == 0) {
            v51[v54][v55][v56] = v50;
          }
          float v62 = v51[v54][v55][v56];
          float v63 = v59 * v61;
          float v64 = v62 + v63;
          v51[v54][v55][v56] = v64;
          if ((v57 - 63) == 0) {
            float v65 = v51[v54][v55][v56];
            v49[v54][v55][v56] = v65;
          }
        }
      }
    }
  }
  return ;
}

void node7(
  float v66[64][128],
  float v67[8][64][16]
) {
  loop18: for (int v68 = 0; v68 < 8; v68++) {
    loop19: for (int v69 = 0; v69 < 64; v69++) {
      loop20: for (int v70 = 0; v70 < 16; v70++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v71 = v66[v69][((v68 * 16) + v70)];
        v67[v68][v69][v70] = v71;
      }
    }
  }
  return ;
}

void node8(
  hls::stream<float> &v72,
  hls::stream<float> &v73,
  hls::stream<float> &v74
) {
  float v75[8][64];
  loop21: for (int v76 = 0; v76 < 8; v76++) {
    loop22: for (int v77 = 0; v77 < 64; v77++) {
      loop23: for (int v78 = 0; v78 < 64; v78++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v79 = v72.read();
        if (v78 == 0) {
          float v80 = v73.read();
          v75[v76][v77] = v80;
        }
        float v81 = v75[v76][v77];
        float v82 = v79 / v81;
        v74.write(v82);
      }
    }
  }
  return ;
}

void node9(
  hls::stream<float> &v83,
  hls::stream<float> &v84,
  float v85
) {
  float v86[8][64];
  loop24: for (int v87 = 0; v87 < 8; v87++) {
    loop25: for (int v88 = 0; v88 < 64; v88++) {
      loop26: for (int v89 = 0; v89 < 64; v89++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v90 = v83.read();
        if (v89 == 0) {
          v86[v87][v88] = v85;
        }
        float v91 = v86[v87][v88];
        float v92 = v90 + v91;
        v86[v87][v88] = v92;
        if ((v89 - 63) == 0) {
          float v93 = v86[v87][v88];
          v84.write(v93);
        }
      }
    }
  }
  return ;
}

void node10(
  hls::stream<float> &v94,
  hls::stream<float> &v95,
  hls::stream<float> &v96,
  hls::stream<float> &v97
) {
  float v98[8][64];
  loop27: for (int v99 = 0; v99 < 8; v99++) {
    loop28: for (int v100 = 0; v100 < 64; v100++) {
      loop29: for (int v101 = 0; v101 < 64; v101++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v102 = v94.read();
        if (v101 == 0) {
          float v103 = v95.read();
          v98[v99][v100] = v103;
        }
        float v104 = v98[v99][v100];
        float v105 = v102 - v104;
        float v106 = exp(v105);
        v96.write(v106);
        v97.write(v106);
      }
    }
  }
  return ;
}

void node11(
  hls::stream<float> &v107,
  hls::stream<float> &v108,
  float v109
) {
  float v110[8][64];
  loop30: for (int v111 = 0; v111 < 8; v111++) {
    loop31: for (int v112 = 0; v112 < 64; v112++) {
      loop32: for (int v113 = 0; v113 < 64; v113++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v114 = v107.read();
        if (v113 == 0) {
          v110[v111][v112] = v109;
        }
        float v115 = v110[v111][v112];
        float v116 = fmax(v114, v115);
        v110[v111][v112] = v116;
        if ((v113 - 63) == 0) {
          float v117 = v110[v111][v112];
          v108.write(v117);
        }
      }
    }
  }
  return ;
}

void node12(
  hls::stream<float> &v118,
  hls::stream<float> &v119,
  hls::stream<float> &v120,
  float v121
) {
  loop33: for (int v122 = 0; v122 < 8; v122++) {
    loop34: for (int v123 = 0; v123 < 64; v123++) {
      loop35: for (int v124 = 0; v124 < 64; v124++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v125 = v118.read();
        float v126 = v125 / v121;
        v119.write(v126);
        v120.write(v126);
      }
    }
  }
  return ;
}

void node13(
  hls::stream<float> &v127,
  float v128[8][16][64],
  hls::stream<float> &v129,
  float v130
) {
  float v131[8][64][64];
  float v132[8][16][64];
  float v133[8][64][16];
  loop36: for (int v134 = 0; v134 < 8; v134++) {
    loop37: for (int v135 = 0; v135 < 64; v135++) {
      loop38: for (int v136 = 0; v136 < 64; v136++) {
        loop39: for (int v137 = 0; v137 < 16; v137++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          if (v136 == 0) {
            float v138 = v127.read();
            v133[v134][v135][v137] = v138;
          }
          float v139 = v133[v134][v135][v137];
          if (v135 == 0) {
            float v140 = v128[v134][v137][v136];
            v132[v134][v137][v136] = v140;
          }
          float v141 = v132[v134][v137][v136];
          if (v137 == 0) {
            v131[v134][v135][v136] = v130;
          }
          float v142 = v131[v134][v135][v136];
          float v143 = v139 * v141;
          float v144 = v142 + v143;
          v131[v134][v135][v136] = v144;
          if ((v137 - 15) == 0) {
            float v145 = v131[v134][v135][v136];
            v129.write(v145);
          }
        }
      }
    }
  }
  return ;
}

void node14(
  float v146[64][128],
  float v147[8][16][64]
) {
  loop40: for (int v148 = 0; v148 < 8; v148++) {
    loop41: for (int v149 = 0; v149 < 16; v149++) {
      loop42: for (int v150 = 0; v150 < 64; v150++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v151 = v146[v150][((v148 * 16) + v149)];
        v147[v148][v149][v150] = v151;
      }
    }
  }
  return ;
}

void node15(
  float v152[64][128],
  hls::stream<float> &v153
) {
  loop43: for (int v154 = 0; v154 < 8; v154++) {
    loop44: for (int v155 = 0; v155 < 64; v155++) {
      loop45: for (int v156 = 0; v156 < 16; v156++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v157 = v152[v155][((v154 * 16) + v156)];
        v153.write(v157);
      }
    }
  }
  return ;
}

void node16(
  hls::stream<float> &v158,
  float v159[128],
  float v160[64][128]
) {
  loop46: for (int v161 = 0; v161 < 64; v161++) {
    loop47: for (int v162 = 0; v162 < 128; v162++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v163 = v158.read();
      float v164 = v159[v162];
      float v165 = v163 + v164;
      v160[v161][v162] = v165;
    }
  }
  return ;
}

void node17(
  hls::stream<float> &v166,
  float v167[128][128],
  hls::stream<float> &v168,
  float v169
) {
  float v170[64][128];
  float v171[128][128];
  float v172[64][128];
  loop48: for (int v173 = 0; v173 < 64; v173++) {
    loop49: for (int v174 = 0; v174 < 128; v174++) {
      loop50: for (int v175 = 0; v175 < 128; v175++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v174 == 0) {
          float v176 = v166.read();
          v172[v173][v175] = v176;
        }
        float v177 = v172[v173][v175];
        if (v173 == 0) {
          float v178 = v167[v175][v174];
          v171[v175][v174] = v178;
        }
        float v179 = v171[v175][v174];
        if (v175 == 0) {
          v170[v173][v174] = v169;
        }
        float v180 = v170[v173][v174];
        float v181 = v177 * v179;
        float v182 = v180 + v181;
        v170[v173][v174] = v182;
        if ((v175 - 127) == 0) {
          float v183 = v170[v173][v174];
          v168.write(v183);
        }
      }
    }
  }
  return ;
}

void node18(
  float v184[128][128],
  float v185[128][128]
) {
  loop51: for (int v186 = 0; v186 < 128; v186++) {
    loop52: for (int v187 = 0; v187 < 128; v187++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v188 = v184[v187][v186];
      v185[v186][v187] = v188;
    }
  }
  return ;
}

void node19(
  hls::stream<float> &v189,
  float v190[128],
  float v191[64][128]
) {
  loop53: for (int v192 = 0; v192 < 64; v192++) {
    loop54: for (int v193 = 0; v193 < 128; v193++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v194 = v189.read();
      float v195 = v190[v193];
      float v196 = v194 + v195;
      v191[v192][v193] = v196;
    }
  }
  return ;
}

void node20(
  float v197[128][128],
  hls::stream<float> &v198,
  hls::stream<float> &v199,
  float v200
) {
  float v201[64][128];
  float v202[128][128];
  float v203[64][128];
  loop55: for (int v204 = 0; v204 < 64; v204++) {
    loop56: for (int v205 = 0; v205 < 128; v205++) {
      loop57: for (int v206 = 0; v206 < 128; v206++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v205 == 0) {
          float v207 = v198.read();
          v203[v204][v206] = v207;
        }
        float v208 = v203[v204][v206];
        if (v204 == 0) {
          float v209 = v197[v206][v205];
          v202[v206][v205] = v209;
        }
        float v210 = v202[v206][v205];
        if (v206 == 0) {
          v201[v204][v205] = v200;
        }
        float v211 = v201[v204][v205];
        float v212 = v208 * v210;
        float v213 = v211 + v212;
        v201[v204][v205] = v213;
        if ((v206 - 127) == 0) {
          float v214 = v201[v204][v205];
          v199.write(v214);
        }
      }
    }
  }
  return ;
}

void node21(
  float v215[128][128],
  float v216[128][128]
) {
  loop58: for (int v217 = 0; v217 < 128; v217++) {
    loop59: for (int v218 = 0; v218 < 128; v218++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v219 = v215[v218][v217];
      v216[v217][v218] = v219;
    }
  }
  return ;
}

void node22(
  hls::stream<float> &v220,
  float v221[128],
  float v222[64][128]
) {
  loop60: for (int v223 = 0; v223 < 64; v223++) {
    loop61: for (int v224 = 0; v224 < 128; v224++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v225 = v220.read();
      float v226 = v221[v224];
      float v227 = v225 + v226;
      v222[v223][v224] = v227;
    }
  }
  return ;
}

void node23(
  hls::stream<float> &v228,
  float v229[128][128],
  hls::stream<float> &v230,
  float v231
) {
  float v232[64][128];
  float v233[128][128];
  float v234[64][128];
  loop62: for (int v235 = 0; v235 < 64; v235++) {
    loop63: for (int v236 = 0; v236 < 128; v236++) {
      loop64: for (int v237 = 0; v237 < 128; v237++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v236 == 0) {
          float v238 = v228.read();
          v234[v235][v237] = v238;
        }
        float v239 = v234[v235][v237];
        if (v235 == 0) {
          float v240 = v229[v237][v236];
          v233[v237][v236] = v240;
        }
        float v241 = v233[v237][v236];
        if (v237 == 0) {
          v232[v235][v236] = v231;
        }
        float v242 = v232[v235][v236];
        float v243 = v239 * v241;
        float v244 = v242 + v243;
        v232[v235][v236] = v244;
        if ((v237 - 127) == 0) {
          float v245 = v232[v235][v236];
          v230.write(v245);
        }
      }
    }
  }
  return ;
}

void node24(
  float v246[128][128],
  float v247[128][128]
) {
  loop65: for (int v248 = 0; v248 < 128; v248++) {
    loop66: for (int v249 = 0; v249 < 128; v249++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v250 = v246[v249][v248];
      v247[v248][v249] = v250;
    }
  }
  return ;
}

void node25(
  float v251[1][64][128],
  hls::stream<float> &v252,
  hls::stream<float> &v253,
  hls::stream<float> &v254
) {
  loop67: for (int v255 = 0; v255 < 64; v255++) {
    loop68: for (int v256 = 0; v256 < 128; v256++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v257 = v251[0][v255][v256];
      v252.write(v257);
      v253.write(v257);
      v254.write(v257);
    }
  }
  return ;
}

void forward(
  float v258[1][64][128],
  float v259[128],
  float v260[128][128],
  float v261[128],
  float v262[128][128],
  float v263[128],
  float v264[128][128],
  float v265[128],
  float v266[128][128],
  float v267[1][64][128]
) {
	#pragma HLS DATAFLOW
  hls::stream<float> v268("fifo_0");
	#pragma HLS STREAM variable=v268 depth=8192
  hls::stream<float> v269("fifo_1");
	#pragma HLS STREAM variable=v269 depth=8192
  float v270[128][128];
  hls::stream<float> v271("fifo_2");
	#pragma HLS STREAM variable=v271 depth=8192
  float v272[64][8][16];
  float v273[8][64][16];
  float v274[8][64][16];
  hls::stream<float> v275("fifo_3");
	#pragma HLS STREAM variable=v275 depth=32768
  hls::stream<float> v276("fifo_4");
	#pragma HLS STREAM variable=v276 depth=512
  hls::stream<float> v277("fifo_5");
	#pragma HLS STREAM variable=v277 depth=32768
  hls::stream<float> v278("fifo_6");
	#pragma HLS STREAM variable=v278 depth=32768
  hls::stream<float> v279("fifo_7");
	#pragma HLS STREAM variable=v279 depth=512
  hls::stream<float> v280("fifo_8");
	#pragma HLS STREAM variable=v280 depth=32768
  hls::stream<float> v281("fifo_9");
	#pragma HLS STREAM variable=v281 depth=32768
  hls::stream<float> v282("fifo_10");
	#pragma HLS STREAM variable=v282 depth=32768
  float v283[8][16][64];
  hls::stream<float> v284("fifo_11");
	#pragma HLS STREAM variable=v284 depth=8192
  float v285[64][128];
  hls::stream<float> v286("fifo_12");
	#pragma HLS STREAM variable=v286 depth=8192
  float v287[128][128];
  float v288[64][128];
  hls::stream<float> v289("fifo_13");
	#pragma HLS STREAM variable=v289 depth=8192
  float v290[128][128];
  float v291[64][128];
  hls::stream<float> v292("fifo_14");
	#pragma HLS STREAM variable=v292 depth=8192
  float v293[128][128];
  hls::stream<float> v294("fifo_15");
	#pragma HLS STREAM variable=v294 depth=8192
  hls::stream<float> v295("fifo_16");
	#pragma HLS STREAM variable=v295 depth=8192
  hls::stream<float> v296("fifo_17");
	#pragma HLS STREAM variable=v296 depth=8192
  node25(v258, v296, v295, v294);
  node24(v260, v293);
  node23(v294, v293, v292, 0.000000);
  node22(v292, v259, v291);
  node21(v262, v290);
  node20(v290, v295, v289, 0.000000);
  node19(v289, v261, v288);
  node18(v264, v287);
  node17(v296, v287, v286, 0.000000);
  node16(v286, v263, v285);
  node15(v291, v284);
  node14(v288, v283);
  node13(v284, v283, v282, 0.000000);
  node12(v282, v281, v280, 4.000000);
  node11(v281, v279, -INFINITY);
  node10(v280, v279, v278, v277);
  node9(v278, v276, 0.000000);
  node8(v277, v276, v275);
  node7(v285, v274);
  node6(v274, v275, v273, 0.000000);
  node5(v273, v272);
  node4(v272, v271);
  node3(v266, v270);
  node2(v270, v271, v269, 0.000000);
  node1(v269, v265, v268);
  node0(v268, v267);
  return ;
}

