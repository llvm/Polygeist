
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
  float v19[16][16],
  hls::stream<float> v20[1][1],
  hls::stream<float> v21[1][1],
  float v22
) {
  float v23[16][16];
  float v24[16][16];
  loop8: for (int v25 = 0; v25 < 16; v25++) {
    loop9: for (int v26 = 0; v26 < 16; v26++) {
      loop10: for (int v27 = 0; v27 < 16; v27++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v28 = 0; v28 < 1; v28++) {
          loop12: for (int v29 = 0; v29 < 1; v29++) {
            loop13: for (int v30 = 0; v30 < 1; v30++) {
              if ((v27 + v30) == 0) {
                float v31 = v20[v28][v29].read();
                v24[(v25 + v28)][(v26 + v29)] = v31;
              }
              float v32 = v24[(v25 + v28)][(v26 + v29)];
              float v33 = v19[(v26 + v29)][(v27 + v30)];
              if ((v26 + v29) == 0) {
                v23[(v25 + v28)][(v27 + v30)] = v22;
              }
              float v34 = v23[(v25 + v28)][(v27 + v30)];
              float v35 = v32 * v33;
              float v36 = v34 + v35;
              v23[(v25 + v28)][(v27 + v30)] = v36;
              if (((v26 + v29) - 15) == 0) {
                float v37 = v23[(v25 + v28)][(v27 + v30)];
                v21[v28][v30].write(v37);
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
  hls::stream<float> v38[1][1],
  hls::stream<float> v39[1][1],
  float v40
) {
  loop14: for (int v41 = 0; v41 < 16; v41++) {
    loop15: for (int v42 = 0; v42 < 16; v42++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v43 = 0; v43 < 1; v43++) {
        loop17: for (int v44 = 0; v44 < 1; v44++) {
          float v45 = v38[v43][v44].read();
          bool v46 = v45 > v40;
          float v47 = v46 ? (float)v45 : (float)v40;
          v39[v43][v44].write(v47);
        }
      }
    }
  }
  return ;
}

void node4(
  hls::stream<float> v48[1][2],
  float v49[32][16],
  hls::stream<float> v50[1][1],
  float v51
) {
  #pragma HLS array_partition variable=v49 cyclic dim=1 factor=2

  float v52[16][16];
  float v53[16][32];
  #pragma HLS array_partition variable=v53 cyclic dim=2 factor=2

  loop18: for (int v54 = 0; v54 < 16; v54++) {
    loop19: for (int v55 = 0; v55 < 16; v55++) {
      loop20: for (int v56 = 0; v56 < 16; v56++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop21: for (int v57 = 0; v57 < 1; v57++) {
          loop22: for (int v58 = 0; v58 < 2; v58++) {
            loop23: for (int v59 = 0; v59 < 1; v59++) {
              if ((v56 + v59) == 0) {
                float v60 = v48[v57][v58].read();
                v53[(v54 + v57)][(v58 + (v55 * 2))] = v60;
              }
              float v61 = v53[(v54 + v57)][(v58 + (v55 * 2))];
              float v62 = v49[(v58 + (v55 * 2))][(v56 + v59)];
              if ((v58 + (v55 * 2)) == 0) {
                v52[(v54 + v57)][(v56 + v59)] = v51;
              }
              float v63 = v52[(v54 + v57)][(v56 + v59)];
              float v64 = v61 * v62;
              float v65 = v63 + v64;
              v52[(v54 + v57)][(v56 + v59)] = v65;
              if (((v58 + (v55 * 2)) - 31) == 0) {
                float v66 = v52[(v54 + v57)][(v56 + v59)];
                v50[v57][v59].write(v66);
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
  hls::stream<float> v67[1][2],
  hls::stream<float> v68[1][2],
  float v69
) {
  loop24: for (int v70 = 0; v70 < 16; v70++) {
    loop25: for (int v71 = 0; v71 < 16; v71++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop26: for (int v72 = 0; v72 < 1; v72++) {
        loop27: for (int v73 = 0; v73 < 2; v73++) {
          float v74 = v67[v72][v73].read();
          bool v75 = v74 > v69;
          float v76 = v75 ? (float)v74 : (float)v69;
          v68[v72][v73].write(v76);
        }
      }
    }
  }
  return ;
}

void node6(
  float v77[64][32],
  hls::stream<float> v78[1][4],
  hls::stream<float> v79[1][2],
  float v80
) {
  #pragma HLS array_partition variable=v77 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v77 cyclic dim=2 factor=2

  float v81[16][32];
  #pragma HLS array_partition variable=v81 cyclic dim=2 factor=2

  float v82[16][64];
  #pragma HLS array_partition variable=v82 cyclic dim=2 factor=4

  loop28: for (int v83 = 0; v83 < 16; v83++) {
    loop29: for (int v84 = 0; v84 < 16; v84++) {
      loop30: for (int v85 = 0; v85 < 16; v85++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop31: for (int v86 = 0; v86 < 1; v86++) {
          loop32: for (int v87 = 0; v87 < 4; v87++) {
            loop33: for (int v88 = 0; v88 < 2; v88++) {
              if ((v88 + (v85 * 2)) == 0) {
                float v89 = v78[v86][v87].read();
                v82[(v83 + v86)][(v87 + (v84 * 4))] = v89;
              }
              float v90 = v82[(v83 + v86)][(v87 + (v84 * 4))];
              float v91 = v77[(v87 + (v84 * 4))][(v88 + (v85 * 2))];
              if ((v87 + (v84 * 4)) == 0) {
                v81[(v83 + v86)][(v88 + (v85 * 2))] = v80;
              }
              float v92 = v81[(v83 + v86)][(v88 + (v85 * 2))];
              float v93 = v90 * v91;
              float v94 = v92 + v93;
              v81[(v83 + v86)][(v88 + (v85 * 2))] = v94;
              if (((v87 + (v84 * 4)) - 63) == 0) {
                float v95 = v81[(v83 + v86)][(v88 + (v85 * 2))];
                v79[v86][v88].write(v95);
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
  hls::stream<float> v96[1][4],
  hls::stream<float> v97[1][4],
  float v98
) {
  loop34: for (int v99 = 0; v99 < 16; v99++) {
    loop35: for (int v100 = 0; v100 < 16; v100++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop36: for (int v101 = 0; v101 < 1; v101++) {
        loop37: for (int v102 = 0; v102 < 4; v102++) {
          float v103 = v96[v101][v102].read();
          bool v104 = v103 > v98;
          float v105 = v104 ? (float)v103 : (float)v98;
          v97[v101][v102].write(v105);
        }
      }
    }
  }
  return ;
}

void node8(
  hls::stream<float> v106[1][8],
  float v107[128][64],
  hls::stream<float> v108[1][4],
  float v109
) {
  #pragma HLS array_partition variable=v107 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v107 cyclic dim=2 factor=4

  float v110[16][64];
  #pragma HLS array_partition variable=v110 cyclic dim=2 factor=4

  float v111[16][128];
  #pragma HLS array_partition variable=v111 cyclic dim=2 factor=8

  loop38: for (int v112 = 0; v112 < 16; v112++) {
    loop39: for (int v113 = 0; v113 < 16; v113++) {
      loop40: for (int v114 = 0; v114 < 16; v114++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v115 = 0; v115 < 1; v115++) {
          loop42: for (int v116 = 0; v116 < 8; v116++) {
            loop43: for (int v117 = 0; v117 < 4; v117++) {
              if ((v117 + (v114 * 4)) == 0) {
                float v118 = v106[v115][v116].read();
                v111[(v112 + v115)][(v116 + (v113 * 8))] = v118;
              }
              float v119 = v111[(v112 + v115)][(v116 + (v113 * 8))];
              float v120 = v107[(v116 + (v113 * 8))][(v117 + (v114 * 4))];
              if ((v116 + (v113 * 8)) == 0) {
                v110[(v112 + v115)][(v117 + (v114 * 4))] = v109;
              }
              float v121 = v110[(v112 + v115)][(v117 + (v114 * 4))];
              float v122 = v119 * v120;
              float v123 = v121 + v122;
              v110[(v112 + v115)][(v117 + (v114 * 4))] = v123;
              if (((v116 + (v113 * 8)) - 127) == 0) {
                float v124 = v110[(v112 + v115)][(v117 + (v114 * 4))];
                v108[v115][v117].write(v124);
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
  hls::stream<float> v125[1][8],
  hls::stream<float> v126[1][8],
  float v127
) {
  loop44: for (int v128 = 0; v128 < 16; v128++) {
    loop45: for (int v129 = 0; v129 < 16; v129++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop46: for (int v130 = 0; v130 < 1; v130++) {
        loop47: for (int v131 = 0; v131 < 8; v131++) {
          float v132 = v125[v130][v131].read();
          bool v133 = v132 > v127;
          float v134 = v133 ? (float)v132 : (float)v127;
          v126[v130][v131].write(v134);
        }
      }
    }
  }
  return ;
}

void node10(
  hls::stream<float> v135[1][4],
  float v136[64][128],
  hls::stream<float> v137[1][8],
  float v138
) {
  #pragma HLS array_partition variable=v136 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v136 cyclic dim=2 factor=8

  float v139[16][128];
  #pragma HLS array_partition variable=v139 cyclic dim=2 factor=8

  float v140[16][64];
  #pragma HLS array_partition variable=v140 cyclic dim=2 factor=4

  loop48: for (int v141 = 0; v141 < 16; v141++) {
    loop49: for (int v142 = 0; v142 < 16; v142++) {
      loop50: for (int v143 = 0; v143 < 16; v143++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop51: for (int v144 = 0; v144 < 1; v144++) {
          loop52: for (int v145 = 0; v145 < 4; v145++) {
            loop53: for (int v146 = 0; v146 < 8; v146++) {
              if ((v146 + (v143 * 8)) == 0) {
                float v147 = v135[v144][v145].read();
                v140[(v141 + v144)][(v145 + (v142 * 4))] = v147;
              }
              float v148 = v140[(v141 + v144)][(v145 + (v142 * 4))];
              float v149 = v136[(v145 + (v142 * 4))][(v146 + (v143 * 8))];
              if ((v145 + (v142 * 4)) == 0) {
                v139[(v141 + v144)][(v146 + (v143 * 8))] = v138;
              }
              float v150 = v139[(v141 + v144)][(v146 + (v143 * 8))];
              float v151 = v148 * v149;
              float v152 = v150 + v151;
              v139[(v141 + v144)][(v146 + (v143 * 8))] = v152;
              if (((v145 + (v142 * 4)) - 63) == 0) {
                float v153 = v139[(v141 + v144)][(v146 + (v143 * 8))];
                v137[v144][v146].write(v153);
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
  hls::stream<float> v154[1][4],
  hls::stream<float> v155[1][4],
  float v156
) {
  loop54: for (int v157 = 0; v157 < 16; v157++) {
    loop55: for (int v158 = 0; v158 < 16; v158++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop56: for (int v159 = 0; v159 < 1; v159++) {
        loop57: for (int v160 = 0; v160 < 4; v160++) {
          float v161 = v154[v159][v160].read();
          bool v162 = v161 > v156;
          float v163 = v162 ? (float)v161 : (float)v156;
          v155[v159][v160].write(v163);
        }
      }
    }
  }
  return ;
}

void node12(
  float v164[128][64],
  hls::stream<float> v165[1][8],
  hls::stream<float> v166[1][4],
  float v167
) {
  #pragma HLS array_partition variable=v164 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v164 cyclic dim=2 factor=4

  float v168[16][64];
  #pragma HLS array_partition variable=v168 cyclic dim=2 factor=4

  float v169[16][128];
  #pragma HLS array_partition variable=v169 cyclic dim=2 factor=8

  loop58: for (int v170 = 0; v170 < 16; v170++) {
    loop59: for (int v171 = 0; v171 < 16; v171++) {
      loop60: for (int v172 = 0; v172 < 16; v172++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop61: for (int v173 = 0; v173 < 1; v173++) {
          loop62: for (int v174 = 0; v174 < 8; v174++) {
            loop63: for (int v175 = 0; v175 < 4; v175++) {
              if ((v175 + (v172 * 4)) == 0) {
                float v176 = v165[v173][v174].read();
                v169[(v170 + v173)][(v174 + (v171 * 8))] = v176;
              }
              float v177 = v169[(v170 + v173)][(v174 + (v171 * 8))];
              float v178 = v164[(v174 + (v171 * 8))][(v175 + (v172 * 4))];
              if ((v174 + (v171 * 8)) == 0) {
                v168[(v170 + v173)][(v175 + (v172 * 4))] = v167;
              }
              float v179 = v168[(v170 + v173)][(v175 + (v172 * 4))];
              float v180 = v177 * v178;
              float v181 = v179 + v180;
              v168[(v170 + v173)][(v175 + (v172 * 4))] = v181;
              if (((v174 + (v171 * 8)) - 127) == 0) {
                float v182 = v168[(v170 + v173)][(v175 + (v172 * 4))];
                v166[v173][v175].write(v182);
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
  hls::stream<float> v183[1][8],
  hls::stream<float> v184[1][8],
  float v185
) {
  loop64: for (int v186 = 0; v186 < 16; v186++) {
    loop65: for (int v187 = 0; v187 < 16; v187++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop66: for (int v188 = 0; v188 < 1; v188++) {
        loop67: for (int v189 = 0; v189 < 8; v189++) {
          float v190 = v183[v188][v189].read();
          bool v191 = v190 > v185;
          float v192 = v191 ? (float)v190 : (float)v185;
          v184[v188][v189].write(v192);
        }
      }
    }
  }
  return ;
}

void node14(
  float v193[32][128],
  hls::stream<float> v194[1][2],
  hls::stream<float> v195[1][8],
  float v196
) {
  #pragma HLS array_partition variable=v193 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v193 cyclic dim=2 factor=8

  float v197[16][128];
  #pragma HLS array_partition variable=v197 cyclic dim=2 factor=8

  float v198[16][32];
  #pragma HLS array_partition variable=v198 cyclic dim=2 factor=2

  loop68: for (int v199 = 0; v199 < 16; v199++) {
    loop69: for (int v200 = 0; v200 < 16; v200++) {
      loop70: for (int v201 = 0; v201 < 16; v201++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v202 = 0; v202 < 1; v202++) {
          loop72: for (int v203 = 0; v203 < 2; v203++) {
            loop73: for (int v204 = 0; v204 < 8; v204++) {
              if ((v204 + (v201 * 8)) == 0) {
                float v205 = v194[v202][v203].read();
                v198[(v199 + v202)][(v203 + (v200 * 2))] = v205;
              }
              float v206 = v198[(v199 + v202)][(v203 + (v200 * 2))];
              float v207 = v193[(v203 + (v200 * 2))][(v204 + (v201 * 8))];
              if ((v203 + (v200 * 2)) == 0) {
                v197[(v199 + v202)][(v204 + (v201 * 8))] = v196;
              }
              float v208 = v197[(v199 + v202)][(v204 + (v201 * 8))];
              float v209 = v206 * v207;
              float v210 = v208 + v209;
              v197[(v199 + v202)][(v204 + (v201 * 8))] = v210;
              if (((v203 + (v200 * 2)) - 31) == 0) {
                float v211 = v197[(v199 + v202)][(v204 + (v201 * 8))];
                v195[v202][v204].write(v211);
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
  hls::stream<float> v212[1][2],
  hls::stream<float> v213[1][2],
  float v214
) {
  loop74: for (int v215 = 0; v215 < 16; v215++) {
    loop75: for (int v216 = 0; v216 < 16; v216++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop76: for (int v217 = 0; v217 < 1; v217++) {
        loop77: for (int v218 = 0; v218 < 2; v218++) {
          float v219 = v212[v217][v218].read();
          bool v220 = v219 > v214;
          float v221 = v220 ? (float)v219 : (float)v214;
          v213[v217][v218].write(v221);
        }
      }
    }
  }
  return ;
}

void node16(
  hls::stream<float> v222[1][1],
  float v223[16][32],
  hls::stream<float> v224[1][2],
  float v225
) {
  #pragma HLS array_partition variable=v223 cyclic dim=2 factor=2

  float v226[16][32];
  #pragma HLS array_partition variable=v226 cyclic dim=2 factor=2

  float v227[16][16];
  loop78: for (int v228 = 0; v228 < 16; v228++) {
    loop79: for (int v229 = 0; v229 < 16; v229++) {
      loop80: for (int v230 = 0; v230 < 16; v230++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop81: for (int v231 = 0; v231 < 1; v231++) {
          loop82: for (int v232 = 0; v232 < 1; v232++) {
            loop83: for (int v233 = 0; v233 < 2; v233++) {
              if ((v233 + (v230 * 2)) == 0) {
                float v234 = v222[v231][v232].read();
                v227[(v228 + v231)][(v229 + v232)] = v234;
              }
              float v235 = v227[(v228 + v231)][(v229 + v232)];
              float v236 = v223[(v229 + v232)][(v233 + (v230 * 2))];
              if ((v229 + v232) == 0) {
                v226[(v228 + v231)][(v233 + (v230 * 2))] = v225;
              }
              float v237 = v226[(v228 + v231)][(v233 + (v230 * 2))];
              float v238 = v235 * v236;
              float v239 = v237 + v238;
              v226[(v228 + v231)][(v233 + (v230 * 2))] = v239;
              if (((v229 + v232) - 15) == 0) {
                float v240 = v226[(v228 + v231)][(v233 + (v230 * 2))];
                v224[v231][v233].write(v240);
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
  hls::stream<float> v241[1][1],
  hls::stream<float> v242[1][1],
  float v243
) {
  loop84: for (int v244 = 0; v244 < 16; v244++) {
    loop85: for (int v245 = 0; v245 < 16; v245++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop86: for (int v246 = 0; v246 < 1; v246++) {
        loop87: for (int v247 = 0; v247 < 1; v247++) {
          float v248 = v241[v246][v247].read();
          bool v249 = v248 > v243;
          float v250 = v249 ? (float)v248 : (float)v243;
          v242[v246][v247].write(v250);
        }
      }
    }
  }
  return ;
}

void node18(
  float v251[16][16],
  hls::stream<float> v252[1][1],
  hls::stream<float> v253[1][1],
  float v254
) {
  float v255[16][16];
  float v256[16][16];
  loop88: for (int v257 = 0; v257 < 16; v257++) {
    loop89: for (int v258 = 0; v258 < 16; v258++) {
      loop90: for (int v259 = 0; v259 < 16; v259++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop91: for (int v260 = 0; v260 < 1; v260++) {
          loop92: for (int v261 = 0; v261 < 1; v261++) {
            loop93: for (int v262 = 0; v262 < 1; v262++) {
              if ((v259 + v262) == 0) {
                float v263 = v252[v260][v261].read();
                v256[(v257 + v260)][(v258 + v261)] = v263;
              }
              float v264 = v256[(v257 + v260)][(v258 + v261)];
              float v265 = v251[(v258 + v261)][(v259 + v262)];
              if ((v258 + v261) == 0) {
                v255[(v257 + v260)][(v259 + v262)] = v254;
              }
              float v266 = v255[(v257 + v260)][(v259 + v262)];
              float v267 = v264 * v265;
              float v268 = v266 + v267;
              v255[(v257 + v260)][(v259 + v262)] = v268;
              if (((v258 + v261) - 15) == 0) {
                float v269 = v255[(v257 + v260)][(v259 + v262)];
                v253[v260][v262].write(v269);
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
  hls::stream<float> v270[1][1],
  hls::stream<float> v271[1][1],
  float v272
) {
  loop94: for (int v273 = 0; v273 < 16; v273++) {
    loop95: for (int v274 = 0; v274 < 16; v274++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop96: for (int v275 = 0; v275 < 1; v275++) {
        loop97: for (int v276 = 0; v276 < 1; v276++) {
          float v277 = v270[v275][v276].read();
          bool v278 = v277 > v272;
          float v279 = v278 ? (float)v277 : (float)v272;
          v271[v275][v276].write(v279);
        }
      }
    }
  }
  return ;
}

void node20(
  hls::stream<float> v280[1][2],
  float v281[32][16],
  hls::stream<float> v282[1][1],
  float v283
) {
  #pragma HLS array_partition variable=v281 cyclic dim=1 factor=2

  float v284[16][16];
  float v285[16][32];
  #pragma HLS array_partition variable=v285 cyclic dim=2 factor=2

  loop98: for (int v286 = 0; v286 < 16; v286++) {
    loop99: for (int v287 = 0; v287 < 16; v287++) {
      loop100: for (int v288 = 0; v288 < 16; v288++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop101: for (int v289 = 0; v289 < 1; v289++) {
          loop102: for (int v290 = 0; v290 < 2; v290++) {
            loop103: for (int v291 = 0; v291 < 1; v291++) {
              if ((v288 + v291) == 0) {
                float v292 = v280[v289][v290].read();
                v285[(v286 + v289)][(v290 + (v287 * 2))] = v292;
              }
              float v293 = v285[(v286 + v289)][(v290 + (v287 * 2))];
              float v294 = v281[(v290 + (v287 * 2))][(v288 + v291)];
              if ((v290 + (v287 * 2)) == 0) {
                v284[(v286 + v289)][(v288 + v291)] = v283;
              }
              float v295 = v284[(v286 + v289)][(v288 + v291)];
              float v296 = v293 * v294;
              float v297 = v295 + v296;
              v284[(v286 + v289)][(v288 + v291)] = v297;
              if (((v290 + (v287 * 2)) - 31) == 0) {
                float v298 = v284[(v286 + v289)][(v288 + v291)];
                v282[v289][v291].write(v298);
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
  hls::stream<float> v299[1][2],
  hls::stream<float> v300[1][2],
  float v301
) {
  loop104: for (int v302 = 0; v302 < 16; v302++) {
    loop105: for (int v303 = 0; v303 < 16; v303++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop106: for (int v304 = 0; v304 < 1; v304++) {
        loop107: for (int v305 = 0; v305 < 2; v305++) {
          float v306 = v299[v304][v305].read();
          bool v307 = v306 > v301;
          float v308 = v307 ? (float)v306 : (float)v301;
          v300[v304][v305].write(v308);
        }
      }
    }
  }
  return ;
}

void node22(
  float v309[64][32],
  hls::stream<float> v310[1][4],
  hls::stream<float> v311[1][2],
  float v312
) {
  #pragma HLS array_partition variable=v309 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v309 cyclic dim=2 factor=2

  float v313[16][32];
  #pragma HLS array_partition variable=v313 cyclic dim=2 factor=2

  float v314[16][64];
  #pragma HLS array_partition variable=v314 cyclic dim=2 factor=4

  loop108: for (int v315 = 0; v315 < 16; v315++) {
    loop109: for (int v316 = 0; v316 < 16; v316++) {
      loop110: for (int v317 = 0; v317 < 16; v317++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop111: for (int v318 = 0; v318 < 1; v318++) {
          loop112: for (int v319 = 0; v319 < 4; v319++) {
            loop113: for (int v320 = 0; v320 < 2; v320++) {
              if ((v320 + (v317 * 2)) == 0) {
                float v321 = v310[v318][v319].read();
                v314[(v315 + v318)][(v319 + (v316 * 4))] = v321;
              }
              float v322 = v314[(v315 + v318)][(v319 + (v316 * 4))];
              float v323 = v309[(v319 + (v316 * 4))][(v320 + (v317 * 2))];
              if ((v319 + (v316 * 4)) == 0) {
                v313[(v315 + v318)][(v320 + (v317 * 2))] = v312;
              }
              float v324 = v313[(v315 + v318)][(v320 + (v317 * 2))];
              float v325 = v322 * v323;
              float v326 = v324 + v325;
              v313[(v315 + v318)][(v320 + (v317 * 2))] = v326;
              if (((v319 + (v316 * 4)) - 63) == 0) {
                float v327 = v313[(v315 + v318)][(v320 + (v317 * 2))];
                v311[v318][v320].write(v327);
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
  hls::stream<float> v328[1][4],
  hls::stream<float> v329[1][4],
  float v330
) {
  loop114: for (int v331 = 0; v331 < 16; v331++) {
    loop115: for (int v332 = 0; v332 < 16; v332++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop116: for (int v333 = 0; v333 < 1; v333++) {
        loop117: for (int v334 = 0; v334 < 4; v334++) {
          float v335 = v328[v333][v334].read();
          bool v336 = v335 > v330;
          float v337 = v336 ? (float)v335 : (float)v330;
          v329[v333][v334].write(v337);
        }
      }
    }
  }
  return ;
}

void node24(
  float v338[128][64],
  hls::stream<float> v339[1][8],
  hls::stream<float> v340[1][4],
  float v341
) {
  #pragma HLS array_partition variable=v338 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v338 cyclic dim=2 factor=4

  float v342[16][64];
  #pragma HLS array_partition variable=v342 cyclic dim=2 factor=4

  float v343[16][128];
  #pragma HLS array_partition variable=v343 cyclic dim=2 factor=8

  loop118: for (int v344 = 0; v344 < 16; v344++) {
    loop119: for (int v345 = 0; v345 < 16; v345++) {
      loop120: for (int v346 = 0; v346 < 16; v346++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop121: for (int v347 = 0; v347 < 1; v347++) {
          loop122: for (int v348 = 0; v348 < 8; v348++) {
            loop123: for (int v349 = 0; v349 < 4; v349++) {
              if ((v349 + (v346 * 4)) == 0) {
                float v350 = v339[v347][v348].read();
                v343[(v344 + v347)][(v348 + (v345 * 8))] = v350;
              }
              float v351 = v343[(v344 + v347)][(v348 + (v345 * 8))];
              float v352 = v338[(v348 + (v345 * 8))][(v349 + (v346 * 4))];
              if ((v348 + (v345 * 8)) == 0) {
                v342[(v344 + v347)][(v349 + (v346 * 4))] = v341;
              }
              float v353 = v342[(v344 + v347)][(v349 + (v346 * 4))];
              float v354 = v351 * v352;
              float v355 = v353 + v354;
              v342[(v344 + v347)][(v349 + (v346 * 4))] = v355;
              if (((v348 + (v345 * 8)) - 127) == 0) {
                float v356 = v342[(v344 + v347)][(v349 + (v346 * 4))];
                v340[v347][v349].write(v356);
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
  hls::stream<float> v357[1][8],
  hls::stream<float> v358[1][8],
  float v359
) {
  loop124: for (int v360 = 0; v360 < 16; v360++) {
    loop125: for (int v361 = 0; v361 < 16; v361++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop126: for (int v362 = 0; v362 < 1; v362++) {
        loop127: for (int v363 = 0; v363 < 8; v363++) {
          float v364 = v357[v362][v363].read();
          bool v365 = v364 > v359;
          float v366 = v365 ? (float)v364 : (float)v359;
          v358[v362][v363].write(v366);
        }
      }
    }
  }
  return ;
}

void node26(
  hls::stream<float> v367[1][4],
  float v368[64][128],
  hls::stream<float> v369[1][8],
  float v370
) {
  #pragma HLS array_partition variable=v368 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v368 cyclic dim=2 factor=8

  float v371[16][128];
  #pragma HLS array_partition variable=v371 cyclic dim=2 factor=8

  float v372[16][64];
  #pragma HLS array_partition variable=v372 cyclic dim=2 factor=4

  loop128: for (int v373 = 0; v373 < 16; v373++) {
    loop129: for (int v374 = 0; v374 < 16; v374++) {
      loop130: for (int v375 = 0; v375 < 16; v375++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop131: for (int v376 = 0; v376 < 1; v376++) {
          loop132: for (int v377 = 0; v377 < 4; v377++) {
            loop133: for (int v378 = 0; v378 < 8; v378++) {
              if ((v378 + (v375 * 8)) == 0) {
                float v379 = v367[v376][v377].read();
                v372[(v373 + v376)][(v377 + (v374 * 4))] = v379;
              }
              float v380 = v372[(v373 + v376)][(v377 + (v374 * 4))];
              float v381 = v368[(v377 + (v374 * 4))][(v378 + (v375 * 8))];
              if ((v377 + (v374 * 4)) == 0) {
                v371[(v373 + v376)][(v378 + (v375 * 8))] = v370;
              }
              float v382 = v371[(v373 + v376)][(v378 + (v375 * 8))];
              float v383 = v380 * v381;
              float v384 = v382 + v383;
              v371[(v373 + v376)][(v378 + (v375 * 8))] = v384;
              if (((v377 + (v374 * 4)) - 63) == 0) {
                float v385 = v371[(v373 + v376)][(v378 + (v375 * 8))];
                v369[v376][v378].write(v385);
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
  hls::stream<float> v386[1][4],
  hls::stream<float> v387[1][4],
  float v388
) {
  loop134: for (int v389 = 0; v389 < 16; v389++) {
    loop135: for (int v390 = 0; v390 < 16; v390++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop136: for (int v391 = 0; v391 < 1; v391++) {
        loop137: for (int v392 = 0; v392 < 4; v392++) {
          float v393 = v386[v391][v392].read();
          bool v394 = v393 > v388;
          float v395 = v394 ? (float)v393 : (float)v388;
          v387[v391][v392].write(v395);
        }
      }
    }
  }
  return ;
}

void node28(
  hls::stream<float> v396[1][8],
  float v397[128][64],
  hls::stream<float> v398[1][4],
  float v399
) {
  #pragma HLS array_partition variable=v397 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v397 cyclic dim=2 factor=4

  float v400[16][64];
  #pragma HLS array_partition variable=v400 cyclic dim=2 factor=4

  float v401[16][128];
  #pragma HLS array_partition variable=v401 cyclic dim=2 factor=8

  loop138: for (int v402 = 0; v402 < 16; v402++) {
    loop139: for (int v403 = 0; v403 < 16; v403++) {
      loop140: for (int v404 = 0; v404 < 16; v404++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop141: for (int v405 = 0; v405 < 1; v405++) {
          loop142: for (int v406 = 0; v406 < 8; v406++) {
            loop143: for (int v407 = 0; v407 < 4; v407++) {
              if ((v407 + (v404 * 4)) == 0) {
                float v408 = v396[v405][v406].read();
                v401[(v402 + v405)][(v406 + (v403 * 8))] = v408;
              }
              float v409 = v401[(v402 + v405)][(v406 + (v403 * 8))];
              float v410 = v397[(v406 + (v403 * 8))][(v407 + (v404 * 4))];
              if ((v406 + (v403 * 8)) == 0) {
                v400[(v402 + v405)][(v407 + (v404 * 4))] = v399;
              }
              float v411 = v400[(v402 + v405)][(v407 + (v404 * 4))];
              float v412 = v409 * v410;
              float v413 = v411 + v412;
              v400[(v402 + v405)][(v407 + (v404 * 4))] = v413;
              if (((v406 + (v403 * 8)) - 127) == 0) {
                float v414 = v400[(v402 + v405)][(v407 + (v404 * 4))];
                v398[v405][v407].write(v414);
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
  hls::stream<float> v415[1][8],
  hls::stream<float> v416[1][8],
  float v417
) {
  loop144: for (int v418 = 0; v418 < 16; v418++) {
    loop145: for (int v419 = 0; v419 < 16; v419++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop146: for (int v420 = 0; v420 < 1; v420++) {
        loop147: for (int v421 = 0; v421 < 8; v421++) {
          float v422 = v415[v420][v421].read();
          bool v423 = v422 > v417;
          float v424 = v423 ? (float)v422 : (float)v417;
          v416[v420][v421].write(v424);
        }
      }
    }
  }
  return ;
}

void node30(
  float v425[32][128],
  float v426[16][32],
  hls::stream<float> v427[1][8],
  float v428
) {
  #pragma HLS array_partition variable=v425 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v425 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v426 cyclic dim=2 factor=2

  float v429[16][128];
  #pragma HLS array_partition variable=v429 cyclic dim=2 factor=8

  loop148: for (int v430 = 0; v430 < 16; v430++) {
    loop149: for (int v431 = 0; v431 < 16; v431++) {
      loop150: for (int v432 = 0; v432 < 16; v432++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop151: for (int v433 = 0; v433 < 1; v433++) {
          loop152: for (int v434 = 0; v434 < 2; v434++) {
            loop153: for (int v435 = 0; v435 < 8; v435++) {
              float v436 = v426[(v430 + v433)][(v434 + (v431 * 2))];
              float v437 = v425[(v434 + (v431 * 2))][(v435 + (v432 * 8))];
              if ((v434 + (v431 * 2)) == 0) {
                v429[(v430 + v433)][(v435 + (v432 * 8))] = v428;
              }
              float v438 = v429[(v430 + v433)][(v435 + (v432 * 8))];
              float v439 = v436 * v437;
              float v440 = v438 + v439;
              v429[(v430 + v433)][(v435 + (v432 * 8))] = v440;
              if (((v434 + (v431 * 2)) - 31) == 0) {
                float v441 = v429[(v430 + v433)][(v435 + (v432 * 8))];
                v427[v433][v435].write(v441);
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
  #pragma HLS array_partition variable=v443 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v444 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v444 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v445 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v445 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v446 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v446 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v447 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v447 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v448 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v450 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v451 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v451 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v452 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v452 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v453 cyclic dim=1 factor=4
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
  hls::stream<float> v463[1][2];
	#pragma HLS STREAM variable=v463 depth=256
  hls::stream<float> v464[1][2];
	#pragma HLS STREAM variable=v464 depth=256
  hls::stream<float> v465[1][4];
	#pragma HLS STREAM variable=v465 depth=256
  hls::stream<float> v466[1][4];
	#pragma HLS STREAM variable=v466 depth=256
  hls::stream<float> v467[1][8];
	#pragma HLS STREAM variable=v467 depth=256
  hls::stream<float> v468[1][8];
	#pragma HLS STREAM variable=v468 depth=256
  hls::stream<float> v469[1][4];
	#pragma HLS STREAM variable=v469 depth=256
  hls::stream<float> v470[1][4];
	#pragma HLS STREAM variable=v470 depth=256
  hls::stream<float> v471[1][8];
	#pragma HLS STREAM variable=v471 depth=256
  hls::stream<float> v472[1][8];
	#pragma HLS STREAM variable=v472 depth=256
  hls::stream<float> v473[1][2];
	#pragma HLS STREAM variable=v473 depth=256
  hls::stream<float> v474[1][2];
	#pragma HLS STREAM variable=v474 depth=256
  hls::stream<float> v475[1][1];
	#pragma HLS STREAM variable=v475 depth=256
  hls::stream<float> v476[1][1];
	#pragma HLS STREAM variable=v476 depth=256
  hls::stream<float> v477[1][1];
	#pragma HLS STREAM variable=v477 depth=256
  hls::stream<float> v478[1][1];
	#pragma HLS STREAM variable=v478 depth=256
  hls::stream<float> v479[1][2];
	#pragma HLS STREAM variable=v479 depth=256
  hls::stream<float> v480[1][2];
	#pragma HLS STREAM variable=v480 depth=256
  hls::stream<float> v481[1][4];
	#pragma HLS STREAM variable=v481 depth=256
  hls::stream<float> v482[1][4];
	#pragma HLS STREAM variable=v482 depth=256
  hls::stream<float> v483[1][8];
	#pragma HLS STREAM variable=v483 depth=256
  hls::stream<float> v484[1][8];
	#pragma HLS STREAM variable=v484 depth=256
  hls::stream<float> v485[1][4];
	#pragma HLS STREAM variable=v485 depth=256
  hls::stream<float> v486[1][4];
	#pragma HLS STREAM variable=v486 depth=256
  hls::stream<float> v487[1][8];
	#pragma HLS STREAM variable=v487 depth=256
  hls::stream<float> v488[1][8];
	#pragma HLS STREAM variable=v488 depth=256
  node30(v443, v442, v488, 0.000000);
  node29(v488, v487, 0.000000);
  node28(v487, v444, v486, 0.000000);
  node27(v486, v485, 0.000000);
  node26(v485, v445, v484, 0.000000);
  node25(v484, v483, 0.000000);
  node24(v446, v483, v482, 0.000000);
  node23(v482, v481, 0.000000);
  node22(v447, v481, v480, 0.000000);
  node21(v480, v479, 0.000000);
  node20(v479, v448, v478, 0.000000);
  node19(v478, v477, 0.000000);
  node18(v449, v477, v476, 0.000000);
  node17(v476, v475, 0.000000);
  node16(v475, v450, v474, 0.000000);
  node15(v474, v473, 0.000000);
  node14(v451, v473, v472, 0.000000);
  node13(v472, v471, 0.000000);
  node12(v452, v471, v470, 0.000000);
  node11(v470, v469, 0.000000);
  node10(v469, v453, v468, 0.000000);
  node9(v468, v467, 0.000000);
  node8(v467, v454, v466, 0.000000);
  node7(v466, v465, 0.000000);
  node6(v455, v465, v464, 0.000000);
  node5(v464, v463, 0.000000);
  node4(v463, v456, v462, 0.000000);
  node3(v462, v461, 0.000000);
  node2(v457, v461, v460, 0.000000);
  node1(v460, v459, 0.000000, 1000000.000000);
  node0(v459, v458);
  return ;
}

