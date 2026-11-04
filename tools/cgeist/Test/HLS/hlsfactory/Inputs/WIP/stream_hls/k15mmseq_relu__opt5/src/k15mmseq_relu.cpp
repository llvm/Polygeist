
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
  hls::stream<float> v0[2][4],
  float v1[64][64]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=4

  loop0: for (int v2 = 0; v2 < 16; v2++) {
    loop1: for (int v3 = 0; v3 < 32; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 4; v4++) {
        loop3: for (int v5 = 0; v5 < 2; v5++) {
          float v6 = v0[v5][v4].read();
          v1[(v5 + (v3 * 2))][(v4 + (v2 * 4))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[2][4],
  hls::stream<float> v8[2][4],
  float v9,
  float v10
) {
  loop4: for (int v11 = 0; v11 < 16; v11++) {
    loop5: for (int v12 = 0; v12 < 32; v12++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v13 = 0; v13 < 4; v13++) {
        loop7: for (int v14 = 0; v14 < 2; v14++) {
          float v15 = v7[v14][v13].read();
          bool v16 = v15 > v9;
          float v17 = v16 ? (float)v15 : (float)v9;
          float v18 = v17 / v10;
          v8[v14][v13].write(v18);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v19[2][4],
  float v20[64][64],
  hls::stream<float> v21[2][4],
  float v22
) {
  #pragma HLS array_partition variable=v20 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v20 cyclic dim=2 factor=4

  float v23[64][64];
  #pragma HLS array_partition variable=v23 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v23 cyclic dim=2 factor=4

  float v24[64][64];
  #pragma HLS array_partition variable=v24 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v24 cyclic dim=2 factor=4

  loop8: for (int v25 = 0; v25 < 16; v25++) {
    loop9: for (int v26 = 0; v26 < 16; v26++) {
      loop10: for (int v27 = 0; v27 < 32; v27++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v28 = 0; v28 < 4; v28++) {
          loop12: for (int v29 = 0; v29 < 4; v29++) {
            loop13: for (int v30 = 0; v30 < 2; v30++) {
              if ((v29 + (v26 * 4)) == 0) {
                float v31 = v19[v30][v28].read();
                v24[(v30 + (v27 * 2))][(v28 + (v25 * 4))] = v31;
              }
              float v32 = v24[(v30 + (v27 * 2))][(v28 + (v25 * 4))];
              float v33 = v20[(v28 + (v25 * 4))][(v29 + (v26 * 4))];
              if ((v28 + (v25 * 4)) == 0) {
                v23[(v30 + (v27 * 2))][(v29 + (v26 * 4))] = v22;
              }
              float v34 = v23[(v30 + (v27 * 2))][(v29 + (v26 * 4))];
              float v35 = v32 * v33;
              float v36 = v34 + v35;
              v23[(v30 + (v27 * 2))][(v29 + (v26 * 4))] = v36;
              if (((v28 + (v25 * 4)) - 63) == 0) {
                float v37 = v23[(v30 + (v27 * 2))][(v29 + (v26 * 4))];
                v21[v30][v29].write(v37);
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
  hls::stream<float> v38[2][4],
  hls::stream<float> v39[2][4],
  float v40
) {
  loop14: for (int v41 = 0; v41 < 16; v41++) {
    loop15: for (int v42 = 0; v42 < 32; v42++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v43 = 0; v43 < 4; v43++) {
        loop17: for (int v44 = 0; v44 < 2; v44++) {
          float v45 = v38[v44][v43].read();
          bool v46 = v45 > v40;
          float v47 = v46 ? (float)v45 : (float)v40;
          v39[v44][v43].write(v47);
        }
      }
    }
  }
  return ;
}

void node4(
  float v48[64][64],
  hls::stream<float> v49[2][4],
  hls::stream<float> v50[2][4],
  float v51
) {
  #pragma HLS array_partition variable=v48 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v48 cyclic dim=2 factor=4

  float v52[64][64];
  #pragma HLS array_partition variable=v52 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v52 cyclic dim=2 factor=4

  float v53[64][64];
  #pragma HLS array_partition variable=v53 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v53 cyclic dim=2 factor=4

  loop18: for (int v54 = 0; v54 < 16; v54++) {
    loop19: for (int v55 = 0; v55 < 16; v55++) {
      loop20: for (int v56 = 0; v56 < 32; v56++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop21: for (int v57 = 0; v57 < 4; v57++) {
          loop22: for (int v58 = 0; v58 < 4; v58++) {
            loop23: for (int v59 = 0; v59 < 2; v59++) {
              if ((v57 + (v54 * 4)) == 0) {
                float v60 = v49[v59][v58].read();
                v53[(v59 + (v56 * 2))][(v58 + (v55 * 4))] = v60;
              }
              float v61 = v53[(v59 + (v56 * 2))][(v58 + (v55 * 4))];
              float v62 = v48[(v58 + (v55 * 4))][(v57 + (v54 * 4))];
              if ((v58 + (v55 * 4)) == 0) {
                v52[(v59 + (v56 * 2))][(v57 + (v54 * 4))] = v51;
              }
              float v63 = v52[(v59 + (v56 * 2))][(v57 + (v54 * 4))];
              float v64 = v61 * v62;
              float v65 = v63 + v64;
              v52[(v59 + (v56 * 2))][(v57 + (v54 * 4))] = v65;
              if (((v58 + (v55 * 4)) - 63) == 0) {
                float v66 = v52[(v59 + (v56 * 2))][(v57 + (v54 * 4))];
                v50[v59][v57].write(v66);
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
  float v67[64][64],
  hls::stream<float> v68[2][4],
  float v69
) {
  #pragma HLS array_partition variable=v67 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v67 cyclic dim=2 factor=4

  loop24: for (int v70 = 0; v70 < 16; v70++) {
    loop25: for (int v71 = 0; v71 < 32; v71++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop26: for (int v72 = 0; v72 < 4; v72++) {
        loop27: for (int v73 = 0; v73 < 2; v73++) {
          float v74 = v67[(v73 + (v71 * 2))][(v72 + (v70 * 4))];
          bool v75 = v74 > v69;
          float v76 = v75 ? (float)v74 : (float)v69;
          v68[v73][v72].write(v76);
        }
      }
    }
  }
  return ;
}

void node6(
  hls::stream<float> v77[2][4],
  float v78[64][64],
  float v79[64][64],
  float v80
) {
  #pragma HLS array_partition variable=v78 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v78 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v79 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v79 cyclic dim=2 factor=4

  float v81[64][64];
  #pragma HLS array_partition variable=v81 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v81 cyclic dim=2 factor=4

  float v82[64][64];
  #pragma HLS array_partition variable=v82 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v82 cyclic dim=2 factor=4

  loop28: for (int v83 = 0; v83 < 32; v83++) {
    loop29: for (int v84 = 0; v84 < 16; v84++) {
      loop30: for (int v85 = 0; v85 < 16; v85++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop31: for (int v86 = 0; v86 < 2; v86++) {
          loop32: for (int v87 = 0; v87 < 4; v87++) {
            loop33: for (int v88 = 0; v88 < 4; v88++) {
              if ((v88 + (v85 * 4)) == 0) {
                float v89 = v77[v86][v87].read();
                v82[(v86 + (v83 * 2))][(v87 + (v84 * 4))] = v89;
              }
              float v90 = v82[(v86 + (v83 * 2))][(v87 + (v84 * 4))];
              float v91 = v78[(v87 + (v84 * 4))][(v88 + (v85 * 4))];
              if ((v87 + (v84 * 4)) == 0) {
                v81[(v86 + (v83 * 2))][(v88 + (v85 * 4))] = v80;
              }
              float v92 = v81[(v86 + (v83 * 2))][(v88 + (v85 * 4))];
              float v93 = v90 * v91;
              float v94 = v92 + v93;
              v81[(v86 + (v83 * 2))][(v88 + (v85 * 4))] = v94;
              if (((v87 + (v84 * 4)) - 63) == 0) {
                float v95 = v81[(v86 + (v83 * 2))][(v88 + (v85 * 4))];
                v79[(v86 + (v83 * 2))][(v88 + (v85 * 4))] = v95;
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
  hls::stream<float> v96[2][4],
  hls::stream<float> v97[2][4],
  float v98
) {
  loop34: for (int v99 = 0; v99 < 32; v99++) {
    loop35: for (int v100 = 0; v100 < 16; v100++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop36: for (int v101 = 0; v101 < 2; v101++) {
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
  hls::stream<float> v106[2][4],
  float v107[64][64],
  hls::stream<float> v108[2][4],
  float v109
) {
  #pragma HLS array_partition variable=v107 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v107 cyclic dim=2 factor=4

  float v110[64][64];
  #pragma HLS array_partition variable=v110 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v110 cyclic dim=2 factor=4

  float v111[64][64];
  #pragma HLS array_partition variable=v111 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v111 cyclic dim=2 factor=4

  loop38: for (int v112 = 0; v112 < 32; v112++) {
    loop39: for (int v113 = 0; v113 < 16; v113++) {
      loop40: for (int v114 = 0; v114 < 16; v114++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop41: for (int v115 = 0; v115 < 2; v115++) {
          loop42: for (int v116 = 0; v116 < 4; v116++) {
            loop43: for (int v117 = 0; v117 < 4; v117++) {
              if ((v117 + (v114 * 4)) == 0) {
                float v118 = v106[v115][v116].read();
                v111[(v115 + (v112 * 2))][(v116 + (v113 * 4))] = v118;
              }
              float v119 = v111[(v115 + (v112 * 2))][(v116 + (v113 * 4))];
              float v120 = v107[(v116 + (v113 * 4))][(v117 + (v114 * 4))];
              if ((v116 + (v113 * 4)) == 0) {
                v110[(v115 + (v112 * 2))][(v117 + (v114 * 4))] = v109;
              }
              float v121 = v110[(v115 + (v112 * 2))][(v117 + (v114 * 4))];
              float v122 = v119 * v120;
              float v123 = v121 + v122;
              v110[(v115 + (v112 * 2))][(v117 + (v114 * 4))] = v123;
              if (((v116 + (v113 * 4)) - 63) == 0) {
                float v124 = v110[(v115 + (v112 * 2))][(v117 + (v114 * 4))];
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
  hls::stream<float> v125[2][4],
  hls::stream<float> v126[2][4],
  float v127
) {
  loop44: for (int v128 = 0; v128 < 32; v128++) {
    loop45: for (int v129 = 0; v129 < 16; v129++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop46: for (int v130 = 0; v130 < 2; v130++) {
        loop47: for (int v131 = 0; v131 < 4; v131++) {
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
  float v135[64][64],
  hls::stream<float> v136[2][4],
  hls::stream<float> v137[2][4],
  float v138
) {
  #pragma HLS array_partition variable=v135 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v135 cyclic dim=2 factor=4

  float v139[64][64];
  #pragma HLS array_partition variable=v139 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v139 cyclic dim=2 factor=4

  float v140[64][64];
  #pragma HLS array_partition variable=v140 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v140 cyclic dim=2 factor=4

  loop48: for (int v141 = 0; v141 < 32; v141++) {
    loop49: for (int v142 = 0; v142 < 16; v142++) {
      loop50: for (int v143 = 0; v143 < 16; v143++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop51: for (int v144 = 0; v144 < 2; v144++) {
          loop52: for (int v145 = 0; v145 < 4; v145++) {
            loop53: for (int v146 = 0; v146 < 4; v146++) {
              if ((v146 + (v143 * 4)) == 0) {
                float v147 = v136[v144][v145].read();
                v140[(v144 + (v141 * 2))][(v145 + (v142 * 4))] = v147;
              }
              float v148 = v140[(v144 + (v141 * 2))][(v145 + (v142 * 4))];
              float v149 = v135[(v145 + (v142 * 4))][(v146 + (v143 * 4))];
              if ((v145 + (v142 * 4)) == 0) {
                v139[(v144 + (v141 * 2))][(v146 + (v143 * 4))] = v138;
              }
              float v150 = v139[(v144 + (v141 * 2))][(v146 + (v143 * 4))];
              float v151 = v148 * v149;
              float v152 = v150 + v151;
              v139[(v144 + (v141 * 2))][(v146 + (v143 * 4))] = v152;
              if (((v145 + (v142 * 4)) - 63) == 0) {
                float v153 = v139[(v144 + (v141 * 2))][(v146 + (v143 * 4))];
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
  hls::stream<float> v154[2][4],
  hls::stream<float> v155[2][4],
  float v156
) {
  loop54: for (int v157 = 0; v157 < 32; v157++) {
    loop55: for (int v158 = 0; v158 < 16; v158++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop56: for (int v159 = 0; v159 < 2; v159++) {
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
  float v164[64][64],
  hls::stream<float> v165[2][4],
  hls::stream<float> v166[2][4],
  float v167
) {
  #pragma HLS array_partition variable=v164 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v164 cyclic dim=2 factor=4

  float v168[64][64];
  #pragma HLS array_partition variable=v168 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v168 cyclic dim=2 factor=4

  float v169[64][64];
  #pragma HLS array_partition variable=v169 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v169 cyclic dim=2 factor=4

  loop58: for (int v170 = 0; v170 < 32; v170++) {
    loop59: for (int v171 = 0; v171 < 16; v171++) {
      loop60: for (int v172 = 0; v172 < 16; v172++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop61: for (int v173 = 0; v173 < 2; v173++) {
          loop62: for (int v174 = 0; v174 < 4; v174++) {
            loop63: for (int v175 = 0; v175 < 4; v175++) {
              if ((v175 + (v172 * 4)) == 0) {
                float v176 = v165[v173][v174].read();
                v169[(v173 + (v170 * 2))][(v174 + (v171 * 4))] = v176;
              }
              float v177 = v169[(v173 + (v170 * 2))][(v174 + (v171 * 4))];
              float v178 = v164[(v174 + (v171 * 4))][(v175 + (v172 * 4))];
              if ((v174 + (v171 * 4)) == 0) {
                v168[(v173 + (v170 * 2))][(v175 + (v172 * 4))] = v167;
              }
              float v179 = v168[(v173 + (v170 * 2))][(v175 + (v172 * 4))];
              float v180 = v177 * v178;
              float v181 = v179 + v180;
              v168[(v173 + (v170 * 2))][(v175 + (v172 * 4))] = v181;
              if (((v174 + (v171 * 4)) - 63) == 0) {
                float v182 = v168[(v173 + (v170 * 2))][(v175 + (v172 * 4))];
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
  hls::stream<float> v183[2][4],
  hls::stream<float> v184[2][4],
  float v185
) {
  loop64: for (int v186 = 0; v186 < 32; v186++) {
    loop65: for (int v187 = 0; v187 < 16; v187++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop66: for (int v188 = 0; v188 < 2; v188++) {
        loop67: for (int v189 = 0; v189 < 4; v189++) {
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
  float v193[64][64],
  hls::stream<float> v194[2][4],
  hls::stream<float> v195[2][4],
  float v196
) {
  #pragma HLS array_partition variable=v193 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v193 cyclic dim=2 factor=4

  float v197[64][64];
  #pragma HLS array_partition variable=v197 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v197 cyclic dim=2 factor=4

  float v198[64][64];
  #pragma HLS array_partition variable=v198 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v198 cyclic dim=2 factor=4

  loop68: for (int v199 = 0; v199 < 32; v199++) {
    loop69: for (int v200 = 0; v200 < 16; v200++) {
      loop70: for (int v201 = 0; v201 < 16; v201++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop71: for (int v202 = 0; v202 < 2; v202++) {
          loop72: for (int v203 = 0; v203 < 4; v203++) {
            loop73: for (int v204 = 0; v204 < 4; v204++) {
              if ((v204 + (v201 * 4)) == 0) {
                float v205 = v194[v202][v203].read();
                v198[(v202 + (v199 * 2))][(v203 + (v200 * 4))] = v205;
              }
              float v206 = v198[(v202 + (v199 * 2))][(v203 + (v200 * 4))];
              float v207 = v193[(v203 + (v200 * 4))][(v204 + (v201 * 4))];
              if ((v203 + (v200 * 4)) == 0) {
                v197[(v202 + (v199 * 2))][(v204 + (v201 * 4))] = v196;
              }
              float v208 = v197[(v202 + (v199 * 2))][(v204 + (v201 * 4))];
              float v209 = v206 * v207;
              float v210 = v208 + v209;
              v197[(v202 + (v199 * 2))][(v204 + (v201 * 4))] = v210;
              if (((v203 + (v200 * 4)) - 63) == 0) {
                float v211 = v197[(v202 + (v199 * 2))][(v204 + (v201 * 4))];
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
  hls::stream<float> v212[2][4],
  hls::stream<float> v213[2][4],
  float v214
) {
  loop74: for (int v215 = 0; v215 < 32; v215++) {
    loop75: for (int v216 = 0; v216 < 16; v216++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop76: for (int v217 = 0; v217 < 2; v217++) {
        loop77: for (int v218 = 0; v218 < 4; v218++) {
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
  float v222[64][64],
  hls::stream<float> v223[2][4],
  hls::stream<float> v224[2][4],
  float v225
) {
  #pragma HLS array_partition variable=v222 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v222 cyclic dim=2 factor=4

  float v226[64][64];
  #pragma HLS array_partition variable=v226 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v226 cyclic dim=2 factor=4

  float v227[64][64];
  #pragma HLS array_partition variable=v227 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v227 cyclic dim=2 factor=4

  loop78: for (int v228 = 0; v228 < 32; v228++) {
    loop79: for (int v229 = 0; v229 < 16; v229++) {
      loop80: for (int v230 = 0; v230 < 16; v230++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop81: for (int v231 = 0; v231 < 2; v231++) {
          loop82: for (int v232 = 0; v232 < 4; v232++) {
            loop83: for (int v233 = 0; v233 < 4; v233++) {
              if ((v233 + (v230 * 4)) == 0) {
                float v234 = v223[v231][v232].read();
                v227[(v231 + (v228 * 2))][(v232 + (v229 * 4))] = v234;
              }
              float v235 = v227[(v231 + (v228 * 2))][(v232 + (v229 * 4))];
              float v236 = v222[(v232 + (v229 * 4))][(v233 + (v230 * 4))];
              if ((v232 + (v229 * 4)) == 0) {
                v226[(v231 + (v228 * 2))][(v233 + (v230 * 4))] = v225;
              }
              float v237 = v226[(v231 + (v228 * 2))][(v233 + (v230 * 4))];
              float v238 = v235 * v236;
              float v239 = v237 + v238;
              v226[(v231 + (v228 * 2))][(v233 + (v230 * 4))] = v239;
              if (((v232 + (v229 * 4)) - 63) == 0) {
                float v240 = v226[(v231 + (v228 * 2))][(v233 + (v230 * 4))];
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
  hls::stream<float> v241[2][4],
  hls::stream<float> v242[2][4],
  float v243
) {
  loop84: for (int v244 = 0; v244 < 32; v244++) {
    loop85: for (int v245 = 0; v245 < 16; v245++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop86: for (int v246 = 0; v246 < 2; v246++) {
        loop87: for (int v247 = 0; v247 < 4; v247++) {
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
  hls::stream<float> v251[2][4],
  float v252[64][64],
  hls::stream<float> v253[2][4],
  float v254
) {
  #pragma HLS array_partition variable=v252 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v252 cyclic dim=2 factor=4

  float v255[64][64];
  #pragma HLS array_partition variable=v255 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v255 cyclic dim=2 factor=4

  float v256[64][64];
  #pragma HLS array_partition variable=v256 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v256 cyclic dim=2 factor=4

  loop88: for (int v257 = 0; v257 < 32; v257++) {
    loop89: for (int v258 = 0; v258 < 16; v258++) {
      loop90: for (int v259 = 0; v259 < 16; v259++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop91: for (int v260 = 0; v260 < 2; v260++) {
          loop92: for (int v261 = 0; v261 < 4; v261++) {
            loop93: for (int v262 = 0; v262 < 4; v262++) {
              if ((v262 + (v259 * 4)) == 0) {
                float v263 = v251[v260][v261].read();
                v256[(v260 + (v257 * 2))][(v261 + (v258 * 4))] = v263;
              }
              float v264 = v256[(v260 + (v257 * 2))][(v261 + (v258 * 4))];
              float v265 = v252[(v261 + (v258 * 4))][(v262 + (v259 * 4))];
              if ((v261 + (v258 * 4)) == 0) {
                v255[(v260 + (v257 * 2))][(v262 + (v259 * 4))] = v254;
              }
              float v266 = v255[(v260 + (v257 * 2))][(v262 + (v259 * 4))];
              float v267 = v264 * v265;
              float v268 = v266 + v267;
              v255[(v260 + (v257 * 2))][(v262 + (v259 * 4))] = v268;
              if (((v261 + (v258 * 4)) - 63) == 0) {
                float v269 = v255[(v260 + (v257 * 2))][(v262 + (v259 * 4))];
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
  hls::stream<float> v270[2][4],
  hls::stream<float> v271[2][4],
  float v272
) {
  loop94: for (int v273 = 0; v273 < 32; v273++) {
    loop95: for (int v274 = 0; v274 < 16; v274++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop96: for (int v275 = 0; v275 < 2; v275++) {
        loop97: for (int v276 = 0; v276 < 4; v276++) {
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
  float v280[64][64],
  hls::stream<float> v281[2][4],
  hls::stream<float> v282[2][4],
  float v283
) {
  #pragma HLS array_partition variable=v280 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v280 cyclic dim=2 factor=4

  float v284[64][64];
  #pragma HLS array_partition variable=v284 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v284 cyclic dim=2 factor=4

  float v285[64][64];
  #pragma HLS array_partition variable=v285 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v285 cyclic dim=2 factor=4

  loop98: for (int v286 = 0; v286 < 32; v286++) {
    loop99: for (int v287 = 0; v287 < 16; v287++) {
      loop100: for (int v288 = 0; v288 < 16; v288++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop101: for (int v289 = 0; v289 < 2; v289++) {
          loop102: for (int v290 = 0; v290 < 4; v290++) {
            loop103: for (int v291 = 0; v291 < 4; v291++) {
              if ((v291 + (v288 * 4)) == 0) {
                float v292 = v281[v289][v290].read();
                v285[(v289 + (v286 * 2))][(v290 + (v287 * 4))] = v292;
              }
              float v293 = v285[(v289 + (v286 * 2))][(v290 + (v287 * 4))];
              float v294 = v280[(v290 + (v287 * 4))][(v291 + (v288 * 4))];
              if ((v290 + (v287 * 4)) == 0) {
                v284[(v289 + (v286 * 2))][(v291 + (v288 * 4))] = v283;
              }
              float v295 = v284[(v289 + (v286 * 2))][(v291 + (v288 * 4))];
              float v296 = v293 * v294;
              float v297 = v295 + v296;
              v284[(v289 + (v286 * 2))][(v291 + (v288 * 4))] = v297;
              if (((v290 + (v287 * 4)) - 63) == 0) {
                float v298 = v284[(v289 + (v286 * 2))][(v291 + (v288 * 4))];
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
  hls::stream<float> v299[2][4],
  hls::stream<float> v300[2][4],
  float v301
) {
  loop104: for (int v302 = 0; v302 < 32; v302++) {
    loop105: for (int v303 = 0; v303 < 16; v303++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop106: for (int v304 = 0; v304 < 2; v304++) {
        loop107: for (int v305 = 0; v305 < 4; v305++) {
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
  hls::stream<float> v309[2][4],
  float v310[64][64],
  hls::stream<float> v311[2][4],
  float v312
) {
  #pragma HLS array_partition variable=v310 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v310 cyclic dim=2 factor=4

  float v313[64][64];
  #pragma HLS array_partition variable=v313 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v313 cyclic dim=2 factor=4

  float v314[64][64];
  #pragma HLS array_partition variable=v314 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v314 cyclic dim=2 factor=4

  loop108: for (int v315 = 0; v315 < 32; v315++) {
    loop109: for (int v316 = 0; v316 < 16; v316++) {
      loop110: for (int v317 = 0; v317 < 16; v317++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop111: for (int v318 = 0; v318 < 2; v318++) {
          loop112: for (int v319 = 0; v319 < 4; v319++) {
            loop113: for (int v320 = 0; v320 < 4; v320++) {
              if ((v320 + (v317 * 4)) == 0) {
                float v321 = v309[v318][v319].read();
                v314[(v318 + (v315 * 2))][(v319 + (v316 * 4))] = v321;
              }
              float v322 = v314[(v318 + (v315 * 2))][(v319 + (v316 * 4))];
              float v323 = v310[(v319 + (v316 * 4))][(v320 + (v317 * 4))];
              if ((v319 + (v316 * 4)) == 0) {
                v313[(v318 + (v315 * 2))][(v320 + (v317 * 4))] = v312;
              }
              float v324 = v313[(v318 + (v315 * 2))][(v320 + (v317 * 4))];
              float v325 = v322 * v323;
              float v326 = v324 + v325;
              v313[(v318 + (v315 * 2))][(v320 + (v317 * 4))] = v326;
              if (((v319 + (v316 * 4)) - 63) == 0) {
                float v327 = v313[(v318 + (v315 * 2))][(v320 + (v317 * 4))];
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
  hls::stream<float> v328[2][4],
  hls::stream<float> v329[2][4],
  float v330
) {
  loop114: for (int v331 = 0; v331 < 32; v331++) {
    loop115: for (int v332 = 0; v332 < 16; v332++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop116: for (int v333 = 0; v333 < 2; v333++) {
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
  float v338[64][64],
  hls::stream<float> v339[2][4],
  hls::stream<float> v340[2][4],
  float v341
) {
  #pragma HLS array_partition variable=v338 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v338 cyclic dim=2 factor=4

  float v342[64][64];
  #pragma HLS array_partition variable=v342 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v342 cyclic dim=2 factor=4

  float v343[64][64];
  #pragma HLS array_partition variable=v343 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v343 cyclic dim=2 factor=4

  loop118: for (int v344 = 0; v344 < 32; v344++) {
    loop119: for (int v345 = 0; v345 < 16; v345++) {
      loop120: for (int v346 = 0; v346 < 16; v346++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop121: for (int v347 = 0; v347 < 2; v347++) {
          loop122: for (int v348 = 0; v348 < 4; v348++) {
            loop123: for (int v349 = 0; v349 < 4; v349++) {
              if ((v349 + (v346 * 4)) == 0) {
                float v350 = v339[v347][v348].read();
                v343[(v347 + (v344 * 2))][(v348 + (v345 * 4))] = v350;
              }
              float v351 = v343[(v347 + (v344 * 2))][(v348 + (v345 * 4))];
              float v352 = v338[(v348 + (v345 * 4))][(v349 + (v346 * 4))];
              if ((v348 + (v345 * 4)) == 0) {
                v342[(v347 + (v344 * 2))][(v349 + (v346 * 4))] = v341;
              }
              float v353 = v342[(v347 + (v344 * 2))][(v349 + (v346 * 4))];
              float v354 = v351 * v352;
              float v355 = v353 + v354;
              v342[(v347 + (v344 * 2))][(v349 + (v346 * 4))] = v355;
              if (((v348 + (v345 * 4)) - 63) == 0) {
                float v356 = v342[(v347 + (v344 * 2))][(v349 + (v346 * 4))];
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
  hls::stream<float> v357[2][4],
  hls::stream<float> v358[2][4],
  float v359
) {
  loop124: for (int v360 = 0; v360 < 32; v360++) {
    loop125: for (int v361 = 0; v361 < 16; v361++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop126: for (int v362 = 0; v362 < 2; v362++) {
        loop127: for (int v363 = 0; v363 < 4; v363++) {
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
  float v367[64][64],
  hls::stream<float> v368[2][4],
  hls::stream<float> v369[2][4],
  float v370
) {
  #pragma HLS array_partition variable=v367 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v367 cyclic dim=2 factor=4

  float v371[64][64];
  #pragma HLS array_partition variable=v371 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v371 cyclic dim=2 factor=4

  float v372[64][64];
  #pragma HLS array_partition variable=v372 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v372 cyclic dim=2 factor=4

  loop128: for (int v373 = 0; v373 < 32; v373++) {
    loop129: for (int v374 = 0; v374 < 16; v374++) {
      loop130: for (int v375 = 0; v375 < 16; v375++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop131: for (int v376 = 0; v376 < 2; v376++) {
          loop132: for (int v377 = 0; v377 < 4; v377++) {
            loop133: for (int v378 = 0; v378 < 4; v378++) {
              if ((v378 + (v375 * 4)) == 0) {
                float v379 = v368[v376][v377].read();
                v372[(v376 + (v373 * 2))][(v377 + (v374 * 4))] = v379;
              }
              float v380 = v372[(v376 + (v373 * 2))][(v377 + (v374 * 4))];
              float v381 = v367[(v377 + (v374 * 4))][(v378 + (v375 * 4))];
              if ((v377 + (v374 * 4)) == 0) {
                v371[(v376 + (v373 * 2))][(v378 + (v375 * 4))] = v370;
              }
              float v382 = v371[(v376 + (v373 * 2))][(v378 + (v375 * 4))];
              float v383 = v380 * v381;
              float v384 = v382 + v383;
              v371[(v376 + (v373 * 2))][(v378 + (v375 * 4))] = v384;
              if (((v377 + (v374 * 4)) - 63) == 0) {
                float v385 = v371[(v376 + (v373 * 2))][(v378 + (v375 * 4))];
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
  hls::stream<float> v386[2][4],
  hls::stream<float> v387[2][4],
  float v388
) {
  loop134: for (int v389 = 0; v389 < 32; v389++) {
    loop135: for (int v390 = 0; v390 < 16; v390++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop136: for (int v391 = 0; v391 < 2; v391++) {
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
  float v396[64][64],
  hls::stream<float> v397[2][4],
  hls::stream<float> v398[2][4],
  float v399
) {
  #pragma HLS array_partition variable=v396 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v396 cyclic dim=2 factor=4

  float v400[64][64];
  #pragma HLS array_partition variable=v400 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v400 cyclic dim=2 factor=4

  float v401[64][64];
  #pragma HLS array_partition variable=v401 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v401 cyclic dim=2 factor=4

  loop138: for (int v402 = 0; v402 < 16; v402++) {
    loop139: for (int v403 = 0; v403 < 32; v403++) {
      loop140: for (int v404 = 0; v404 < 16; v404++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop141: for (int v405 = 0; v405 < 4; v405++) {
          loop142: for (int v406 = 0; v406 < 2; v406++) {
            loop143: for (int v407 = 0; v407 < 4; v407++) {
              if ((v407 + (v404 * 4)) == 0) {
                float v408 = v397[v406][v405].read();
                v401[(v406 + (v403 * 2))][(v405 + (v402 * 4))] = v408;
              }
              float v409 = v401[(v406 + (v403 * 2))][(v405 + (v402 * 4))];
              float v410 = v396[(v405 + (v402 * 4))][(v407 + (v404 * 4))];
              if ((v405 + (v402 * 4)) == 0) {
                v400[(v406 + (v403 * 2))][(v407 + (v404 * 4))] = v399;
              }
              float v411 = v400[(v406 + (v403 * 2))][(v407 + (v404 * 4))];
              float v412 = v409 * v410;
              float v413 = v411 + v412;
              v400[(v406 + (v403 * 2))][(v407 + (v404 * 4))] = v413;
              if (((v405 + (v402 * 4)) - 63) == 0) {
                float v414 = v400[(v406 + (v403 * 2))][(v407 + (v404 * 4))];
                v398[v406][v407].write(v414);
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
  hls::stream<float> v415[2][4],
  hls::stream<float> v416[2][4],
  float v417
) {
  loop144: for (int v418 = 0; v418 < 16; v418++) {
    loop145: for (int v419 = 0; v419 < 32; v419++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop146: for (int v420 = 0; v420 < 4; v420++) {
        loop147: for (int v421 = 0; v421 < 2; v421++) {
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
  float v425[64][64],
  float v426[64][64],
  hls::stream<float> v427[2][4],
  float v428
) {
  #pragma HLS array_partition variable=v425 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v425 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v426 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v426 cyclic dim=2 factor=4

  float v429[64][64];
  #pragma HLS array_partition variable=v429 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v429 cyclic dim=2 factor=4

  loop148: for (int v430 = 0; v430 < 16; v430++) {
    loop149: for (int v431 = 0; v431 < 16; v431++) {
      loop150: for (int v432 = 0; v432 < 32; v432++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop151: for (int v433 = 0; v433 < 4; v433++) {
          loop152: for (int v434 = 0; v434 < 4; v434++) {
            loop153: for (int v435 = 0; v435 < 2; v435++) {
              float v436 = v426[(v435 + (v432 * 2))][(v434 + (v431 * 4))];
              float v437 = v425[(v434 + (v431 * 4))][(v433 + (v430 * 4))];
              if ((v434 + (v431 * 4)) == 0) {
                v429[(v435 + (v432 * 2))][(v433 + (v430 * 4))] = v428;
              }
              float v438 = v429[(v435 + (v432 * 2))][(v433 + (v430 * 4))];
              float v439 = v436 * v437;
              float v440 = v438 + v439;
              v429[(v435 + (v432 * 2))][(v433 + (v430 * 4))] = v440;
              if (((v434 + (v431 * 4)) - 63) == 0) {
                float v441 = v429[(v435 + (v432 * 2))][(v433 + (v430 * 4))];
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
  float v442[64][64],
  float v443[64][64],
  float v444[64][64],
  float v445[64][64],
  float v446[64][64],
  float v447[64][64],
  float v448[64][64],
  float v449[64][64],
  float v450[64][64],
  float v451[64][64],
  float v452[64][64],
  float v453[64][64],
  float v454[64][64],
  float v455[64][64],
  float v456[64][64],
  float v457[64][64],
  float v458[64][64]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v442 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v442 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v443 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v443 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v444 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v444 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v445 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v445 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v446 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v446 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v447 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v447 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v448 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v448 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v449 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v449 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v450 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v450 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v451 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v451 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v452 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v452 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v453 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v453 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v454 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v454 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v455 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v455 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v456 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v456 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v457 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v457 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v458 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v458 cyclic dim=2 factor=4

  hls::stream<float> v459[2][4];
	#pragma HLS STREAM variable=v459 depth=512
  hls::stream<float> v460[2][4];
	#pragma HLS STREAM variable=v460 depth=512
  hls::stream<float> v461[2][4];
	#pragma HLS STREAM variable=v461 depth=512
  hls::stream<float> v462[2][4];
	#pragma HLS STREAM variable=v462 depth=512
  hls::stream<float> v463[2][4];
	#pragma HLS STREAM variable=v463 depth=512
  float v464[64][64];
  #pragma HLS array_partition variable=v464 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v464 cyclic dim=2 factor=4

  hls::stream<float> v465[2][4];
	#pragma HLS STREAM variable=v465 depth=512
  hls::stream<float> v466[2][4];
	#pragma HLS STREAM variable=v466 depth=512
  hls::stream<float> v467[2][4];
	#pragma HLS STREAM variable=v467 depth=512
  hls::stream<float> v468[2][4];
	#pragma HLS STREAM variable=v468 depth=512
  hls::stream<float> v469[2][4];
	#pragma HLS STREAM variable=v469 depth=512
  hls::stream<float> v470[2][4];
	#pragma HLS STREAM variable=v470 depth=512
  hls::stream<float> v471[2][4];
	#pragma HLS STREAM variable=v471 depth=512
  hls::stream<float> v472[2][4];
	#pragma HLS STREAM variable=v472 depth=512
  hls::stream<float> v473[2][4];
	#pragma HLS STREAM variable=v473 depth=512
  hls::stream<float> v474[2][4];
	#pragma HLS STREAM variable=v474 depth=512
  hls::stream<float> v475[2][4];
	#pragma HLS STREAM variable=v475 depth=512
  hls::stream<float> v476[2][4];
	#pragma HLS STREAM variable=v476 depth=512
  hls::stream<float> v477[2][4];
	#pragma HLS STREAM variable=v477 depth=512
  hls::stream<float> v478[2][4];
	#pragma HLS STREAM variable=v478 depth=512
  hls::stream<float> v479[2][4];
	#pragma HLS STREAM variable=v479 depth=512
  hls::stream<float> v480[2][4];
	#pragma HLS STREAM variable=v480 depth=512
  hls::stream<float> v481[2][4];
	#pragma HLS STREAM variable=v481 depth=512
  hls::stream<float> v482[2][4];
	#pragma HLS STREAM variable=v482 depth=512
  hls::stream<float> v483[2][4];
	#pragma HLS STREAM variable=v483 depth=512
  hls::stream<float> v484[2][4];
	#pragma HLS STREAM variable=v484 depth=512
  hls::stream<float> v485[2][4];
	#pragma HLS STREAM variable=v485 depth=512
  hls::stream<float> v486[2][4];
	#pragma HLS STREAM variable=v486 depth=512
  hls::stream<float> v487[2][4];
	#pragma HLS STREAM variable=v487 depth=512
  hls::stream<float> v488[2][4];
	#pragma HLS STREAM variable=v488 depth=512
  node30(v443, v442, v488, 0.000000);
  node29(v488, v487, 0.000000);
  node28(v444, v487, v486, 0.000000);
  node27(v486, v485, 0.000000);
  node26(v445, v485, v484, 0.000000);
  node25(v484, v483, 0.000000);
  node24(v446, v483, v482, 0.000000);
  node23(v482, v481, 0.000000);
  node22(v481, v447, v480, 0.000000);
  node21(v480, v479, 0.000000);
  node20(v448, v479, v478, 0.000000);
  node19(v478, v477, 0.000000);
  node18(v477, v449, v476, 0.000000);
  node17(v476, v475, 0.000000);
  node16(v450, v475, v474, 0.000000);
  node15(v474, v473, 0.000000);
  node14(v451, v473, v472, 0.000000);
  node13(v472, v471, 0.000000);
  node12(v452, v471, v470, 0.000000);
  node11(v470, v469, 0.000000);
  node10(v453, v469, v468, 0.000000);
  node9(v468, v467, 0.000000);
  node8(v467, v454, v466, 0.000000);
  node7(v466, v465, 0.000000);
  node6(v465, v455, v464, 0.000000);
  node5(v464, v463, 0.000000);
  node4(v456, v463, v462, 0.000000);
  node3(v462, v461, 0.000000);
  node2(v461, v457, v460, 0.000000);
  node1(v460, v459, 0.000000, 1000000.000000);
  node0(v459, v458);
  return ;
}

