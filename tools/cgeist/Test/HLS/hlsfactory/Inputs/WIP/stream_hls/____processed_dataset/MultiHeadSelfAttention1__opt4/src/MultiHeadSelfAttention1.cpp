
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
  hls::stream<float> v0[2][32],
  float v1[1][64][128]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v1 cyclic dim=3 factor=32

  loop0: for (int v2 = 0; v2 < 4; v2++) {
    loop1: for (int v3 = 0; v3 < 32; v3++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 32; v4++) {
        loop3: for (int v5 = 0; v5 < 2; v5++) {
          float v6 = v0[v5][v4].read();
          v1[0][(v5 + (v3 * 2))][(v4 + (v2 * 32))] = v6;
        }
      }
    }
  }
  return ;
}

void node1(
  hls::stream<float> v7[2][32],
  float v8[128],
  hls::stream<float> v9[2][32]
) {
  #pragma HLS array_partition variable=v8 cyclic dim=1 factor=32

  loop4: for (int v10 = 0; v10 < 4; v10++) {
    loop5: for (int v11 = 0; v11 < 32; v11++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v12 = 0; v12 < 32; v12++) {
        loop7: for (int v13 = 0; v13 < 2; v13++) {
          float v14 = v7[v13][v12].read();
          float v15 = v8[(v12 + (v10 * 32))];
          float v16 = v14 + v15;
          v9[v13][v12].write(v16);
        }
      }
    }
  }
  return ;
}

void node2(
  hls::stream<float> v17[2][32],
  hls::stream<float> v18[2][2],
  hls::stream<float> v19[2][32],
  float v20
) {
  float v21[64][128];
  #pragma HLS array_partition variable=v21 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v21 cyclic dim=2 factor=32

  float v22[128][128];
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v22 cyclic dim=2 factor=32

  float v23[64][128];
  #pragma HLS array_partition variable=v23 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v23 cyclic dim=2 factor=2

  loop8: for (int v24 = 0; v24 < 4; v24++) {
    loop9: for (int v25 = 0; v25 < 64; v25++) {
      loop10: for (int v26 = 0; v26 < 32; v26++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v27 = 0; v27 < 32; v27++) {
          loop12: for (int v28 = 0; v28 < 2; v28++) {
            loop13: for (int v29 = 0; v29 < 2; v29++) {
              if ((v27 + (v24 * 32)) == 0) {
                float v30 = v18[v29][v28].read();
                v23[(v29 + (v26 * 2))][(v28 + (v25 * 2))] = v30;
              }
              float v31 = v23[(v29 + (v26 * 2))][(v28 + (v25 * 2))];
              if ((v29 + (v26 * 2)) == 0) {
                float v32 = v17[v28][v27].read();
                v22[(v28 + (v25 * 2))][(v27 + (v24 * 32))] = v32;
              }
              float v33 = v22[(v28 + (v25 * 2))][(v27 + (v24 * 32))];
              if ((v28 + (v25 * 2)) == 0) {
                v21[(v29 + (v26 * 2))][(v27 + (v24 * 32))] = v20;
              }
              float v34 = v21[(v29 + (v26 * 2))][(v27 + (v24 * 32))];
              float v35 = v31 * v33;
              float v36 = v34 + v35;
              v21[(v29 + (v26 * 2))][(v27 + (v24 * 32))] = v36;
              if (((v28 + (v25 * 2)) - 127) == 0) {
                float v37 = v21[(v29 + (v26 * 2))][(v27 + (v24 * 32))];
                v19[v29][v27].write(v37);
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
  float v38[128][128],
  hls::stream<float> v39[2][32]
) {
  #pragma HLS array_partition variable=v38 cyclic dim=1 factor=32
  #pragma HLS array_partition variable=v38 cyclic dim=2 factor=2

  loop14: for (int v40 = 0; v40 < 4; v40++) {
    loop15: for (int v41 = 0; v41 < 64; v41++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v42 = 0; v42 < 32; v42++) {
        loop17: for (int v43 = 0; v43 < 2; v43++) {
          float v44 = v38[(v42 + (v40 * 32))][(v43 + (v41 * 2))];
          v39[v43][v42].write(v44);
        }
      }
    }
  }
  return ;
}

void node4(
  float v45[64][8][16],
  hls::stream<float> v46[2][2]
) {
  #pragma HLS array_partition variable=v45 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v45 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v45 cyclic dim=3 factor=4

  loop18: for (int v47 = 0; v47 < 64; v47++) {
    loop19: for (int v48 = 0; v48 < 32; v48++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop20: for (int v49 = 0; v49 < 2; v49++) {
        loop21: for (int v50 = 0; v50 < 2; v50++) {
          float v51 = v45[(v50 + (v48 * 2))][(v47 / 8)][((v49 + (v47 * 2)) % 16)];
          v46[v50][v49].write(v51);
        }
      }
    }
  }
  return ;
}

void node5(
  float v52[8][64][16],
  float v53[64][8][16]
) {
  #pragma HLS array_partition variable=v52 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v52 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v52 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v53 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v53 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v53 cyclic dim=3 factor=4

  loop22: for (int v54 = 0; v54 < 4; v54++) {
    loop23: for (int v55 = 0; v55 < 32; v55++) {
      loop24: for (int v56 = 0; v56 < 4; v56++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v57 = 0; v57 < 2; v57++) {
          loop26: for (int v58 = 0; v58 < 2; v58++) {
            loop27: for (int v59 = 0; v59 < 4; v59++) {
              float v60 = v52[(v57 + (v54 * 2))][(v58 + (v55 * 2))][(v59 + (v56 * 4))];
              v53[(v58 + (v55 * 2))][(v57 + (v54 * 2))][(v59 + (v56 * 4))] = v60;
            }
          }
        }
      }
    }
  }
  return ;
}

void node6(
  float v61[8][64][16],
  hls::stream<float> v62[2][4][4],
  float v63[8][64][16],
  float v64
) {
  #pragma HLS array_partition variable=v61 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v61 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v61 cyclic dim=3 factor=2

  #pragma HLS array_partition variable=v63 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v63 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v63 cyclic dim=3 factor=2

  float v65[8][64][16];
  #pragma HLS array_partition variable=v65 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v65 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v65 cyclic dim=3 factor=2

  float v66[8][64][16];
  #pragma HLS array_partition variable=v66 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v66 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v66 cyclic dim=3 factor=2

  float v67[8][64][64];
  #pragma HLS array_partition variable=v67 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v67 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v67 cyclic dim=3 factor=4

  loop28: for (int v68 = 0; v68 < 4; v68++) {
    loop29: for (int v69 = 0; v69 < 16; v69++) {
      loop30: for (int v70 = 0; v70 < 16; v70++) {
        loop31: for (int v71 = 0; v71 < 8; v71++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          loop32: for (int v72 = 0; v72 < 2; v72++) {
            loop33: for (int v73 = 0; v73 < 4; v73++) {
              loop34: for (int v74 = 0; v74 < 4; v74++) {
                loop35: for (int v75 = 0; v75 < 2; v75++) {
                  if ((v75 + (v71 * 2)) == 0) {
                    float v76 = v62[v72][v74][v73].read();
                    v67[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v73 + (v69 * 4))] = v76;
                  }
                  float v77 = v67[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v73 + (v69 * 4))];
                  if ((v74 + (v70 * 4)) == 0) {
                    float v78 = v61[(v72 + (v68 * 2))][(v73 + (v69 * 4))][(v75 + (v71 * 2))];
                    v66[(v72 + (v68 * 2))][(v73 + (v69 * 4))][(v75 + (v71 * 2))] = v78;
                  }
                  float v79 = v66[(v72 + (v68 * 2))][(v73 + (v69 * 4))][(v75 + (v71 * 2))];
                  if ((v73 + (v69 * 4)) == 0) {
                    v65[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v75 + (v71 * 2))] = v64;
                  }
                  float v80 = v65[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v75 + (v71 * 2))];
                  float v81 = v77 * v79;
                  float v82 = v80 + v81;
                  v65[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v75 + (v71 * 2))] = v82;
                  if (((v73 + (v69 * 4)) - 63) == 0) {
                    float v83 = v65[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v75 + (v71 * 2))];
                    v63[(v72 + (v68 * 2))][(v74 + (v70 * 4))][(v75 + (v71 * 2))] = v83;
                  }
                }
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
  float v84[64][128],
  float v85[8][64][16]
) {
  #pragma HLS array_partition variable=v84 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v84 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v85 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v85 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v85 cyclic dim=3 factor=2

  loop36: for (int v86 = 0; v86 < 4; v86++) {
    loop37: for (int v87 = 0; v87 < 8; v87++) {
      loop38: for (int v88 = 0; v88 < 16; v88++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v89 = 0; v89 < 2; v89++) {
          loop40: for (int v90 = 0; v90 < 2; v90++) {
            loop41: for (int v91 = 0; v91 < 4; v91++) {
              float v92 = v84[(v91 + (v88 * 4))][((((v89 * 16) + (v86 * 32)) + v90) + (v87 * 2))];
              v85[(v89 + (v86 * 2))][(v91 + (v88 * 4))][(v90 + (v87 * 2))] = v92;
            }
          }
        }
      }
    }
  }
  return ;
}

void node8(
  hls::stream<float> v93[2][4][4],
  hls::stream<float> v94[2][4],
  hls::stream<float> v95[2][4][4]
) {
  float v96[8][64];
  #pragma HLS array_partition variable=v96 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v96 cyclic dim=2 factor=4

  loop42: for (int v97 = 0; v97 < 4; v97++) {
    loop43: for (int v98 = 0; v98 < 16; v98++) {
      loop44: for (int v99 = 0; v99 < 16; v99++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop45: for (int v100 = 0; v100 < 2; v100++) {
          loop46: for (int v101 = 0; v101 < 4; v101++) {
            loop47: for (int v102 = 0; v102 < 4; v102++) {
              float v103 = v93[v100][v102][v101].read();
              if ((v101 + (v98 * 4)) == 0) {
                float v104 = v94[v100][v102].read();
                v96[(v100 + (v97 * 2))][(v102 + (v99 * 4))] = v104;
              }
              float v105 = v96[(v100 + (v97 * 2))][(v102 + (v99 * 4))];
              float v106 = v103 / v105;
              v95[v100][v102][v101].write(v106);
            }
          }
        }
      }
    }
  }
  return ;
}

void node9(
  hls::stream<float> v107[2][4][4],
  hls::stream<float> v108[2][4],
  float v109
) {
  float v110[8][64];
  #pragma HLS array_partition variable=v110 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v110 cyclic dim=2 factor=4

  loop48: for (int v111 = 0; v111 < 4; v111++) {
    loop49: for (int v112 = 0; v112 < 16; v112++) {
      loop50: for (int v113 = 0; v113 < 16; v113++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop51: for (int v114 = 0; v114 < 2; v114++) {
          loop52: for (int v115 = 0; v115 < 4; v115++) {
            loop53: for (int v116 = 0; v116 < 4; v116++) {
              float v117 = v107[v114][v116][v115].read();
              if ((v115 + (v112 * 4)) == 0) {
                v110[(v114 + (v111 * 2))][(v116 + (v113 * 4))] = v109;
              }
              float v118 = v110[(v114 + (v111 * 2))][(v116 + (v113 * 4))];
              float v119 = v117 + v118;
              v110[(v114 + (v111 * 2))][(v116 + (v113 * 4))] = v119;
              if (((v115 + (v112 * 4)) - 63) == 0) {
                float v120 = v110[(v114 + (v111 * 2))][(v116 + (v113 * 4))];
                v108[v114][v116].write(v120);
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
  hls::stream<float> v121[2][4][4],
  hls::stream<float> v122[2][4],
  hls::stream<float> v123[2][4][4],
  hls::stream<float> v124[2][4][4]
) {
  float v125[8][64];
  #pragma HLS array_partition variable=v125 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v125 cyclic dim=2 factor=4

  loop54: for (int v126 = 0; v126 < 4; v126++) {
    loop55: for (int v127 = 0; v127 < 16; v127++) {
      loop56: for (int v128 = 0; v128 < 16; v128++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop57: for (int v129 = 0; v129 < 2; v129++) {
          loop58: for (int v130 = 0; v130 < 4; v130++) {
            loop59: for (int v131 = 0; v131 < 4; v131++) {
              float v132 = v121[v129][v131][v130].read();
              if ((v130 + (v127 * 4)) == 0) {
                float v133 = v122[v129][v131].read();
                v125[(v129 + (v126 * 2))][(v131 + (v128 * 4))] = v133;
              }
              float v134 = v125[(v129 + (v126 * 2))][(v131 + (v128 * 4))];
              float v135 = v132 - v134;
              float v136 = exp(v135);
              v123[v129][v131][v130].write(v136);
              v124[v129][v131][v130].write(v136);
            }
          }
        }
      }
    }
  }
  return ;
}

void node11(
  hls::stream<float> v137[2][4][4],
  hls::stream<float> v138[2][4],
  float v139
) {
  float v140[8][64];
  #pragma HLS array_partition variable=v140 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v140 cyclic dim=2 factor=4

  loop60: for (int v141 = 0; v141 < 4; v141++) {
    loop61: for (int v142 = 0; v142 < 16; v142++) {
      loop62: for (int v143 = 0; v143 < 16; v143++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop63: for (int v144 = 0; v144 < 2; v144++) {
          loop64: for (int v145 = 0; v145 < 4; v145++) {
            loop65: for (int v146 = 0; v146 < 4; v146++) {
              float v147 = v137[v144][v146][v145].read();
              if ((v145 + (v142 * 4)) == 0) {
                v140[(v144 + (v141 * 2))][(v146 + (v143 * 4))] = v139;
              }
              float v148 = v140[(v144 + (v141 * 2))][(v146 + (v143 * 4))];
              float v149 = fmax(v147, v148);
              v140[(v144 + (v141 * 2))][(v146 + (v143 * 4))] = v149;
              if (((v145 + (v142 * 4)) - 63) == 0) {
                float v150 = v140[(v144 + (v141 * 2))][(v146 + (v143 * 4))];
                v138[v144][v146].write(v150);
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
  hls::stream<float> v151[2][4][4],
  hls::stream<float> v152[2][4][4],
  hls::stream<float> v153[2][4][4],
  float v154
) {
  loop66: for (int v155 = 0; v155 < 4; v155++) {
    loop67: for (int v156 = 0; v156 < 16; v156++) {
      loop68: for (int v157 = 0; v157 < 16; v157++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop69: for (int v158 = 0; v158 < 2; v158++) {
          loop70: for (int v159 = 0; v159 < 4; v159++) {
            loop71: for (int v160 = 0; v160 < 4; v160++) {
              float v161 = v151[v158][v160][v159].read();
              float v162 = v161 / v154;
              v152[v158][v160][v159].write(v162);
              v153[v158][v160][v159].write(v162);
            }
          }
        }
      }
    }
  }
  return ;
}

void node13(
  hls::stream<float> v163[2][4][2],
  hls::stream<float> v164[2][2][4],
  hls::stream<float> v165[2][4][4],
  float v166
) {
  float v167[8][64][64];
  #pragma HLS array_partition variable=v167 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v167 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v167 cyclic dim=3 factor=4

  float v168[8][16][64];
  #pragma HLS array_partition variable=v168 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v168 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v168 cyclic dim=3 factor=4

  float v169[8][64][16];
  #pragma HLS array_partition variable=v169 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v169 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v169 cyclic dim=3 factor=2

  loop72: for (int v170 = 0; v170 < 4; v170++) {
    loop73: for (int v171 = 0; v171 < 8; v171++) {
      loop74: for (int v172 = 0; v172 < 16; v172++) {
        loop75: for (int v173 = 0; v173 < 16; v173++) {
          #pragma HLS pipeline II=1
          #pragma HLS loop_flatten
          loop76: for (int v174 = 0; v174 < 2; v174++) {
            loop77: for (int v175 = 0; v175 < 2; v175++) {
              loop78: for (int v176 = 0; v176 < 4; v176++) {
                loop79: for (int v177 = 0; v177 < 4; v177++) {
                  if ((v176 + (v172 * 4)) == 0) {
                    float v178 = v163[v174][v177][v175].read();
                    v169[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v175 + (v171 * 2))] = v178;
                  }
                  float v179 = v169[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v175 + (v171 * 2))];
                  if ((v177 + (v173 * 4)) == 0) {
                    float v180 = v164[v174][v175][v176].read();
                    v168[(v174 + (v170 * 2))][(v175 + (v171 * 2))][(v176 + (v172 * 4))] = v180;
                  }
                  float v181 = v168[(v174 + (v170 * 2))][(v175 + (v171 * 2))][(v176 + (v172 * 4))];
                  if ((v175 + (v171 * 2)) == 0) {
                    v167[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v176 + (v172 * 4))] = v166;
                  }
                  float v182 = v167[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v176 + (v172 * 4))];
                  float v183 = v179 * v181;
                  float v184 = v182 + v183;
                  v167[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v176 + (v172 * 4))] = v184;
                  if (((v175 + (v171 * 2)) - 15) == 0) {
                    float v185 = v167[(v174 + (v170 * 2))][(v177 + (v173 * 4))][(v176 + (v172 * 4))];
                    v165[v174][v177][v176].write(v185);
                  }
                }
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
  float v186[64][128],
  hls::stream<float> v187[2][2][4]
) {
  #pragma HLS array_partition variable=v186 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v186 cyclic dim=2 factor=2

  loop80: for (int v188 = 0; v188 < 4; v188++) {
    loop81: for (int v189 = 0; v189 < 8; v189++) {
      loop82: for (int v190 = 0; v190 < 16; v190++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop83: for (int v191 = 0; v191 < 2; v191++) {
          loop84: for (int v192 = 0; v192 < 2; v192++) {
            loop85: for (int v193 = 0; v193 < 4; v193++) {
              float v194 = v186[(v193 + (v190 * 4))][((((v191 * 16) + (v188 * 32)) + v192) + (v189 * 2))];
              v187[v191][v192][v193].write(v194);
            }
          }
        }
      }
    }
  }
  return ;
}

void node15(
  float v195[64][128],
  hls::stream<float> v196[2][4][2]
) {
  #pragma HLS array_partition variable=v195 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v195 cyclic dim=2 factor=2

  loop86: for (int v197 = 0; v197 < 4; v197++) {
    loop87: for (int v198 = 0; v198 < 8; v198++) {
      loop88: for (int v199 = 0; v199 < 16; v199++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop89: for (int v200 = 0; v200 < 2; v200++) {
          loop90: for (int v201 = 0; v201 < 2; v201++) {
            loop91: for (int v202 = 0; v202 < 4; v202++) {
              float v203 = v195[(v202 + (v199 * 4))][((((v200 * 16) + (v197 * 32)) + v201) + (v198 * 2))];
              v196[v200][v202][v201].write(v203);
            }
          }
        }
      }
    }
  }
  return ;
}

void node16(
  hls::stream<float> v204[4][2],
  float v205[128],
  float v206[64][128]
) {
  #pragma HLS array_partition variable=v205 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v206 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v206 cyclic dim=2 factor=2

  loop92: for (int v207 = 0; v207 < 16; v207++) {
    loop93: for (int v208 = 0; v208 < 64; v208++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop94: for (int v209 = 0; v209 < 4; v209++) {
        loop95: for (int v210 = 0; v210 < 2; v210++) {
          float v211 = v204[v209][v210].read();
          float v212 = v205[(v210 + (v208 * 2))];
          float v213 = v211 + v212;
          v206[(v209 + (v207 * 4))][(v210 + (v208 * 2))] = v213;
        }
      }
    }
  }
  return ;
}

void node17(
  hls::stream<float> v214[8][2],
  hls::stream<float> v215[4][8],
  hls::stream<float> v216[4][2],
  float v217
) {
  float v218[64][128];
  #pragma HLS array_partition variable=v218 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v218 cyclic dim=2 factor=2

  float v219[128][128];
  #pragma HLS array_partition variable=v219 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v219 cyclic dim=2 factor=2

  float v220[64][128];
  #pragma HLS array_partition variable=v220 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v220 cyclic dim=2 factor=8

  loop96: for (int v221 = 0; v221 < 16; v221++) {
    loop97: for (int v222 = 0; v222 < 16; v222++) {
      loop98: for (int v223 = 0; v223 < 64; v223++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop99: for (int v224 = 0; v224 < 4; v224++) {
          loop100: for (int v225 = 0; v225 < 8; v225++) {
            loop101: for (int v226 = 0; v226 < 2; v226++) {
              if ((v226 + (v223 * 2)) == 0) {
                float v227 = v215[v224][v225].read();
                v220[(v224 + (v221 * 4))][(v225 + (v222 * 8))] = v227;
              }
              float v228 = v220[(v224 + (v221 * 4))][(v225 + (v222 * 8))];
              if ((v224 + (v221 * 4)) == 0) {
                float v229 = v214[v225][v226].read();
                v219[(v225 + (v222 * 8))][(v226 + (v223 * 2))] = v229;
              }
              float v230 = v219[(v225 + (v222 * 8))][(v226 + (v223 * 2))];
              if ((v225 + (v222 * 8)) == 0) {
                v218[(v224 + (v221 * 4))][(v226 + (v223 * 2))] = v217;
              }
              float v231 = v218[(v224 + (v221 * 4))][(v226 + (v223 * 2))];
              float v232 = v228 * v230;
              float v233 = v231 + v232;
              v218[(v224 + (v221 * 4))][(v226 + (v223 * 2))] = v233;
              if (((v225 + (v222 * 8)) - 127) == 0) {
                float v234 = v218[(v224 + (v221 * 4))][(v226 + (v223 * 2))];
                v216[v224][v226].write(v234);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node18(
  float v235[128][128],
  hls::stream<float> v236[8][2]
) {
  #pragma HLS array_partition variable=v235 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v235 cyclic dim=2 factor=8

  loop102: for (int v237 = 0; v237 < 16; v237++) {
    loop103: for (int v238 = 0; v238 < 64; v238++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop104: for (int v239 = 0; v239 < 8; v239++) {
        loop105: for (int v240 = 0; v240 < 2; v240++) {
          float v241 = v235[(v240 + (v238 * 2))][(v239 + (v237 * 8))];
          v236[v239][v240].write(v241);
        }
      }
    }
  }
  return ;
}

void node19(
  hls::stream<float> v242[4][2],
  float v243[128],
  float v244[64][128]
) {
  #pragma HLS array_partition variable=v243 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v244 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v244 cyclic dim=2 factor=2

  loop106: for (int v245 = 0; v245 < 16; v245++) {
    loop107: for (int v246 = 0; v246 < 64; v246++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop108: for (int v247 = 0; v247 < 4; v247++) {
        loop109: for (int v248 = 0; v248 < 2; v248++) {
          float v249 = v242[v247][v248].read();
          float v250 = v243[(v248 + (v246 * 2))];
          float v251 = v249 + v250;
          v244[(v247 + (v245 * 4))][(v248 + (v246 * 2))] = v251;
        }
      }
    }
  }
  return ;
}

void node20(
  hls::stream<float> v252[8][2],
  hls::stream<float> v253[4][8],
  hls::stream<float> v254[4][2],
  float v255
) {
  float v256[64][128];
  #pragma HLS array_partition variable=v256 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v256 cyclic dim=2 factor=2

  float v257[128][128];
  #pragma HLS array_partition variable=v257 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v257 cyclic dim=2 factor=2

  float v258[64][128];
  #pragma HLS array_partition variable=v258 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v258 cyclic dim=2 factor=8

  loop110: for (int v259 = 0; v259 < 16; v259++) {
    loop111: for (int v260 = 0; v260 < 16; v260++) {
      loop112: for (int v261 = 0; v261 < 64; v261++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop113: for (int v262 = 0; v262 < 4; v262++) {
          loop114: for (int v263 = 0; v263 < 8; v263++) {
            loop115: for (int v264 = 0; v264 < 2; v264++) {
              if ((v264 + (v261 * 2)) == 0) {
                float v265 = v253[v262][v263].read();
                v258[(v262 + (v259 * 4))][(v263 + (v260 * 8))] = v265;
              }
              float v266 = v258[(v262 + (v259 * 4))][(v263 + (v260 * 8))];
              if ((v262 + (v259 * 4)) == 0) {
                float v267 = v252[v263][v264].read();
                v257[(v263 + (v260 * 8))][(v264 + (v261 * 2))] = v267;
              }
              float v268 = v257[(v263 + (v260 * 8))][(v264 + (v261 * 2))];
              if ((v263 + (v260 * 8)) == 0) {
                v256[(v262 + (v259 * 4))][(v264 + (v261 * 2))] = v255;
              }
              float v269 = v256[(v262 + (v259 * 4))][(v264 + (v261 * 2))];
              float v270 = v266 * v268;
              float v271 = v269 + v270;
              v256[(v262 + (v259 * 4))][(v264 + (v261 * 2))] = v271;
              if (((v263 + (v260 * 8)) - 127) == 0) {
                float v272 = v256[(v262 + (v259 * 4))][(v264 + (v261 * 2))];
                v254[v262][v264].write(v272);
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
  float v273[128][128],
  hls::stream<float> v274[8][2]
) {
  #pragma HLS array_partition variable=v273 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v273 cyclic dim=2 factor=8

  loop116: for (int v275 = 0; v275 < 16; v275++) {
    loop117: for (int v276 = 0; v276 < 64; v276++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop118: for (int v277 = 0; v277 < 8; v277++) {
        loop119: for (int v278 = 0; v278 < 2; v278++) {
          float v279 = v273[(v278 + (v276 * 2))][(v277 + (v275 * 8))];
          v274[v277][v278].write(v279);
        }
      }
    }
  }
  return ;
}

void node22(
  hls::stream<float> v280[4][2],
  float v281[128],
  float v282[64][128]
) {
  #pragma HLS array_partition variable=v281 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v282 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v282 cyclic dim=2 factor=2

  loop120: for (int v283 = 0; v283 < 16; v283++) {
    loop121: for (int v284 = 0; v284 < 64; v284++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop122: for (int v285 = 0; v285 < 4; v285++) {
        loop123: for (int v286 = 0; v286 < 2; v286++) {
          float v287 = v280[v285][v286].read();
          float v288 = v281[(v286 + (v284 * 2))];
          float v289 = v287 + v288;
          v282[(v285 + (v283 * 4))][(v286 + (v284 * 2))] = v289;
        }
      }
    }
  }
  return ;
}

void node23(
  hls::stream<float> v290[8][2],
  hls::stream<float> v291[4][8],
  hls::stream<float> v292[4][2],
  float v293
) {
  float v294[64][128];
  #pragma HLS array_partition variable=v294 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v294 cyclic dim=2 factor=2

  float v295[128][128];
  #pragma HLS array_partition variable=v295 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v295 cyclic dim=2 factor=2

  float v296[64][128];
  #pragma HLS array_partition variable=v296 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v296 cyclic dim=2 factor=8

  loop124: for (int v297 = 0; v297 < 16; v297++) {
    loop125: for (int v298 = 0; v298 < 16; v298++) {
      loop126: for (int v299 = 0; v299 < 64; v299++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop127: for (int v300 = 0; v300 < 4; v300++) {
          loop128: for (int v301 = 0; v301 < 8; v301++) {
            loop129: for (int v302 = 0; v302 < 2; v302++) {
              if ((v302 + (v299 * 2)) == 0) {
                float v303 = v291[v300][v301].read();
                v296[(v300 + (v297 * 4))][(v301 + (v298 * 8))] = v303;
              }
              float v304 = v296[(v300 + (v297 * 4))][(v301 + (v298 * 8))];
              if ((v300 + (v297 * 4)) == 0) {
                float v305 = v290[v301][v302].read();
                v295[(v301 + (v298 * 8))][(v302 + (v299 * 2))] = v305;
              }
              float v306 = v295[(v301 + (v298 * 8))][(v302 + (v299 * 2))];
              if ((v301 + (v298 * 8)) == 0) {
                v294[(v300 + (v297 * 4))][(v302 + (v299 * 2))] = v293;
              }
              float v307 = v294[(v300 + (v297 * 4))][(v302 + (v299 * 2))];
              float v308 = v304 * v306;
              float v309 = v307 + v308;
              v294[(v300 + (v297 * 4))][(v302 + (v299 * 2))] = v309;
              if (((v301 + (v298 * 8)) - 127) == 0) {
                float v310 = v294[(v300 + (v297 * 4))][(v302 + (v299 * 2))];
                v292[v300][v302].write(v310);
              }
            }
          }
        }
      }
    }
  }
  return ;
}

void node24(
  float v311[128][128],
  hls::stream<float> v312[8][2]
) {
  #pragma HLS array_partition variable=v311 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v311 cyclic dim=2 factor=8

  loop130: for (int v313 = 0; v313 < 16; v313++) {
    loop131: for (int v314 = 0; v314 < 64; v314++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop132: for (int v315 = 0; v315 < 8; v315++) {
        loop133: for (int v316 = 0; v316 < 2; v316++) {
          float v317 = v311[(v316 + (v314 * 2))][(v315 + (v313 * 8))];
          v312[v315][v316].write(v317);
        }
      }
    }
  }
  return ;
}

void node25(
  float v318[1][64][128],
  hls::stream<float> v319[4][8],
  hls::stream<float> v320[4][8],
  hls::stream<float> v321[4][8]
) {
  #pragma HLS array_partition variable=v318 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v318 cyclic dim=3 factor=8

  loop134: for (int v322 = 0; v322 < 16; v322++) {
    loop135: for (int v323 = 0; v323 < 16; v323++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop136: for (int v324 = 0; v324 < 4; v324++) {
        loop137: for (int v325 = 0; v325 < 8; v325++) {
          float v326 = v318[0][(v324 + (v322 * 4))][(v325 + (v323 * 8))];
          v319[v324][v325].write(v326);
          v320[v324][v325].write(v326);
          v321[v324][v325].write(v326);
        }
      }
    }
  }
  return ;
}

void forward(
  float v327[1][64][128],
  float v328[128],
  float v329[128][128],
  float v330[128],
  float v331[128][128],
  float v332[128],
  float v333[128][128],
  float v334[128],
  float v335[128][128],
  float v336[1][64][128]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v327 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v327 cyclic dim=3 factor=8

  #pragma HLS array_partition variable=v328 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v329 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v329 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v330 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v331 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v331 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v332 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v333 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v333 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v334 cyclic dim=1 factor=32

  #pragma HLS array_partition variable=v335 cyclic dim=1 factor=32
  #pragma HLS array_partition variable=v335 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v336 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v336 cyclic dim=3 factor=32

  hls::stream<float> v337[2][32];
	#pragma HLS STREAM variable=v337 depth=128
  hls::stream<float> v338[2][32];
	#pragma HLS STREAM variable=v338 depth=128
  hls::stream<float> v339[2][32];
	#pragma HLS STREAM variable=v339 depth=256
  hls::stream<float> v340[2][2];
	#pragma HLS STREAM variable=v340 depth=2048
  float v341[64][8][16];
  #pragma HLS array_partition variable=v341 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v341 cyclic dim=2 factor=2
  #pragma HLS array_partition variable=v341 cyclic dim=3 factor=4

  float v342[8][64][16];
  #pragma HLS array_partition variable=v342 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v342 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v342 cyclic dim=3 factor=2

  float v343[8][64][16];
  #pragma HLS array_partition variable=v343 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v343 cyclic dim=2 factor=4
  #pragma HLS array_partition variable=v343 cyclic dim=3 factor=2

  hls::stream<float> v344[2][4][4];
	#pragma HLS STREAM variable=v344 depth=1024
  hls::stream<float> v345[2][4];
	#pragma HLS STREAM variable=v345 depth=64
  hls::stream<float> v346[2][4][4];
	#pragma HLS STREAM variable=v346 depth=1024
  hls::stream<float> v347[2][4][4];
	#pragma HLS STREAM variable=v347 depth=1024
  hls::stream<float> v348[2][4];
	#pragma HLS STREAM variable=v348 depth=64
  hls::stream<float> v349[2][4][4];
	#pragma HLS STREAM variable=v349 depth=1024
  hls::stream<float> v350[2][4][4];
	#pragma HLS STREAM variable=v350 depth=1024
  hls::stream<float> v351[2][4][4];
	#pragma HLS STREAM variable=v351 depth=1024
  hls::stream<float> v352[2][2][4];
	#pragma HLS STREAM variable=v352 depth=512
  hls::stream<float> v353[2][4][2];
	#pragma HLS STREAM variable=v353 depth=512
  float v354[64][128];
  #pragma HLS array_partition variable=v354 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v354 cyclic dim=2 factor=2

  hls::stream<float> v355[4][2];
	#pragma HLS STREAM variable=v355 depth=1024
  hls::stream<float> v356[8][2];
	#pragma HLS STREAM variable=v356 depth=1024
  float v357[64][128];
  #pragma HLS array_partition variable=v357 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v357 cyclic dim=2 factor=2

  hls::stream<float> v358[4][2];
	#pragma HLS STREAM variable=v358 depth=1024
  hls::stream<float> v359[8][2];
	#pragma HLS STREAM variable=v359 depth=1024
  float v360[64][128];
  #pragma HLS array_partition variable=v360 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v360 cyclic dim=2 factor=2

  hls::stream<float> v361[4][2];
	#pragma HLS STREAM variable=v361 depth=1024
  hls::stream<float> v362[8][2];
	#pragma HLS STREAM variable=v362 depth=1024
  hls::stream<float> v363[4][8];
	#pragma HLS STREAM variable=v363 depth=256
  hls::stream<float> v364[4][8];
	#pragma HLS STREAM variable=v364 depth=256
  hls::stream<float> v365[4][8];
	#pragma HLS STREAM variable=v365 depth=256
  node25(v327, v365, v364, v363);
  node24(v329, v362);
  node23(v362, v363, v361, 0.000000);
  node22(v361, v328, v360);
  node21(v331, v359);
  node20(v359, v364, v358, 0.000000);
  node19(v358, v330, v357);
  node18(v333, v356);
  node17(v356, v365, v355, 0.000000);
  node16(v355, v332, v354);
  node15(v360, v353);
  node14(v357, v352);
  node13(v353, v352, v351, 0.000000);
  node12(v351, v350, v349, 4.000000);
  node11(v350, v348, -INFINITY);
  node10(v349, v348, v347, v346);
  node9(v347, v345, 0.000000);
  node8(v346, v345, v344);
  node7(v354, v343);
  node6(v343, v344, v342, 0.000000);
  node5(v342, v341);
  node4(v341, v340);
  node3(v335, v339);
  node2(v339, v340, v338, 0.000000);
  node1(v338, v334, v337);
  node0(v337, v336);
  return ;
}

