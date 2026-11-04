
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
  hls::stream<float> v0[1][28],
  float v1[32][784]
) {	// L34
  #pragma HLS array_partition variable=v1 cyclic dim=2 factor=28

  loop0: for (int v2 = 0; v2 < 32; v2++) {	// L35
    loop1: for (int v3 = 0; v3 < 28; v3++) {	// L36
      #pragma HLS pipeline II=1
      loop2: for (int v4 = 0; v4 < 1; v4++) {	// L37
        loop3: for (int v5 = 0; v5 < 28; v5++) {	// L38
          float v6 = v0[v4][v5].read();	// L39
          v1[(v2 + v4)][(v5 + (v3 * 28))] = v6;	// L40
        }
      }
    }
  }
  return ;	// L45
}

void node1(
  hls::stream<float> v7[1][28],
  float v8[784],
  hls::stream<float> v9[1][28],
  float v10
) {	// L47
  #pragma HLS array_partition variable=v8 cyclic dim=1 factor=28

  loop4: for (int v11 = 0; v11 < 32; v11++) {	// L48
    loop5: for (int v12 = 0; v12 < 28; v12++) {	// L49
      #pragma HLS pipeline II=1
      loop6: for (int v13 = 0; v13 < 1; v13++) {	// L50
        loop7: for (int v14 = 0; v14 < 28; v14++) {	// L51
          float v15 = v7[v13][v14].read();	// L52
          float v16 = v8[(v14 + (v12 * 28))];	// L53
          float v17 = v15 + v16;	// L54
          float v18 = -(v17);	// L55
          float v19 = exp(v18);	// L56
          float v20 = v19 + v10;	// L57
          float v21 = v10 / v20;	// L58
          v9[v13][v14].write(v21);	// L59
        }
      }
    }
  }
  return ;	// L64
}

void node2(
  hls::stream<float> v22[1][8],
  float v23[256][784],
  hls::stream<float> v24[1][28],
  float v25
) {	// L66
  float v26[32][784];	// L67
  #pragma HLS array_partition variable=v26 cyclic dim=2 factor=28

  float v27[256][784];	// L68
  #pragma HLS array_partition variable=v27 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v27 cyclic dim=2 factor=28

  float v28[32][256];	// L69
  #pragma HLS array_partition variable=v28 cyclic dim=2 factor=8

  loop8: for (int v29 = 0; v29 < 32; v29++) {	// L70
    loop9: for (int v30 = 0; v30 < 32; v30++) {	// L71
      loop10: for (int v31 = 0; v31 < 28; v31++) {	// L72
        #pragma HLS pipeline II=1
        loop11: for (int v32 = 0; v32 < 1; v32++) {	// L73
          loop12: for (int v33 = 0; v33 < 8; v33++) {	// L74
            loop13: for (int v34 = 0; v34 < 28; v34++) {	// L75
              if ((v34 + (v31 * 28)) == 0) {	// L76
                float v35 = v22[v32][v33].read();	// L77
                v28[(v29 + v32)][(v33 + (v30 * 8))] = v35;	// L78
              }
              float v36 = v28[(v29 + v32)][(v33 + (v30 * 8))];	// L80
              if ((v29 + v32) == 0) {	// L81
                float v37 = v23[(v33 + (v30 * 8))][(v34 + (v31 * 28))];	// L82
                v27[(v33 + (v30 * 8))][(v34 + (v31 * 28))] = v37;	// L83
              }
              float v38 = v27[(v33 + (v30 * 8))][(v34 + (v31 * 28))];	// L85
              if ((v33 + (v30 * 8)) == 0) {	// L86
                v26[(v29 + v32)][(v34 + (v31 * 28))] = v25;	// L87
              }
              float v39 = v26[(v29 + v32)][(v34 + (v31 * 28))];	// L89
              float v40 = v36 * v38;	// L90
              float v41 = v39 + v40;	// L91
              v26[(v29 + v32)][(v34 + (v31 * 28))] = v41;	// L92
              if (((v33 + (v30 * 8)) - 255) == 0) {	// L93
                float v42 = v26[(v29 + v32)][(v34 + (v31 * 28))];	// L94
                v24[v32][v34].write(v42);	// L95
              }
            }
          }
        }
      }
    }
  }
  return ;	// L103
}

void node3(
  float v43[784][256],
  float v44[256][784]
) {	// L105
  #pragma HLS array_partition variable=v43 cyclic dim=1 factor=28
  #pragma HLS array_partition variable=v43 cyclic dim=2 factor=8

  loop14: for (int v45 = 0; v45 < 28; v45++) {	// L106
    loop15: for (int v46 = 0; v46 < 32; v46++) {	// L107
      #pragma HLS pipeline II=1
      loop16: for (int v47 = 0; v47 < 28; v47++) {	// L108
        loop17: for (int v48 = 0; v48 < 8; v48++) {	// L109
          float v49 = v43[(v47 + (v45 * 28))][(v48 + (v46 * 8))];	// L110
          v44[(v48 + (v46 * 8))][(v47 + (v45 * 28))] = v49;	// L111
        }
      }
    }
  }
  return ;	// L116
}

void node4(
  hls::stream<float> v50[1][8],
  float v51[256],
  hls::stream<float> v52[1][8],
  float v53
) {	// L118
  #pragma HLS array_partition variable=v51 cyclic dim=1 factor=8

  loop18: for (int v54 = 0; v54 < 32; v54++) {	// L119
    loop19: for (int v55 = 0; v55 < 32; v55++) {	// L120
      #pragma HLS pipeline II=1
      loop20: for (int v56 = 0; v56 < 1; v56++) {	// L121
        loop21: for (int v57 = 0; v57 < 8; v57++) {	// L122
          float v58 = v50[v56][v57].read();	// L123
          float v59 = v51[(v57 + (v55 * 8))];	// L124
          float v60 = v58 + v59;	// L125
          bool v61 = v60 > v53;	// L126
          float v62 = v61 ? (float)v60 : (float)v53;	// L127
          v52[v56][v57].write(v62);	// L128
        }
      }
    }
  }
  return ;	// L133
}

void node5(
  hls::stream<float> v63[1][4],
  hls::stream<float> v64[4][8],
  hls::stream<float> v65[1][8],
  float v66
) {	// L135
  float v67[32][256];	// L136
  #pragma HLS array_partition variable=v67 cyclic dim=2 factor=8

  float v68[128][256];	// L137
  #pragma HLS array_partition variable=v68 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v68 cyclic dim=2 factor=8

  float v69[32][128];	// L138
  #pragma HLS array_partition variable=v69 cyclic dim=2 factor=4

  loop22: for (int v70 = 0; v70 < 32; v70++) {	// L139
    loop23: for (int v71 = 0; v71 < 32; v71++) {	// L140
      loop24: for (int v72 = 0; v72 < 32; v72++) {	// L141
        #pragma HLS pipeline II=1
        loop25: for (int v73 = 0; v73 < 1; v73++) {	// L142
          loop26: for (int v74 = 0; v74 < 4; v74++) {	// L143
            loop27: for (int v75 = 0; v75 < 8; v75++) {	// L144
              if ((v75 + (v72 * 8)) == 0) {	// L145
                float v76 = v63[v73][v74].read();	// L146
                v69[(v70 + v73)][(v74 + (v71 * 4))] = v76;	// L147
              }
              float v77 = v69[(v70 + v73)][(v74 + (v71 * 4))];	// L149
              if ((v70 + v73) == 0) {	// L150
                float v78 = v64[v74][v75].read();	// L151
                v68[(v74 + (v71 * 4))][(v75 + (v72 * 8))] = v78;	// L152
              }
              float v79 = v68[(v74 + (v71 * 4))][(v75 + (v72 * 8))];	// L154
              if ((v74 + (v71 * 4)) == 0) {	// L155
                v67[(v70 + v73)][(v75 + (v72 * 8))] = v66;	// L156
              }
              float v80 = v67[(v70 + v73)][(v75 + (v72 * 8))];	// L158
              float v81 = v77 * v79;	// L159
              float v82 = v80 + v81;	// L160
              v67[(v70 + v73)][(v75 + (v72 * 8))] = v82;	// L161
              if (((v74 + (v71 * 4)) - 127) == 0) {	// L162
                float v83 = v67[(v70 + v73)][(v75 + (v72 * 8))];	// L163
                v65[v73][v75].write(v83);	// L164
              }
            }
          }
        }
      }
    }
  }
  return ;	// L172
}

void node6(
  float v84[256][128],
  hls::stream<float> v85[4][8]
) {	// L174
  #pragma HLS array_partition variable=v84 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v84 cyclic dim=2 factor=4

  loop28: for (int v86 = 0; v86 < 32; v86++) {	// L175
    loop29: for (int v87 = 0; v87 < 32; v87++) {	// L176
      #pragma HLS pipeline II=1
      loop30: for (int v88 = 0; v88 < 4; v88++) {	// L177
        loop31: for (int v89 = 0; v89 < 8; v89++) {	// L178
          float v90 = v84[(v89 + (v87 * 8))][(v88 + (v86 * 4))];	// L179
          v85[v88][v89].write(v90);	// L180
        }
      }
    }
  }
  return ;	// L185
}

void node7(
  hls::stream<float> v91[1][4],
  float v92[128],
  hls::stream<float> v93[1][4],
  float v94
) {	// L187
  #pragma HLS array_partition variable=v92 cyclic dim=1 factor=4

  loop32: for (int v95 = 0; v95 < 32; v95++) {	// L188
    loop33: for (int v96 = 0; v96 < 32; v96++) {	// L189
      #pragma HLS pipeline II=1
      loop34: for (int v97 = 0; v97 < 1; v97++) {	// L190
        loop35: for (int v98 = 0; v98 < 4; v98++) {	// L191
          float v99 = v91[v97][v98].read();	// L192
          float v100 = v92[(v98 + (v96 * 4))];	// L193
          float v101 = v99 + v100;	// L194
          bool v102 = v101 > v94;	// L195
          float v103 = v102 ? (float)v101 : (float)v94;	// L196
          v93[v97][v98].write(v103);	// L197
        }
      }
    }
  }
  return ;	// L202
}

void node8(
  hls::stream<float> v104[1][2],
  hls::stream<float> v105[2][4],
  hls::stream<float> v106[1][4],
  float v107
) {	// L204
  float v108[32][128];	// L205
  #pragma HLS array_partition variable=v108 cyclic dim=2 factor=4

  float v109[64][128];	// L206
  #pragma HLS array_partition variable=v109 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v109 cyclic dim=2 factor=4

  float v110[32][64];	// L207
  #pragma HLS array_partition variable=v110 cyclic dim=2 factor=2

  loop36: for (int v111 = 0; v111 < 32; v111++) {	// L208
    loop37: for (int v112 = 0; v112 < 32; v112++) {	// L209
      loop38: for (int v113 = 0; v113 < 32; v113++) {	// L210
        #pragma HLS pipeline II=1
        loop39: for (int v114 = 0; v114 < 1; v114++) {	// L211
          loop40: for (int v115 = 0; v115 < 2; v115++) {	// L212
            loop41: for (int v116 = 0; v116 < 4; v116++) {	// L213
              if ((v116 + (v113 * 4)) == 0) {	// L214
                float v117 = v104[v114][v115].read();	// L215
                v110[(v111 + v114)][(v115 + (v112 * 2))] = v117;	// L216
              }
              float v118 = v110[(v111 + v114)][(v115 + (v112 * 2))];	// L218
              if ((v111 + v114) == 0) {	// L219
                float v119 = v105[v115][v116].read();	// L220
                v109[(v115 + (v112 * 2))][(v116 + (v113 * 4))] = v119;	// L221
              }
              float v120 = v109[(v115 + (v112 * 2))][(v116 + (v113 * 4))];	// L223
              if ((v115 + (v112 * 2)) == 0) {	// L224
                v108[(v111 + v114)][(v116 + (v113 * 4))] = v107;	// L225
              }
              float v121 = v108[(v111 + v114)][(v116 + (v113 * 4))];	// L227
              float v122 = v118 * v120;	// L228
              float v123 = v121 + v122;	// L229
              v108[(v111 + v114)][(v116 + (v113 * 4))] = v123;	// L230
              if (((v115 + (v112 * 2)) - 63) == 0) {	// L231
                float v124 = v108[(v111 + v114)][(v116 + (v113 * 4))];	// L232
                v106[v114][v116].write(v124);	// L233
              }
            }
          }
        }
      }
    }
  }
  return ;	// L241
}

void node9(
  float v125[128][64],
  hls::stream<float> v126[2][4]
) {	// L243
  #pragma HLS array_partition variable=v125 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v125 cyclic dim=2 factor=2

  loop42: for (int v127 = 0; v127 < 32; v127++) {	// L244
    loop43: for (int v128 = 0; v128 < 32; v128++) {	// L245
      #pragma HLS pipeline II=1
      loop44: for (int v129 = 0; v129 < 2; v129++) {	// L246
        loop45: for (int v130 = 0; v130 < 4; v130++) {	// L247
          float v131 = v125[(v130 + (v128 * 4))][(v129 + (v127 * 2))];	// L248
          v126[v129][v130].write(v131);	// L249
        }
      }
    }
  }
  return ;	// L254
}

void node10(
  hls::stream<float> v132[1][2],
  float v133[64],
  hls::stream<float> v134[1][2]
) {	// L256
  #pragma HLS array_partition variable=v133 cyclic dim=1 factor=2

  loop46: for (int v135 = 0; v135 < 32; v135++) {	// L257
    loop47: for (int v136 = 0; v136 < 32; v136++) {	// L258
      #pragma HLS pipeline II=1
      loop48: for (int v137 = 0; v137 < 1; v137++) {	// L259
        loop49: for (int v138 = 0; v138 < 2; v138++) {	// L260
          float v139 = v132[v137][v138].read();	// L261
          float v140 = v133[(v138 + (v136 * 2))];	// L262
          float v141 = v139 + v140;	// L263
          v134[v137][v138].write(v141);	// L264
        }
      }
    }
  }
  return ;	// L269
}

void node11(
  hls::stream<float> v142[8][2],
  hls::stream<float> v143[1][8],
  hls::stream<float> v144[1][2],
  float v145
) {	// L271
  float v146[32][64];	// L272
  #pragma HLS array_partition variable=v146 cyclic dim=2 factor=2

  float v147[128][64];	// L273
  #pragma HLS array_partition variable=v147 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v147 cyclic dim=2 factor=2

  float v148[32][128];	// L274
  #pragma HLS array_partition variable=v148 cyclic dim=2 factor=8

  loop50: for (int v149 = 0; v149 < 32; v149++) {	// L275
    loop51: for (int v150 = 0; v150 < 16; v150++) {	// L276
      loop52: for (int v151 = 0; v151 < 32; v151++) {	// L277
        #pragma HLS pipeline II=1
        loop53: for (int v152 = 0; v152 < 1; v152++) {	// L278
          loop54: for (int v153 = 0; v153 < 8; v153++) {	// L279
            loop55: for (int v154 = 0; v154 < 2; v154++) {	// L280
              if ((v154 + (v151 * 2)) == 0) {	// L281
                float v155 = v143[v152][v153].read();	// L282
                v148[(v149 + v152)][(v153 + (v150 * 8))] = v155;	// L283
              }
              float v156 = v148[(v149 + v152)][(v153 + (v150 * 8))];	// L285
              if ((v149 + v152) == 0) {	// L286
                float v157 = v142[v153][v154].read();	// L287
                v147[(v153 + (v150 * 8))][(v154 + (v151 * 2))] = v157;	// L288
              }
              float v158 = v147[(v153 + (v150 * 8))][(v154 + (v151 * 2))];	// L290
              if ((v153 + (v150 * 8)) == 0) {	// L291
                v146[(v149 + v152)][(v154 + (v151 * 2))] = v145;	// L292
              }
              float v159 = v146[(v149 + v152)][(v154 + (v151 * 2))];	// L294
              float v160 = v156 * v158;	// L295
              float v161 = v159 + v160;	// L296
              v146[(v149 + v152)][(v154 + (v151 * 2))] = v161;	// L297
              if (((v153 + (v150 * 8)) - 127) == 0) {	// L298
                float v162 = v146[(v149 + v152)][(v154 + (v151 * 2))];	// L299
                v144[v152][v154].write(v162);	// L300
              }
            }
          }
        }
      }
    }
  }
  return ;	// L308
}

void node12(
  float v163[64][128],
  hls::stream<float> v164[8][2]
) {	// L310
  #pragma HLS array_partition variable=v163 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v163 cyclic dim=2 factor=8

  loop56: for (int v165 = 0; v165 < 16; v165++) {	// L311
    loop57: for (int v166 = 0; v166 < 32; v166++) {	// L312
      #pragma HLS pipeline II=1
      loop58: for (int v167 = 0; v167 < 8; v167++) {	// L313
        loop59: for (int v168 = 0; v168 < 2; v168++) {	// L314
          float v169 = v163[(v168 + (v166 * 2))][(v167 + (v165 * 8))];	// L315
          v164[v167][v168].write(v169);	// L316
        }
      }
    }
  }
  return ;	// L321
}

void node13(
  hls::stream<float> v170[1][8],
  float v171[128],
  hls::stream<float> v172[1][8],
  float v173
) {	// L323
  #pragma HLS array_partition variable=v171 cyclic dim=1 factor=8

  loop60: for (int v174 = 0; v174 < 32; v174++) {	// L324
    loop61: for (int v175 = 0; v175 < 16; v175++) {	// L325
      #pragma HLS pipeline II=1
      loop62: for (int v176 = 0; v176 < 1; v176++) {	// L326
        loop63: for (int v177 = 0; v177 < 8; v177++) {	// L327
          float v178 = v170[v176][v177].read();	// L328
          float v179 = v171[(v177 + (v175 * 8))];	// L329
          float v180 = v178 + v179;	// L330
          bool v181 = v180 > v173;	// L331
          float v182 = v181 ? (float)v180 : (float)v173;	// L332
          v172[v176][v177].write(v182);	// L333
        }
      }
    }
  }
  return ;	// L338
}

void node14(
  hls::stream<float> v183[1][4],
  hls::stream<float> v184[4][8],
  hls::stream<float> v185[1][8],
  float v186
) {	// L340
  float v187[32][128];	// L341
  #pragma HLS array_partition variable=v187 cyclic dim=2 factor=8

  float v188[256][128];	// L342
  #pragma HLS array_partition variable=v188 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v188 cyclic dim=2 factor=8

  float v189[32][256];	// L343
  #pragma HLS array_partition variable=v189 cyclic dim=2 factor=4

  loop64: for (int v190 = 0; v190 < 32; v190++) {	// L344
    loop65: for (int v191 = 0; v191 < 64; v191++) {	// L345
      loop66: for (int v192 = 0; v192 < 16; v192++) {	// L346
        #pragma HLS pipeline II=1
        loop67: for (int v193 = 0; v193 < 1; v193++) {	// L347
          loop68: for (int v194 = 0; v194 < 4; v194++) {	// L348
            loop69: for (int v195 = 0; v195 < 8; v195++) {	// L349
              if ((v195 + (v192 * 8)) == 0) {	// L350
                float v196 = v183[v193][v194].read();	// L351
                v189[(v190 + v193)][(v194 + (v191 * 4))] = v196;	// L352
              }
              float v197 = v189[(v190 + v193)][(v194 + (v191 * 4))];	// L354
              if ((v190 + v193) == 0) {	// L355
                float v198 = v184[v194][v195].read();	// L356
                v188[(v194 + (v191 * 4))][(v195 + (v192 * 8))] = v198;	// L357
              }
              float v199 = v188[(v194 + (v191 * 4))][(v195 + (v192 * 8))];	// L359
              if ((v194 + (v191 * 4)) == 0) {	// L360
                v187[(v190 + v193)][(v195 + (v192 * 8))] = v186;	// L361
              }
              float v200 = v187[(v190 + v193)][(v195 + (v192 * 8))];	// L363
              float v201 = v197 * v199;	// L364
              float v202 = v200 + v201;	// L365
              v187[(v190 + v193)][(v195 + (v192 * 8))] = v202;	// L366
              if (((v194 + (v191 * 4)) - 255) == 0) {	// L367
                float v203 = v187[(v190 + v193)][(v195 + (v192 * 8))];	// L368
                v185[v193][v195].write(v203);	// L369
              }
            }
          }
        }
      }
    }
  }
  return ;	// L377
}

void node15(
  float v204[128][256],
  hls::stream<float> v205[4][8]
) {	// L379
  #pragma HLS array_partition variable=v204 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v204 cyclic dim=2 factor=4

  loop70: for (int v206 = 0; v206 < 64; v206++) {	// L380
    loop71: for (int v207 = 0; v207 < 16; v207++) {	// L381
      #pragma HLS pipeline II=1
      loop72: for (int v208 = 0; v208 < 4; v208++) {	// L382
        loop73: for (int v209 = 0; v209 < 8; v209++) {	// L383
          float v210 = v204[(v209 + (v207 * 8))][(v208 + (v206 * 4))];	// L384
          v205[v208][v209].write(v210);	// L385
        }
      }
    }
  }
  return ;	// L390
}

void node16(
  hls::stream<float> v211[1][4],
  float v212[256],
  hls::stream<float> v213[1][4],
  float v214
) {	// L392
  #pragma HLS array_partition variable=v212 cyclic dim=1 factor=4

  loop74: for (int v215 = 0; v215 < 32; v215++) {	// L393
    loop75: for (int v216 = 0; v216 < 64; v216++) {	// L394
      #pragma HLS pipeline II=1
      loop76: for (int v217 = 0; v217 < 1; v217++) {	// L395
        loop77: for (int v218 = 0; v218 < 4; v218++) {	// L396
          float v219 = v211[v217][v218].read();	// L397
          float v220 = v212[(v218 + (v216 * 4))];	// L398
          float v221 = v219 + v220;	// L399
          bool v222 = v221 > v214;	// L400
          float v223 = v222 ? (float)v221 : (float)v214;	// L401
          v213[v217][v218].write(v223);	// L402
        }
      }
    }
  }
  return ;	// L407
}

void node17(
  hls::stream<float> v224[49][4],
  float v225[32][784],
  hls::stream<float> v226[1][4],
  float v227
) {	// L409
  #pragma HLS array_partition variable=v225 cyclic dim=2 factor=49

  float v228[32][256];	// L410
  #pragma HLS array_partition variable=v228 cyclic dim=2 factor=4

  float v229[784][256];	// L411
  #pragma HLS array_partition variable=v229 cyclic dim=1 factor=49
  #pragma HLS array_partition variable=v229 cyclic dim=2 factor=4

  loop78: for (int v230 = 0; v230 < 32; v230++) {	// L412
    loop79: for (int v231 = 0; v231 < 16; v231++) {	// L413
      loop80: for (int v232 = 0; v232 < 64; v232++) {	// L414
        #pragma HLS pipeline II=1
        loop81: for (int v233 = 0; v233 < 1; v233++) {	// L415
          loop82: for (int v234 = 0; v234 < 49; v234++) {	// L416
            loop83: for (int v235 = 0; v235 < 4; v235++) {	// L417
              float v236 = v225[(v230 + v233)][(v234 + (v231 * 49))];	// L418
              if ((v230 + v233) == 0) {	// L419
                float v237 = v224[v234][v235].read();	// L420
                v229[(v234 + (v231 * 49))][(v235 + (v232 * 4))] = v237;	// L421
              }
              float v238 = v229[(v234 + (v231 * 49))][(v235 + (v232 * 4))];	// L423
              if ((v234 + (v231 * 49)) == 0) {	// L424
                v228[(v230 + v233)][(v235 + (v232 * 4))] = v227;	// L425
              }
              float v239 = v228[(v230 + v233)][(v235 + (v232 * 4))];	// L427
              float v240 = v236 * v238;	// L428
              float v241 = v239 + v240;	// L429
              v228[(v230 + v233)][(v235 + (v232 * 4))] = v241;	// L430
              if (((v234 + (v231 * 49)) - 783) == 0) {	// L431
                float v242 = v228[(v230 + v233)][(v235 + (v232 * 4))];	// L432
                v226[v233][v235].write(v242);	// L433
              }
            }
          }
        }
      }
    }
  }
  return ;	// L441
}

void node18(
  float v243[256][784],
  hls::stream<float> v244[49][4]
) {	// L443
  #pragma HLS array_partition variable=v243 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v243 cyclic dim=2 factor=49

  loop84: for (int v245 = 0; v245 < 16; v245++) {	// L444
    loop85: for (int v246 = 0; v246 < 64; v246++) {	// L445
      #pragma HLS pipeline II=1
      loop86: for (int v247 = 0; v247 < 49; v247++) {	// L446
        loop87: for (int v248 = 0; v248 < 4; v248++) {	// L447
          float v249 = v243[(v248 + (v246 * 4))][(v247 + (v245 * 49))];	// L448
          v244[v247][v248].write(v249);	// L449
        }
      }
    }
  }
  return ;	// L454
}

void forward(
  float v250[32][784],
  float v251[256],
  float v252[256][784],
  float v253[128],
  float v254[128][256],
  float v255[64],
  float v256[64][128],
  float v257[128],
  float v258[128][64],
  float v259[256],
  float v260[256][128],
  float v261[784],
  float v262[784][256],
  float v263[32][784]
) {	// L456
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v250 cyclic dim=2 factor=49

  #pragma HLS array_partition variable=v251 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v252 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v252 cyclic dim=2 factor=49

  #pragma HLS array_partition variable=v253 cyclic dim=1 factor=8

  #pragma HLS array_partition variable=v254 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v254 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v255 cyclic dim=1 factor=2

  #pragma HLS array_partition variable=v256 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v256 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v257 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v258 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v258 cyclic dim=2 factor=2

  #pragma HLS array_partition variable=v259 cyclic dim=1 factor=8

  #pragma HLS array_partition variable=v260 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v260 cyclic dim=2 factor=4

  #pragma HLS array_partition variable=v261 cyclic dim=1 factor=28

  #pragma HLS array_partition variable=v262 cyclic dim=1 factor=28
  #pragma HLS array_partition variable=v262 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v263 cyclic dim=2 factor=28

  hls::stream<float> v264[1][28];
	#pragma HLS STREAM variable=v264 depth=896	// L459
  hls::stream<float> v265[1][28];
	#pragma HLS STREAM variable=v265 depth=896	// L460
  float v266[256][784];	// L461
  hls::stream<float> v267[1][8];
	#pragma HLS STREAM variable=v267 depth=1024	// L462
  hls::stream<float> v268[1][8];
	#pragma HLS STREAM variable=v268 depth=1024	// L463
  hls::stream<float> v269[4][8];
	#pragma HLS STREAM variable=v269 depth=1024	// L464
  hls::stream<float> v270[1][4];
	#pragma HLS STREAM variable=v270 depth=1024	// L465
  hls::stream<float> v271[1][4];
	#pragma HLS STREAM variable=v271 depth=1024	// L466
  hls::stream<float> v272[2][4];
	#pragma HLS STREAM variable=v272 depth=1024	// L467
  hls::stream<float> v273[1][2];
	#pragma HLS STREAM variable=v273 depth=1024	// L468
  hls::stream<float> v274[1][2];
	#pragma HLS STREAM variable=v274 depth=1024	// L469
  hls::stream<float> v275[8][2];
	#pragma HLS STREAM variable=v275 depth=512	// L470
  hls::stream<float> v276[1][8];
	#pragma HLS STREAM variable=v276 depth=512	// L471
  hls::stream<float> v277[1][8];
	#pragma HLS STREAM variable=v277 depth=512	// L472
  hls::stream<float> v278[4][8];
	#pragma HLS STREAM variable=v278 depth=1024	// L473
  hls::stream<float> v279[1][4];
	#pragma HLS STREAM variable=v279 depth=2048	// L474
  hls::stream<float> v280[1][4];
	#pragma HLS STREAM variable=v280 depth=2048	// L475
  hls::stream<float> v281[49][4];
	#pragma HLS STREAM variable=v281 depth=1024	// L476
  node18(v252, v281);	// L477
  node17(v281, v250, v280, 0.000000);	// L478
  node16(v280, v251, v279, 0.000000);	// L479
  node15(v254, v278);	// L480
  node14(v279, v278, v277, 0.000000);	// L481
  node13(v277, v253, v276, 0.000000);	// L482
  node12(v256, v275);	// L483
  node11(v275, v276, v274, 0.000000);	// L484
  node10(v274, v255, v273);	// L485
  node9(v258, v272);	// L486
  node8(v273, v272, v271, 0.000000);	// L487
  node7(v271, v257, v270, 0.000000);	// L488
  node6(v260, v269);	// L489
  node5(v270, v269, v268, 0.000000);	// L490
  node4(v268, v259, v267, 0.000000);	// L491
  node3(v262, v266);	// L492
  node2(v267, v266, v265, 0.000000);	// L493
  node1(v265, v261, v264, 1.000000);	// L494
  node0(v264, v263);	// L495
  return ;	// L496
}

