
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
  hls::stream<float> v0[1][49],
  float v1[32][784]
) {	// L15
  loop0: for (int v2 = 0; v2 < 32; v2++) {	// L16
    loop1: for (int v3 = 0; v3 < 16; v3++) {	// L17
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 1; v4++) {	// L18
        loop3: for (int v5 = 0; v5 < 49; v5++) {	// L19
          float v6 = v0[v4][v5].read();	// L20
          v1[(v2 + v4)][(v5 + (v3 * 49))] = v6;	// L21
        }
      }
    }
  }
  return ;	// L26
}

void node1(
  hls::stream<float> v7[1][49],
  float v8[784],
  hls::stream<float> v9[1][49],
  float v10
) {	// L28
  loop4: for (int v11 = 0; v11 < 32; v11++) {	// L29
    loop5: for (int v12 = 0; v12 < 16; v12++) {	// L30
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v13 = 0; v13 < 1; v13++) {	// L31
        loop7: for (int v14 = 0; v14 < 49; v14++) {	// L32
          float v15 = v7[v13][v14].read();	// L33
          float v16 = v8[(v14 + (v12 * 49))];	// L34
          float v17 = v15 + v16;	// L35
          float v18 = -(v17);	// L36
          float v19 = exp(v18);	// L37
          float v20 = v19 + v10;	// L38
          float v21 = v10 / v20;	// L39
          v9[v13][v14].write(v21);	// L40
        }
      }
    }
  }
  return ;	// L45
}

void node2(
  hls::stream<float> v22[1][4],
  hls::stream<float> v23[4][49],
  hls::stream<float> v24[1][49],
  float v25
) {	// L47
  float v26[32][784];	// L48
  float v27[256][784];	// L49
  float v28[32][256];	// L50
  loop8: for (int v29 = 0; v29 < 32; v29++) {	// L51
    loop9: for (int v30 = 0; v30 < 16; v30++) {	// L52
      loop10: for (int v31 = 0; v31 < 64; v31++) {	// L53
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v32 = 0; v32 < 1; v32++) {	// L54
          loop12: for (int v33 = 0; v33 < 49; v33++) {	// L55
            loop13: for (int v34 = 0; v34 < 4; v34++) {	// L56
              if ((v33 + (v30 * 49)) == 0) {	// L57
                float v35 = v22[v32][v34].read();	// L58
                v28[(v29 + v32)][(v34 + (v31 * 4))] = v35;	// L59
              }
              float v36 = v28[(v29 + v32)][(v34 + (v31 * 4))];	// L61
              if ((v29 + v32) == 0) {	// L62
                float v37 = v23[v34][v33].read();	// L63
                v27[(v34 + (v31 * 4))][(v33 + (v30 * 49))] = v37;	// L64
              }
              float v38 = v27[(v34 + (v31 * 4))][(v33 + (v30 * 49))];	// L66
              if ((v34 + (v31 * 4)) == 0) {	// L67
                v26[(v29 + v32)][(v33 + (v30 * 49))] = v25;	// L68
              }
              float v39 = v26[(v29 + v32)][(v33 + (v30 * 49))];	// L70
              float v40 = v36 * v38;	// L71
              float v41 = v39 + v40;	// L72
              v26[(v29 + v32)][(v33 + (v30 * 49))] = v41;	// L73
              if (((v34 + (v31 * 4)) - 255) == 0) {	// L74
                float v42 = v26[(v29 + v32)][(v33 + (v30 * 49))];	// L75
                v24[v32][v33].write(v42);	// L76
              }
            }
          }
        }
      }
    }
  }
  return ;	// L84
}

void node3(
  float v43[784][256],
  hls::stream<float> v44[4][49]
) {	// L86
  loop14: for (int v45 = 0; v45 < 16; v45++) {	// L87
    loop15: for (int v46 = 0; v46 < 64; v46++) {	// L88
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v47 = 0; v47 < 49; v47++) {	// L89
        loop17: for (int v48 = 0; v48 < 4; v48++) {	// L90
          float v49 = v43[(v47 + (v45 * 49))][(v48 + (v46 * 4))];	// L91
          v44[v48][v47].write(v49);	// L92
        }
      }
    }
  }
  return ;	// L97
}

void node4(
  hls::stream<float> v50[1][4],
  float v51[256],
  hls::stream<float> v52[1][4],
  float v53
) {	// L99
  loop18: for (int v54 = 0; v54 < 32; v54++) {	// L100
    loop19: for (int v55 = 0; v55 < 64; v55++) {	// L101
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop20: for (int v56 = 0; v56 < 1; v56++) {	// L102
        loop21: for (int v57 = 0; v57 < 4; v57++) {	// L103
          float v58 = v50[v56][v57].read();	// L104
          float v59 = v51[(v57 + (v55 * 4))];	// L105
          float v60 = v58 + v59;	// L106
          bool v61 = v60 > v53;	// L107
          float v62 = v61 ? (float)v60 : (float)v53;	// L108
          v52[v56][v57].write(v62);	// L109
        }
      }
    }
  }
  return ;	// L114
}

void node5(
  hls::stream<float> v63[1][8],
  hls::stream<float> v64[8][4],
  hls::stream<float> v65[1][4],
  float v66
) {	// L116
  float v67[32][256];	// L117
  float v68[128][256];	// L118
  float v69[32][128];	// L119
  loop22: for (int v70 = 0; v70 < 32; v70++) {	// L120
    loop23: for (int v71 = 0; v71 < 64; v71++) {	// L121
      loop24: for (int v72 = 0; v72 < 16; v72++) {	// L122
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v73 = 0; v73 < 1; v73++) {	// L123
          loop26: for (int v74 = 0; v74 < 4; v74++) {	// L124
            loop27: for (int v75 = 0; v75 < 8; v75++) {	// L125
              if ((v74 + (v71 * 4)) == 0) {	// L126
                float v76 = v63[v73][v75].read();	// L127
                v69[(v70 + v73)][(v75 + (v72 * 8))] = v76;	// L128
              }
              float v77 = v69[(v70 + v73)][(v75 + (v72 * 8))];	// L130
              if ((v70 + v73) == 0) {	// L131
                float v78 = v64[v75][v74].read();	// L132
                v68[(v75 + (v72 * 8))][(v74 + (v71 * 4))] = v78;	// L133
              }
              float v79 = v68[(v75 + (v72 * 8))][(v74 + (v71 * 4))];	// L135
              if ((v75 + (v72 * 8)) == 0) {	// L136
                v67[(v70 + v73)][(v74 + (v71 * 4))] = v66;	// L137
              }
              float v80 = v67[(v70 + v73)][(v74 + (v71 * 4))];	// L139
              float v81 = v77 * v79;	// L140
              float v82 = v80 + v81;	// L141
              v67[(v70 + v73)][(v74 + (v71 * 4))] = v82;	// L142
              if (((v75 + (v72 * 8)) - 127) == 0) {	// L143
                float v83 = v67[(v70 + v73)][(v74 + (v71 * 4))];	// L144
                v65[v73][v74].write(v83);	// L145
              }
            }
          }
        }
      }
    }
  }
  return ;	// L153
}

void node6(
  float v84[256][128],
  hls::stream<float> v85[8][4]
) {	// L155
  loop28: for (int v86 = 0; v86 < 64; v86++) {	// L156
    loop29: for (int v87 = 0; v87 < 16; v87++) {	// L157
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop30: for (int v88 = 0; v88 < 4; v88++) {	// L158
        loop31: for (int v89 = 0; v89 < 8; v89++) {	// L159
          float v90 = v84[(v88 + (v86 * 4))][(v89 + (v87 * 8))];	// L160
          v85[v89][v88].write(v90);	// L161
        }
      }
    }
  }
  return ;	// L166
}

void node7(
  hls::stream<float> v91[1][8],
  float v92[128],
  hls::stream<float> v93[1][8],
  float v94
) {	// L168
  loop32: for (int v95 = 0; v95 < 32; v95++) {	// L169
    loop33: for (int v96 = 0; v96 < 16; v96++) {	// L170
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop34: for (int v97 = 0; v97 < 1; v97++) {	// L171
        loop35: for (int v98 = 0; v98 < 8; v98++) {	// L172
          float v99 = v91[v97][v98].read();	// L173
          float v100 = v92[(v98 + (v96 * 8))];	// L174
          float v101 = v99 + v100;	// L175
          bool v102 = v101 > v94;	// L176
          float v103 = v102 ? (float)v101 : (float)v94;	// L177
          v93[v97][v98].write(v103);	// L178
        }
      }
    }
  }
  return ;	// L183
}

void node8(
  hls::stream<float> v104[2][8],
  hls::stream<float> v105[1][2],
  hls::stream<float> v106[1][8],
  float v107
) {	// L185
  float v108[32][128];	// L186
  float v109[64][128];	// L187
  float v110[32][64];	// L188
  loop36: for (int v111 = 0; v111 < 32; v111++) {	// L189
    loop37: for (int v112 = 0; v112 < 16; v112++) {	// L190
      loop38: for (int v113 = 0; v113 < 32; v113++) {	// L191
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v114 = 0; v114 < 1; v114++) {	// L192
          loop40: for (int v115 = 0; v115 < 8; v115++) {	// L193
            loop41: for (int v116 = 0; v116 < 2; v116++) {	// L194
              if ((v115 + (v112 * 8)) == 0) {	// L195
                float v117 = v105[v114][v116].read();	// L196
                v110[(v111 + v114)][(v116 + (v113 * 2))] = v117;	// L197
              }
              float v118 = v110[(v111 + v114)][(v116 + (v113 * 2))];	// L199
              if ((v111 + v114) == 0) {	// L200
                float v119 = v104[v116][v115].read();	// L201
                v109[(v116 + (v113 * 2))][(v115 + (v112 * 8))] = v119;	// L202
              }
              float v120 = v109[(v116 + (v113 * 2))][(v115 + (v112 * 8))];	// L204
              if ((v116 + (v113 * 2)) == 0) {	// L205
                v108[(v111 + v114)][(v115 + (v112 * 8))] = v107;	// L206
              }
              float v121 = v108[(v111 + v114)][(v115 + (v112 * 8))];	// L208
              float v122 = v118 * v120;	// L209
              float v123 = v121 + v122;	// L210
              v108[(v111 + v114)][(v115 + (v112 * 8))] = v123;	// L211
              if (((v116 + (v113 * 2)) - 63) == 0) {	// L212
                float v124 = v108[(v111 + v114)][(v115 + (v112 * 8))];	// L213
                v106[v114][v115].write(v124);	// L214
              }
            }
          }
        }
      }
    }
  }
  return ;	// L222
}

void node9(
  float v125[128][64],
  hls::stream<float> v126[2][8]
) {	// L224
  loop42: for (int v127 = 0; v127 < 16; v127++) {	// L225
    loop43: for (int v128 = 0; v128 < 32; v128++) {	// L226
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop44: for (int v129 = 0; v129 < 8; v129++) {	// L227
        loop45: for (int v130 = 0; v130 < 2; v130++) {	// L228
          float v131 = v125[(v129 + (v127 * 8))][(v130 + (v128 * 2))];	// L229
          v126[v130][v129].write(v131);	// L230
        }
      }
    }
  }
  return ;	// L235
}

void node10(
  hls::stream<float> v132[1][2],
  float v133[64],
  hls::stream<float> v134[1][2]
) {	// L237
  loop46: for (int v135 = 0; v135 < 32; v135++) {	// L238
    loop47: for (int v136 = 0; v136 < 32; v136++) {	// L239
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop48: for (int v137 = 0; v137 < 1; v137++) {	// L240
        loop49: for (int v138 = 0; v138 < 2; v138++) {	// L241
          float v139 = v132[v137][v138].read();	// L242
          float v140 = v133[(v138 + (v136 * 2))];	// L243
          float v141 = v139 + v140;	// L244
          v134[v137][v138].write(v141);	// L245
        }
      }
    }
  }
  return ;	// L250
}

void node11(
  hls::stream<float> v142[1][4],
  hls::stream<float> v143[4][2],
  hls::stream<float> v144[1][2],
  float v145
) {	// L252
  float v146[32][64];	// L253
  float v147[128][64];	// L254
  float v148[32][128];	// L255
  loop50: for (int v149 = 0; v149 < 32; v149++) {	// L256
    loop51: for (int v150 = 0; v150 < 32; v150++) {	// L257
      loop52: for (int v151 = 0; v151 < 32; v151++) {	// L258
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop53: for (int v152 = 0; v152 < 1; v152++) {	// L259
          loop54: for (int v153 = 0; v153 < 2; v153++) {	// L260
            loop55: for (int v154 = 0; v154 < 4; v154++) {	// L261
              if ((v153 + (v150 * 2)) == 0) {	// L262
                float v155 = v142[v152][v154].read();	// L263
                v148[(v149 + v152)][(v154 + (v151 * 4))] = v155;	// L264
              }
              float v156 = v148[(v149 + v152)][(v154 + (v151 * 4))];	// L266
              if ((v149 + v152) == 0) {	// L267
                float v157 = v143[v154][v153].read();	// L268
                v147[(v154 + (v151 * 4))][(v153 + (v150 * 2))] = v157;	// L269
              }
              float v158 = v147[(v154 + (v151 * 4))][(v153 + (v150 * 2))];	// L271
              if ((v154 + (v151 * 4)) == 0) {	// L272
                v146[(v149 + v152)][(v153 + (v150 * 2))] = v145;	// L273
              }
              float v159 = v146[(v149 + v152)][(v153 + (v150 * 2))];	// L275
              float v160 = v156 * v158;	// L276
              float v161 = v159 + v160;	// L277
              v146[(v149 + v152)][(v153 + (v150 * 2))] = v161;	// L278
              if (((v154 + (v151 * 4)) - 127) == 0) {	// L279
                float v162 = v146[(v149 + v152)][(v153 + (v150 * 2))];	// L280
                v144[v152][v153].write(v162);	// L281
              }
            }
          }
        }
      }
    }
  }
  return ;	// L289
}

void node12(
  float v163[64][128],
  hls::stream<float> v164[4][2]
) {	// L291
  loop56: for (int v165 = 0; v165 < 32; v165++) {	// L292
    loop57: for (int v166 = 0; v166 < 32; v166++) {	// L293
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop58: for (int v167 = 0; v167 < 2; v167++) {	// L294
        loop59: for (int v168 = 0; v168 < 4; v168++) {	// L295
          float v169 = v163[(v167 + (v165 * 2))][(v168 + (v166 * 4))];	// L296
          v164[v168][v167].write(v169);	// L297
        }
      }
    }
  }
  return ;	// L302
}

void node13(
  hls::stream<float> v170[1][4],
  float v171[128],
  hls::stream<float> v172[1][4],
  float v173
) {	// L304
  loop60: for (int v174 = 0; v174 < 32; v174++) {	// L305
    loop61: for (int v175 = 0; v175 < 32; v175++) {	// L306
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop62: for (int v176 = 0; v176 < 1; v176++) {	// L307
        loop63: for (int v177 = 0; v177 < 4; v177++) {	// L308
          float v178 = v170[v176][v177].read();	// L309
          float v179 = v171[(v177 + (v175 * 4))];	// L310
          float v180 = v178 + v179;	// L311
          bool v181 = v180 > v173;	// L312
          float v182 = v181 ? (float)v180 : (float)v173;	// L313
          v172[v176][v177].write(v182);	// L314
        }
      }
    }
  }
  return ;	// L319
}

void node14(
  hls::stream<float> v183[8][4],
  hls::stream<float> v184[1][8],
  hls::stream<float> v185[1][4],
  float v186
) {	// L321
  float v187[32][128];	// L322
  float v188[256][128];	// L323
  float v189[32][256];	// L324
  loop64: for (int v190 = 0; v190 < 32; v190++) {	// L325
    loop65: for (int v191 = 0; v191 < 32; v191++) {	// L326
      loop66: for (int v192 = 0; v192 < 32; v192++) {	// L327
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop67: for (int v193 = 0; v193 < 1; v193++) {	// L328
          loop68: for (int v194 = 0; v194 < 4; v194++) {	// L329
            loop69: for (int v195 = 0; v195 < 8; v195++) {	// L330
              if ((v194 + (v191 * 4)) == 0) {	// L331
                float v196 = v184[v193][v195].read();	// L332
                v189[(v190 + v193)][(v195 + (v192 * 8))] = v196;	// L333
              }
              float v197 = v189[(v190 + v193)][(v195 + (v192 * 8))];	// L335
              if ((v190 + v193) == 0) {	// L336
                float v198 = v183[v195][v194].read();	// L337
                v188[(v195 + (v192 * 8))][(v194 + (v191 * 4))] = v198;	// L338
              }
              float v199 = v188[(v195 + (v192 * 8))][(v194 + (v191 * 4))];	// L340
              if ((v195 + (v192 * 8)) == 0) {	// L341
                v187[(v190 + v193)][(v194 + (v191 * 4))] = v186;	// L342
              }
              float v200 = v187[(v190 + v193)][(v194 + (v191 * 4))];	// L344
              float v201 = v197 * v199;	// L345
              float v202 = v200 + v201;	// L346
              v187[(v190 + v193)][(v194 + (v191 * 4))] = v202;	// L347
              if (((v195 + (v192 * 8)) - 255) == 0) {	// L348
                float v203 = v187[(v190 + v193)][(v194 + (v191 * 4))];	// L349
                v185[v193][v194].write(v203);	// L350
              }
            }
          }
        }
      }
    }
  }
  return ;	// L358
}

void node15(
  float v204[128][256],
  hls::stream<float> v205[8][4]
) {	// L360
  loop70: for (int v206 = 0; v206 < 32; v206++) {	// L361
    loop71: for (int v207 = 0; v207 < 32; v207++) {	// L362
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop72: for (int v208 = 0; v208 < 4; v208++) {	// L363
        loop73: for (int v209 = 0; v209 < 8; v209++) {	// L364
          float v210 = v204[(v208 + (v206 * 4))][(v209 + (v207 * 8))];	// L365
          v205[v209][v208].write(v210);	// L366
        }
      }
    }
  }
  return ;	// L371
}

void node16(
  hls::stream<float> v211[1][8],
  float v212[256],
  hls::stream<float> v213[1][8],
  float v214
) {	// L373
  loop74: for (int v215 = 0; v215 < 32; v215++) {	// L374
    loop75: for (int v216 = 0; v216 < 32; v216++) {	// L375
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop76: for (int v217 = 0; v217 < 1; v217++) {	// L376
        loop77: for (int v218 = 0; v218 < 8; v218++) {	// L377
          float v219 = v211[v217][v218].read();	// L378
          float v220 = v212[(v218 + (v216 * 8))];	// L379
          float v221 = v219 + v220;	// L380
          bool v222 = v221 > v214;	// L381
          float v223 = v222 ? (float)v221 : (float)v214;	// L382
          v213[v217][v218].write(v223);	// L383
        }
      }
    }
  }
  return ;	// L388
}

void node17(
  float v224[32][784],
  hls::stream<float> v225[28][8],
  hls::stream<float> v226[1][8],
  float v227
) {	// L390
  float v228[32][256];	// L391
  float v229[784][256];	// L392
  loop78: for (int v230 = 0; v230 < 32; v230++) {	// L393
    loop79: for (int v231 = 0; v231 < 32; v231++) {	// L394
      loop80: for (int v232 = 0; v232 < 28; v232++) {	// L395
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop81: for (int v233 = 0; v233 < 1; v233++) {	// L396
          loop82: for (int v234 = 0; v234 < 8; v234++) {	// L397
            loop83: for (int v235 = 0; v235 < 28; v235++) {	// L398
              float v236 = v224[(v230 + v233)][(v235 + (v232 * 28))];	// L399
              if ((v230 + v233) == 0) {	// L400
                float v237 = v225[v235][v234].read();	// L401
                v229[(v235 + (v232 * 28))][(v234 + (v231 * 8))] = v237;	// L402
              }
              float v238 = v229[(v235 + (v232 * 28))][(v234 + (v231 * 8))];	// L404
              if ((v235 + (v232 * 28)) == 0) {	// L405
                v228[(v230 + v233)][(v234 + (v231 * 8))] = v227;	// L406
              }
              float v239 = v228[(v230 + v233)][(v234 + (v231 * 8))];	// L408
              float v240 = v236 * v238;	// L409
              float v241 = v239 + v240;	// L410
              v228[(v230 + v233)][(v234 + (v231 * 8))] = v241;	// L411
              if (((v235 + (v232 * 28)) - 783) == 0) {	// L412
                float v242 = v228[(v230 + v233)][(v234 + (v231 * 8))];	// L413
                v226[v233][v234].write(v242);	// L414
              }
            }
          }
        }
      }
    }
  }
  return ;	// L422
}

void node18(
  float v243[256][784],
  hls::stream<float> v244[28][8]
) {	// L424
  loop84: for (int v245 = 0; v245 < 32; v245++) {	// L425
    loop85: for (int v246 = 0; v246 < 28; v246++) {	// L426
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop86: for (int v247 = 0; v247 < 8; v247++) {	// L427
        loop87: for (int v248 = 0; v248 < 28; v248++) {	// L428
          float v249 = v243[(v247 + (v245 * 8))][(v248 + (v246 * 28))];	// L429
          v244[v248][v247].write(v249);	// L430
        }
      }
    }
  }
  return ;	// L435
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
) {	// L437
	#pragma HLS DATAFLOW
  hls::stream<float> v264[1][49];
	#pragma HLS STREAM variable=v264 depth=512	// L440
  hls::stream<float> v265[1][49];
	#pragma HLS STREAM variable=v265 depth=512	// L441
  hls::stream<float> v266[4][49];
	#pragma HLS STREAM variable=v266 depth=1024	// L442
  hls::stream<float> v267[1][4];
	#pragma HLS STREAM variable=v267 depth=2048	// L443
  hls::stream<float> v268[1][4];
	#pragma HLS STREAM variable=v268 depth=2048	// L444
  hls::stream<float> v269[8][4];
	#pragma HLS STREAM variable=v269 depth=1024	// L445
  hls::stream<float> v270[1][8];
	#pragma HLS STREAM variable=v270 depth=512	// L446
  hls::stream<float> v271[1][8];
	#pragma HLS STREAM variable=v271 depth=512	// L447
  hls::stream<float> v272[2][8];
	#pragma HLS STREAM variable=v272 depth=512	// L448
  hls::stream<float> v273[1][2];
	#pragma HLS STREAM variable=v273 depth=1024	// L449
  hls::stream<float> v274[1][2];
	#pragma HLS STREAM variable=v274 depth=1024	// L450
  hls::stream<float> v275[4][2];
	#pragma HLS STREAM variable=v275 depth=1024	// L451
  hls::stream<float> v276[1][4];
	#pragma HLS STREAM variable=v276 depth=1024	// L452
  hls::stream<float> v277[1][4];
	#pragma HLS STREAM variable=v277 depth=1024	// L453
  hls::stream<float> v278[8][4];
	#pragma HLS STREAM variable=v278 depth=1024	// L454
  hls::stream<float> v279[1][8];
	#pragma HLS STREAM variable=v279 depth=1024	// L455
  hls::stream<float> v280[1][8];
	#pragma HLS STREAM variable=v280 depth=1024	// L456
  hls::stream<float> v281[28][8];
	#pragma HLS STREAM variable=v281 depth=896	// L457
  node18(v252, v281);	// L458
  node17(v250, v281, v280, 0.000000);	// L459
  node16(v280, v251, v279, 0.000000);	// L460
  node15(v254, v278);	// L461
  node14(v278, v279, v277, 0.000000);	// L462
  node13(v277, v253, v276, 0.000000);	// L463
  node12(v256, v275);	// L464
  node11(v276, v275, v274, 0.000000);	// L465
  node10(v274, v255, v273);	// L466
  node9(v258, v272);	// L467
  node8(v272, v273, v271, 0.000000);	// L468
  node7(v271, v257, v270, 0.000000);	// L469
  node6(v260, v269);	// L470
  node5(v270, v269, v268, 0.000000);	// L471
  node4(v268, v259, v267, 0.000000);	// L472
  node3(v262, v266);	// L473
  node2(v267, v266, v265, 0.000000);	// L474
  node1(v265, v261, v264, 1.000000);	// L475
  node0(v264, v263);	// L476
  return ;	// L477
}

