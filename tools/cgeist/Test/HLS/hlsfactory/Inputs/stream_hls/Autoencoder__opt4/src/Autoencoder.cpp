
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
  hls::stream<float> v0[1][8],
  float v1[32][784]
) {	// L12
  loop0: for (int v2 = 0; v2 < 32; v2++) {	// L13
    loop1: for (int v3 = 0; v3 < 98; v3++) {	// L14
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 1; v4++) {	// L15
        loop3: for (int v5 = 0; v5 < 8; v5++) {	// L16
          float v6 = v0[v4][v5].read();	// L17
          v1[(v2 + v4)][(v5 + (v3 * 8))] = v6;	// L18
        }
      }
    }
  }
  return ;	// L23
}

void node1(
  hls::stream<float> v7[1][8],
  float v8[784],
  hls::stream<float> v9[1][8],
  float v10
) {	// L25
  loop4: for (int v11 = 0; v11 < 32; v11++) {	// L26
    loop5: for (int v12 = 0; v12 < 98; v12++) {	// L27
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v13 = 0; v13 < 1; v13++) {	// L28
        loop7: for (int v14 = 0; v14 < 8; v14++) {	// L29
          float v15 = v7[v13][v14].read();	// L30
          float v16 = v8[(v14 + (v12 * 8))];	// L31
          float v17 = v15 + v16;	// L32
          float v18 = -(v17);	// L33
          float v19 = exp(v18);	// L34
          float v20 = v19 + v10;	// L35
          float v21 = v10 / v20;	// L36
          v9[v13][v14].write(v21);	// L37
        }
      }
    }
  }
  return ;	// L42
}

void node2(
  hls::stream<float> v22[1][16],
  float v23[256][784],
  hls::stream<float> v24[1][8],
  float v25
) {	// L44
  float v26[32][784];	// L45
  float v27[256][784];	// L46
  float v28[32][256];	// L47
  loop8: for (int v29 = 0; v29 < 32; v29++) {	// L48
    loop9: for (int v30 = 0; v30 < 16; v30++) {	// L49
      loop10: for (int v31 = 0; v31 < 98; v31++) {	// L50
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v32 = 0; v32 < 1; v32++) {	// L51
          loop12: for (int v33 = 0; v33 < 16; v33++) {	// L52
            loop13: for (int v34 = 0; v34 < 8; v34++) {	// L53
              if ((v34 + (v31 * 8)) == 0) {	// L54
                float v35 = v22[v32][v33].read();	// L55
                v28[(v29 + v32)][(v33 + (v30 * 16))] = v35;	// L56
              }
              float v36 = v28[(v29 + v32)][(v33 + (v30 * 16))];	// L58
              if ((v29 + v32) == 0) {	// L59
                float v37 = v23[(v33 + (v30 * 16))][(v34 + (v31 * 8))];	// L60
                v27[(v33 + (v30 * 16))][(v34 + (v31 * 8))] = v37;	// L61
              }
              float v38 = v27[(v33 + (v30 * 16))][(v34 + (v31 * 8))];	// L63
              if ((v33 + (v30 * 16)) == 0) {	// L64
                v26[(v29 + v32)][(v34 + (v31 * 8))] = v25;	// L65
              }
              float v39 = v26[(v29 + v32)][(v34 + (v31 * 8))];	// L67
              float v40 = v36 * v38;	// L68
              float v41 = v39 + v40;	// L69
              v26[(v29 + v32)][(v34 + (v31 * 8))] = v41;	// L70
              if (((v33 + (v30 * 16)) - 255) == 0) {	// L71
                float v42 = v26[(v29 + v32)][(v34 + (v31 * 8))];	// L72
                v24[v32][v34].write(v42);	// L73
              }
            }
          }
        }
      }
    }
  }
  return ;	// L81
}

void node3(
  float v43[784][256],
  float v44[256][784]
) {	// L83
  loop14: for (int v45 = 0; v45 < 98; v45++) {	// L84
    loop15: for (int v46 = 0; v46 < 16; v46++) {	// L85
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v47 = 0; v47 < 8; v47++) {	// L86
        loop17: for (int v48 = 0; v48 < 16; v48++) {	// L87
          float v49 = v43[(v47 + (v45 * 8))][(v48 + (v46 * 16))];	// L88
          v44[(v48 + (v46 * 16))][(v47 + (v45 * 8))] = v49;	// L89
        }
      }
    }
  }
  return ;	// L94
}

void node4(
  hls::stream<float> v50[1][16],
  float v51[256],
  hls::stream<float> v52[1][16],
  float v53
) {	// L96
  loop18: for (int v54 = 0; v54 < 32; v54++) {	// L97
    loop19: for (int v55 = 0; v55 < 16; v55++) {	// L98
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop20: for (int v56 = 0; v56 < 1; v56++) {	// L99
        loop21: for (int v57 = 0; v57 < 16; v57++) {	// L100
          float v58 = v50[v56][v57].read();	// L101
          float v59 = v51[(v57 + (v55 * 16))];	// L102
          float v60 = v58 + v59;	// L103
          bool v61 = v60 > v53;	// L104
          float v62 = v61 ? (float)v60 : (float)v53;	// L105
          v52[v56][v57].write(v62);	// L106
        }
      }
    }
  }
  return ;	// L111
}

void node5(
  hls::stream<float> v63[1][8],
  float v64[128][256],
  hls::stream<float> v65[1][16],
  float v66
) {	// L113
  float v67[32][256];	// L114
  float v68[128][256];	// L115
  float v69[32][128];	// L116
  loop22: for (int v70 = 0; v70 < 32; v70++) {	// L117
    loop23: for (int v71 = 0; v71 < 16; v71++) {	// L118
      loop24: for (int v72 = 0; v72 < 16; v72++) {	// L119
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v73 = 0; v73 < 1; v73++) {	// L120
          loop26: for (int v74 = 0; v74 < 16; v74++) {	// L121
            loop27: for (int v75 = 0; v75 < 8; v75++) {	// L122
              if ((v74 + (v71 * 16)) == 0) {	// L123
                float v76 = v63[v73][v75].read();	// L124
                v69[(v70 + v73)][(v75 + (v72 * 8))] = v76;	// L125
              }
              float v77 = v69[(v70 + v73)][(v75 + (v72 * 8))];	// L127
              if ((v70 + v73) == 0) {	// L128
                float v78 = v64[(v75 + (v72 * 8))][(v74 + (v71 * 16))];	// L129
                v68[(v75 + (v72 * 8))][(v74 + (v71 * 16))] = v78;	// L130
              }
              float v79 = v68[(v75 + (v72 * 8))][(v74 + (v71 * 16))];	// L132
              if ((v75 + (v72 * 8)) == 0) {	// L133
                v67[(v70 + v73)][(v74 + (v71 * 16))] = v66;	// L134
              }
              float v80 = v67[(v70 + v73)][(v74 + (v71 * 16))];	// L136
              float v81 = v77 * v79;	// L137
              float v82 = v80 + v81;	// L138
              v67[(v70 + v73)][(v74 + (v71 * 16))] = v82;	// L139
              if (((v75 + (v72 * 8)) - 127) == 0) {	// L140
                float v83 = v67[(v70 + v73)][(v74 + (v71 * 16))];	// L141
                v65[v73][v74].write(v83);	// L142
              }
            }
          }
        }
      }
    }
  }
  return ;	// L150
}

void node6(
  float v84[256][128],
  float v85[128][256]
) {	// L152
  loop28: for (int v86 = 0; v86 < 16; v86++) {	// L153
    loop29: for (int v87 = 0; v87 < 16; v87++) {	// L154
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop30: for (int v88 = 0; v88 < 8; v88++) {	// L155
        loop31: for (int v89 = 0; v89 < 16; v89++) {	// L156
          float v90 = v84[(v89 + (v87 * 16))][(v88 + (v86 * 8))];	// L157
          v85[(v88 + (v86 * 8))][(v89 + (v87 * 16))] = v90;	// L158
        }
      }
    }
  }
  return ;	// L163
}

void node7(
  hls::stream<float> v91[1][8],
  float v92[128],
  hls::stream<float> v93[1][8],
  float v94
) {	// L165
  loop32: for (int v95 = 0; v95 < 32; v95++) {	// L166
    loop33: for (int v96 = 0; v96 < 16; v96++) {	// L167
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop34: for (int v97 = 0; v97 < 1; v97++) {	// L168
        loop35: for (int v98 = 0; v98 < 8; v98++) {	// L169
          float v99 = v91[v97][v98].read();	// L170
          float v100 = v92[(v98 + (v96 * 8))];	// L171
          float v101 = v99 + v100;	// L172
          bool v102 = v101 > v94;	// L173
          float v103 = v102 ? (float)v101 : (float)v94;	// L174
          v93[v97][v98].write(v103);	// L175
        }
      }
    }
  }
  return ;	// L180
}

void node8(
  hls::stream<float> v104[1][4],
  float v105[64][128],
  hls::stream<float> v106[1][8],
  float v107
) {	// L182
  float v108[32][128];	// L183
  float v109[64][128];	// L184
  float v110[32][64];	// L185
  loop36: for (int v111 = 0; v111 < 32; v111++) {	// L186
    loop37: for (int v112 = 0; v112 < 16; v112++) {	// L187
      loop38: for (int v113 = 0; v113 < 16; v113++) {	// L188
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v114 = 0; v114 < 1; v114++) {	// L189
          loop40: for (int v115 = 0; v115 < 8; v115++) {	// L190
            loop41: for (int v116 = 0; v116 < 4; v116++) {	// L191
              if ((v115 + (v112 * 8)) == 0) {	// L192
                float v117 = v104[v114][v116].read();	// L193
                v110[(v111 + v114)][(v116 + (v113 * 4))] = v117;	// L194
              }
              float v118 = v110[(v111 + v114)][(v116 + (v113 * 4))];	// L196
              if ((v111 + v114) == 0) {	// L197
                float v119 = v105[(v116 + (v113 * 4))][(v115 + (v112 * 8))];	// L198
                v109[(v116 + (v113 * 4))][(v115 + (v112 * 8))] = v119;	// L199
              }
              float v120 = v109[(v116 + (v113 * 4))][(v115 + (v112 * 8))];	// L201
              if ((v116 + (v113 * 4)) == 0) {	// L202
                v108[(v111 + v114)][(v115 + (v112 * 8))] = v107;	// L203
              }
              float v121 = v108[(v111 + v114)][(v115 + (v112 * 8))];	// L205
              float v122 = v118 * v120;	// L206
              float v123 = v121 + v122;	// L207
              v108[(v111 + v114)][(v115 + (v112 * 8))] = v123;	// L208
              if (((v116 + (v113 * 4)) - 63) == 0) {	// L209
                float v124 = v108[(v111 + v114)][(v115 + (v112 * 8))];	// L210
                v106[v114][v115].write(v124);	// L211
              }
            }
          }
        }
      }
    }
  }
  return ;	// L219
}

void node9(
  float v125[128][64],
  float v126[64][128]
) {	// L221
  loop42: for (int v127 = 0; v127 < 16; v127++) {	// L222
    loop43: for (int v128 = 0; v128 < 16; v128++) {	// L223
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop44: for (int v129 = 0; v129 < 4; v129++) {	// L224
        loop45: for (int v130 = 0; v130 < 8; v130++) {	// L225
          float v131 = v125[(v130 + (v128 * 8))][(v129 + (v127 * 4))];	// L226
          v126[(v129 + (v127 * 4))][(v130 + (v128 * 8))] = v131;	// L227
        }
      }
    }
  }
  return ;	// L232
}

void node10(
  hls::stream<float> v132[1][4],
  float v133[64],
  hls::stream<float> v134[1][4]
) {	// L234
  loop46: for (int v135 = 0; v135 < 32; v135++) {	// L235
    loop47: for (int v136 = 0; v136 < 16; v136++) {	// L236
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop48: for (int v137 = 0; v137 < 1; v137++) {	// L237
        loop49: for (int v138 = 0; v138 < 4; v138++) {	// L238
          float v139 = v132[v137][v138].read();	// L239
          float v140 = v133[(v138 + (v136 * 4))];	// L240
          float v141 = v139 + v140;	// L241
          v134[v137][v138].write(v141);	// L242
        }
      }
    }
  }
  return ;	// L247
}

void node11(
  hls::stream<float> v142[1][8],
  float v143[128][64],
  hls::stream<float> v144[1][4],
  float v145
) {	// L249
  float v146[32][64];	// L250
  float v147[128][64];	// L251
  float v148[32][128];	// L252
  loop50: for (int v149 = 0; v149 < 32; v149++) {	// L253
    loop51: for (int v150 = 0; v150 < 16; v150++) {	// L254
      loop52: for (int v151 = 0; v151 < 16; v151++) {	// L255
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop53: for (int v152 = 0; v152 < 1; v152++) {	// L256
          loop54: for (int v153 = 0; v153 < 4; v153++) {	// L257
            loop55: for (int v154 = 0; v154 < 8; v154++) {	// L258
              if ((v153 + (v150 * 4)) == 0) {	// L259
                float v155 = v142[v152][v154].read();	// L260
                v148[(v149 + v152)][(v154 + (v151 * 8))] = v155;	// L261
              }
              float v156 = v148[(v149 + v152)][(v154 + (v151 * 8))];	// L263
              if ((v149 + v152) == 0) {	// L264
                float v157 = v143[(v154 + (v151 * 8))][(v153 + (v150 * 4))];	// L265
                v147[(v154 + (v151 * 8))][(v153 + (v150 * 4))] = v157;	// L266
              }
              float v158 = v147[(v154 + (v151 * 8))][(v153 + (v150 * 4))];	// L268
              if ((v154 + (v151 * 8)) == 0) {	// L269
                v146[(v149 + v152)][(v153 + (v150 * 4))] = v145;	// L270
              }
              float v159 = v146[(v149 + v152)][(v153 + (v150 * 4))];	// L272
              float v160 = v156 * v158;	// L273
              float v161 = v159 + v160;	// L274
              v146[(v149 + v152)][(v153 + (v150 * 4))] = v161;	// L275
              if (((v154 + (v151 * 8)) - 127) == 0) {	// L276
                float v162 = v146[(v149 + v152)][(v153 + (v150 * 4))];	// L277
                v144[v152][v153].write(v162);	// L278
              }
            }
          }
        }
      }
    }
  }
  return ;	// L286
}

void node12(
  float v163[64][128],
  float v164[128][64]
) {	// L288
  loop56: for (int v165 = 0; v165 < 16; v165++) {	// L289
    loop57: for (int v166 = 0; v166 < 16; v166++) {	// L290
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop58: for (int v167 = 0; v167 < 8; v167++) {	// L291
        loop59: for (int v168 = 0; v168 < 4; v168++) {	// L292
          float v169 = v163[(v168 + (v166 * 4))][(v167 + (v165 * 8))];	// L293
          v164[(v167 + (v165 * 8))][(v168 + (v166 * 4))] = v169;	// L294
        }
      }
    }
  }
  return ;	// L299
}

void node13(
  hls::stream<float> v170[1][8],
  float v171[128],
  hls::stream<float> v172[1][8],
  float v173
) {	// L301
  loop60: for (int v174 = 0; v174 < 32; v174++) {	// L302
    loop61: for (int v175 = 0; v175 < 16; v175++) {	// L303
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop62: for (int v176 = 0; v176 < 1; v176++) {	// L304
        loop63: for (int v177 = 0; v177 < 8; v177++) {	// L305
          float v178 = v170[v176][v177].read();	// L306
          float v179 = v171[(v177 + (v175 * 8))];	// L307
          float v180 = v178 + v179;	// L308
          bool v181 = v180 > v173;	// L309
          float v182 = v181 ? (float)v180 : (float)v173;	// L310
          v172[v176][v177].write(v182);	// L311
        }
      }
    }
  }
  return ;	// L316
}

void node14(
  hls::stream<float> v183[1][8],
  hls::stream<float> v184[8][8],
  hls::stream<float> v185[1][8],
  float v186
) {	// L318
  float v187[32][128];	// L319
  float v188[256][128];	// L320
  float v189[32][256];	// L321
  loop64: for (int v190 = 0; v190 < 32; v190++) {	// L322
    loop65: for (int v191 = 0; v191 < 32; v191++) {	// L323
      loop66: for (int v192 = 0; v192 < 16; v192++) {	// L324
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop67: for (int v193 = 0; v193 < 1; v193++) {	// L325
          loop68: for (int v194 = 0; v194 < 8; v194++) {	// L326
            loop69: for (int v195 = 0; v195 < 8; v195++) {	// L327
              if ((v195 + (v192 * 8)) == 0) {	// L328
                float v196 = v183[v193][v194].read();	// L329
                v189[(v190 + v193)][(v194 + (v191 * 8))] = v196;	// L330
              }
              float v197 = v189[(v190 + v193)][(v194 + (v191 * 8))];	// L332
              if ((v190 + v193) == 0) {	// L333
                float v198 = v184[v194][v195].read();	// L334
                v188[(v194 + (v191 * 8))][(v195 + (v192 * 8))] = v198;	// L335
              }
              float v199 = v188[(v194 + (v191 * 8))][(v195 + (v192 * 8))];	// L337
              if ((v194 + (v191 * 8)) == 0) {	// L338
                v187[(v190 + v193)][(v195 + (v192 * 8))] = v186;	// L339
              }
              float v200 = v187[(v190 + v193)][(v195 + (v192 * 8))];	// L341
              float v201 = v197 * v199;	// L342
              float v202 = v200 + v201;	// L343
              v187[(v190 + v193)][(v195 + (v192 * 8))] = v202;	// L344
              if (((v194 + (v191 * 8)) - 255) == 0) {	// L345
                float v203 = v187[(v190 + v193)][(v195 + (v192 * 8))];	// L346
                v185[v193][v195].write(v203);	// L347
              }
            }
          }
        }
      }
    }
  }
  return ;	// L355
}

void node15(
  float v204[128][256],
  hls::stream<float> v205[8][8]
) {	// L357
  loop70: for (int v206 = 0; v206 < 32; v206++) {	// L358
    loop71: for (int v207 = 0; v207 < 16; v207++) {	// L359
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop72: for (int v208 = 0; v208 < 8; v208++) {	// L360
        loop73: for (int v209 = 0; v209 < 8; v209++) {	// L361
          float v210 = v204[(v209 + (v207 * 8))][(v208 + (v206 * 8))];	// L362
          v205[v208][v209].write(v210);	// L363
        }
      }
    }
  }
  return ;	// L368
}

void node16(
  hls::stream<float> v211[1][8],
  float v212[256],
  hls::stream<float> v213[1][8],
  float v214
) {	// L370
  loop74: for (int v215 = 0; v215 < 32; v215++) {	// L371
    loop75: for (int v216 = 0; v216 < 32; v216++) {	// L372
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop76: for (int v217 = 0; v217 < 1; v217++) {	// L373
        loop77: for (int v218 = 0; v218 < 8; v218++) {	// L374
          float v219 = v211[v217][v218].read();	// L375
          float v220 = v212[(v218 + (v216 * 8))];	// L376
          float v221 = v219 + v220;	// L377
          bool v222 = v221 > v214;	// L378
          float v223 = v222 ? (float)v221 : (float)v214;	// L379
          v213[v217][v218].write(v223);	// L380
        }
      }
    }
  }
  return ;	// L385
}

void node17(
  hls::stream<float> v224[16][8],
  float v225[32][784],
  hls::stream<float> v226[1][8],
  float v227
) {	// L387
  float v228[32][256];	// L388
  float v229[784][256];	// L389
  loop78: for (int v230 = 0; v230 < 32; v230++) {	// L390
    loop79: for (int v231 = 0; v231 < 49; v231++) {	// L391
      loop80: for (int v232 = 0; v232 < 32; v232++) {	// L392
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop81: for (int v233 = 0; v233 < 1; v233++) {	// L393
          loop82: for (int v234 = 0; v234 < 16; v234++) {	// L394
            loop83: for (int v235 = 0; v235 < 8; v235++) {	// L395
              float v236 = v225[(v230 + v233)][(v234 + (v231 * 16))];	// L396
              if ((v230 + v233) == 0) {	// L397
                float v237 = v224[v234][v235].read();	// L398
                v229[(v234 + (v231 * 16))][(v235 + (v232 * 8))] = v237;	// L399
              }
              float v238 = v229[(v234 + (v231 * 16))][(v235 + (v232 * 8))];	// L401
              if ((v234 + (v231 * 16)) == 0) {	// L402
                v228[(v230 + v233)][(v235 + (v232 * 8))] = v227;	// L403
              }
              float v239 = v228[(v230 + v233)][(v235 + (v232 * 8))];	// L405
              float v240 = v236 * v238;	// L406
              float v241 = v239 + v240;	// L407
              v228[(v230 + v233)][(v235 + (v232 * 8))] = v241;	// L408
              if (((v234 + (v231 * 16)) - 783) == 0) {	// L409
                float v242 = v228[(v230 + v233)][(v235 + (v232 * 8))];	// L410
                v226[v233][v235].write(v242);	// L411
              }
            }
          }
        }
      }
    }
  }
  return ;	// L419
}

void node18(
  float v243[256][784],
  hls::stream<float> v244[16][8]
) {	// L421
  loop84: for (int v245 = 0; v245 < 49; v245++) {	// L422
    loop85: for (int v246 = 0; v246 < 32; v246++) {	// L423
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop86: for (int v247 = 0; v247 < 16; v247++) {	// L424
        loop87: for (int v248 = 0; v248 < 8; v248++) {	// L425
          float v249 = v243[(v248 + (v246 * 8))][(v247 + (v245 * 16))];	// L426
          v244[v247][v248].write(v249);	// L427
        }
      }
    }
  }
  return ;	// L432
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
) {	// L434
	#pragma HLS DATAFLOW
  hls::stream<float> v264[1][8];
	#pragma HLS STREAM variable=v264 depth=3136	// L437
  hls::stream<float> v265[1][8];
	#pragma HLS STREAM variable=v265 depth=3136	// L438
  float v266[256][784];	// L439
  hls::stream<float> v267[1][16];
	#pragma HLS STREAM variable=v267 depth=512	// L440
  hls::stream<float> v268[1][16];
	#pragma HLS STREAM variable=v268 depth=512	// L441
  float v269[128][256];	// L442
  hls::stream<float> v270[1][8];
	#pragma HLS STREAM variable=v270 depth=512	// L443
  hls::stream<float> v271[1][8];
	#pragma HLS STREAM variable=v271 depth=512	// L444
  float v272[64][128];	// L445
  hls::stream<float> v273[1][4];
	#pragma HLS STREAM variable=v273 depth=512	// L446
  hls::stream<float> v274[1][4];
	#pragma HLS STREAM variable=v274 depth=512	// L447
  float v275[128][64];	// L448
  hls::stream<float> v276[1][8];
	#pragma HLS STREAM variable=v276 depth=512	// L449
  hls::stream<float> v277[1][8];
	#pragma HLS STREAM variable=v277 depth=512	// L450
  hls::stream<float> v278[8][8];
	#pragma HLS STREAM variable=v278 depth=512	// L451
  hls::stream<float> v279[1][8];
	#pragma HLS STREAM variable=v279 depth=1024	// L452
  hls::stream<float> v280[1][8];
	#pragma HLS STREAM variable=v280 depth=1024	// L453
  hls::stream<float> v281[16][8];
	#pragma HLS STREAM variable=v281 depth=1568	// L454
  node18(v252, v281);	// L455
  node17(v281, v250, v280, 0.000000);	// L456
  node16(v280, v251, v279, 0.000000);	// L457
  node15(v254, v278);	// L458
  node14(v279, v278, v277, 0.000000);	// L459
  node13(v277, v253, v276, 0.000000);	// L460
  node12(v256, v275);	// L461
  node11(v276, v275, v274, 0.000000);	// L462
  node10(v274, v255, v273);	// L463
  node9(v258, v272);	// L464
  node8(v273, v272, v271, 0.000000);	// L465
  node7(v271, v257, v270, 0.000000);	// L466
  node6(v260, v269);	// L467
  node5(v270, v269, v268, 0.000000);	// L468
  node4(v268, v259, v267, 0.000000);	// L469
  node3(v262, v266);	// L470
  node2(v267, v266, v265, 0.000000);	// L471
  node1(v265, v261, v264, 1.000000);	// L472
  node0(v264, v263);	// L473
  return ;	// L474
}

