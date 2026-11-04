
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
  float v1[32][784]
) {	// L7
  loop0: for (int v2 = 0; v2 < 32; v2++) {	// L8
    loop1: for (int v3 = 0; v3 < 784; v3++) {	// L9
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v4 = v0.read();	// L10
      v1[v2][v3] = v4;	// L11
    }
  }
  return ;	// L14
}

void node1(
  hls::stream<float> &v5,
  float v6[784],
  hls::stream<float> &v7,
  float v8
) {	// L16
  loop2: for (int v9 = 0; v9 < 32; v9++) {	// L17
    loop3: for (int v10 = 0; v10 < 784; v10++) {	// L18
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v11 = v5.read();	// L19
      float v12 = v6[v10];	// L20
      float v13 = v11 + v12;	// L21
      float v14 = -(v13);	// L22
      float v15 = exp(v14);	// L23
      float v16 = v15 + v8;	// L24
      float v17 = v8 / v16;	// L25
      v7.write(v17);	// L26
    }
  }
  return ;	// L29
}

void node2(
  float v18[256][784],
  hls::stream<float> &v19,
  hls::stream<float> &v20,
  float v21
) {	// L31
  float v22[32][784];	// L32
  float v23[256][784];	// L33
  float v24[32][256];	// L34
  loop4: for (int v25 = 0; v25 < 32; v25++) {	// L35
    loop5: for (int v26 = 0; v26 < 256; v26++) {	// L36
      loop6: for (int v27 = 0; v27 < 784; v27++) {	// L37
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v27 == 0) {	// L38
          float v28 = v19.read();	// L39
          v24[v25][v26] = v28;	// L40
        }
        float v29 = v24[v25][v26];	// L42
        if (v25 == 0) {	// L43
          float v30 = v18[v26][v27];	// L44
          v23[v26][v27] = v30;	// L45
        }
        float v31 = v23[v26][v27];	// L47
        if (v26 == 0) {	// L48
          v22[v25][v27] = v21;	// L49
        }
        float v32 = v22[v25][v27];	// L51
        float v33 = v29 * v31;	// L52
        float v34 = v32 + v33;	// L53
        v22[v25][v27] = v34;	// L54
        if ((v26 - 255) == 0) {	// L55
          float v35 = v22[v25][v27];	// L56
          v20.write(v35);	// L57
        }
      }
    }
  }
  return ;	// L62
}

void node3(
  float v36[784][256],
  float v37[256][784]
) {	// L64
  loop7: for (int v38 = 0; v38 < 784; v38++) {	// L65
    loop8: for (int v39 = 0; v39 < 256; v39++) {	// L66
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v40 = v36[v38][v39];	// L67
      v37[v39][v38] = v40;	// L68
    }
  }
  return ;	// L71
}

void node4(
  hls::stream<float> &v41,
  float v42[256],
  hls::stream<float> &v43,
  float v44
) {	// L73
  loop9: for (int v45 = 0; v45 < 32; v45++) {	// L74
    loop10: for (int v46 = 0; v46 < 256; v46++) {	// L75
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v47 = v41.read();	// L76
      float v48 = v42[v46];	// L77
      float v49 = v47 + v48;	// L78
      bool v50 = v49 > v44;	// L79
      float v51 = v50 ? (float)v49 : (float)v44;	// L80
      v43.write(v51);	// L81
    }
  }
  return ;	// L84
}

void node5(
  float v52[128][256],
  hls::stream<float> &v53,
  hls::stream<float> &v54,
  float v55
) {	// L86
  float v56[32][256];	// L87
  float v57[128][256];	// L88
  float v58[32][128];	// L89
  loop11: for (int v59 = 0; v59 < 32; v59++) {	// L90
    loop12: for (int v60 = 0; v60 < 256; v60++) {	// L91
      loop13: for (int v61 = 0; v61 < 128; v61++) {	// L92
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v60 == 0) {	// L93
          float v62 = v53.read();	// L94
          v58[v59][v61] = v62;	// L95
        }
        float v63 = v58[v59][v61];	// L97
        if (v59 == 0) {	// L98
          float v64 = v52[v61][v60];	// L99
          v57[v61][v60] = v64;	// L100
        }
        float v65 = v57[v61][v60];	// L102
        if (v61 == 0) {	// L103
          v56[v59][v60] = v55;	// L104
        }
        float v66 = v56[v59][v60];	// L106
        float v67 = v63 * v65;	// L107
        float v68 = v66 + v67;	// L108
        v56[v59][v60] = v68;	// L109
        if ((v61 - 127) == 0) {	// L110
          float v69 = v56[v59][v60];	// L111
          v54.write(v69);	// L112
        }
      }
    }
  }
  return ;	// L117
}

void node6(
  float v70[256][128],
  float v71[128][256]
) {	// L119
  loop14: for (int v72 = 0; v72 < 128; v72++) {	// L120
    loop15: for (int v73 = 0; v73 < 256; v73++) {	// L121
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v74 = v70[v73][v72];	// L122
      v71[v72][v73] = v74;	// L123
    }
  }
  return ;	// L126
}

void node7(
  hls::stream<float> &v75,
  float v76[128],
  hls::stream<float> &v77,
  float v78
) {	// L128
  loop16: for (int v79 = 0; v79 < 32; v79++) {	// L129
    loop17: for (int v80 = 0; v80 < 128; v80++) {	// L130
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v81 = v75.read();	// L131
      float v82 = v76[v80];	// L132
      float v83 = v81 + v82;	// L133
      bool v84 = v83 > v78;	// L134
      float v85 = v84 ? (float)v83 : (float)v78;	// L135
      v77.write(v85);	// L136
    }
  }
  return ;	// L139
}

void node8(
  float v86[64][128],
  hls::stream<float> &v87,
  hls::stream<float> &v88,
  float v89
) {	// L141
  float v90[32][128];	// L142
  float v91[64][128];	// L143
  float v92[32][64];	// L144
  loop18: for (int v93 = 0; v93 < 32; v93++) {	// L145
    loop19: for (int v94 = 0; v94 < 128; v94++) {	// L146
      loop20: for (int v95 = 0; v95 < 64; v95++) {	// L147
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v94 == 0) {	// L148
          float v96 = v87.read();	// L149
          v92[v93][v95] = v96;	// L150
        }
        float v97 = v92[v93][v95];	// L152
        if (v93 == 0) {	// L153
          float v98 = v86[v95][v94];	// L154
          v91[v95][v94] = v98;	// L155
        }
        float v99 = v91[v95][v94];	// L157
        if (v95 == 0) {	// L158
          v90[v93][v94] = v89;	// L159
        }
        float v100 = v90[v93][v94];	// L161
        float v101 = v97 * v99;	// L162
        float v102 = v100 + v101;	// L163
        v90[v93][v94] = v102;	// L164
        if ((v95 - 63) == 0) {	// L165
          float v103 = v90[v93][v94];	// L166
          v88.write(v103);	// L167
        }
      }
    }
  }
  return ;	// L172
}

void node9(
  float v104[128][64],
  float v105[64][128]
) {	// L174
  loop21: for (int v106 = 0; v106 < 64; v106++) {	// L175
    loop22: for (int v107 = 0; v107 < 128; v107++) {	// L176
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v108 = v104[v107][v106];	// L177
      v105[v106][v107] = v108;	// L178
    }
  }
  return ;	// L181
}

void node10(
  hls::stream<float> &v109,
  float v110[64],
  hls::stream<float> &v111
) {	// L183
  loop23: for (int v112 = 0; v112 < 32; v112++) {	// L184
    loop24: for (int v113 = 0; v113 < 64; v113++) {	// L185
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v114 = v109.read();	// L186
      float v115 = v110[v113];	// L187
      float v116 = v114 + v115;	// L188
      v111.write(v116);	// L189
    }
  }
  return ;	// L192
}

void node11(
  hls::stream<float> &v117,
  float v118[128][64],
  hls::stream<float> &v119,
  float v120
) {	// L194
  float v121[32][64];	// L195
  float v122[128][64];	// L196
  float v123[32][128];	// L197
  loop25: for (int v124 = 0; v124 < 32; v124++) {	// L198
    loop26: for (int v125 = 0; v125 < 64; v125++) {	// L199
      loop27: for (int v126 = 0; v126 < 128; v126++) {	// L200
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v125 == 0) {	// L201
          float v127 = v117.read();	// L202
          v123[v124][v126] = v127;	// L203
        }
        float v128 = v123[v124][v126];	// L205
        if (v124 == 0) {	// L206
          float v129 = v118[v126][v125];	// L207
          v122[v126][v125] = v129;	// L208
        }
        float v130 = v122[v126][v125];	// L210
        if (v126 == 0) {	// L211
          v121[v124][v125] = v120;	// L212
        }
        float v131 = v121[v124][v125];	// L214
        float v132 = v128 * v130;	// L215
        float v133 = v131 + v132;	// L216
        v121[v124][v125] = v133;	// L217
        if ((v126 - 127) == 0) {	// L218
          float v134 = v121[v124][v125];	// L219
          v119.write(v134);	// L220
        }
      }
    }
  }
  return ;	// L225
}

void node12(
  float v135[64][128],
  float v136[128][64]
) {	// L227
  loop28: for (int v137 = 0; v137 < 128; v137++) {	// L228
    loop29: for (int v138 = 0; v138 < 64; v138++) {	// L229
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v139 = v135[v138][v137];	// L230
      v136[v137][v138] = v139;	// L231
    }
  }
  return ;	// L234
}

void node13(
  hls::stream<float> &v140,
  float v141[128],
  hls::stream<float> &v142,
  float v143
) {	// L236
  loop30: for (int v144 = 0; v144 < 32; v144++) {	// L237
    loop31: for (int v145 = 0; v145 < 128; v145++) {	// L238
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v146 = v140.read();	// L239
      float v147 = v141[v145];	// L240
      float v148 = v146 + v147;	// L241
      bool v149 = v148 > v143;	// L242
      float v150 = v149 ? (float)v148 : (float)v143;	// L243
      v142.write(v150);	// L244
    }
  }
  return ;	// L247
}

void node14(
  hls::stream<float> &v151,
  hls::stream<float> &v152,
  hls::stream<float> &v153,
  float v154
) {	// L249
  float v155[32][128];	// L250
  float v156[256][128];	// L251
  float v157[32][256];	// L252
  loop32: for (int v158 = 0; v158 < 32; v158++) {	// L253
    loop33: for (int v159 = 0; v159 < 256; v159++) {	// L254
      loop34: for (int v160 = 0; v160 < 128; v160++) {	// L255
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v160 == 0) {	// L256
          float v161 = v152.read();	// L257
          v157[v158][v159] = v161;	// L258
        }
        float v162 = v157[v158][v159];	// L260
        if (v158 == 0) {	// L261
          float v163 = v151.read();	// L262
          v156[v159][v160] = v163;	// L263
        }
        float v164 = v156[v159][v160];	// L265
        if (v159 == 0) {	// L266
          v155[v158][v160] = v154;	// L267
        }
        float v165 = v155[v158][v160];	// L269
        float v166 = v162 * v164;	// L270
        float v167 = v165 + v166;	// L271
        v155[v158][v160] = v167;	// L272
        if ((v159 - 255) == 0) {	// L273
          float v168 = v155[v158][v160];	// L274
          v153.write(v168);	// L275
        }
      }
    }
  }
  return ;	// L280
}

void node15(
  float v169[128][256],
  hls::stream<float> &v170
) {	// L282
  loop35: for (int v171 = 0; v171 < 256; v171++) {	// L283
    loop36: for (int v172 = 0; v172 < 128; v172++) {	// L284
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v173 = v169[v172][v171];	// L285
      v170.write(v173);	// L286
    }
  }
  return ;	// L289
}

void node16(
  hls::stream<float> &v174,
  float v175[256],
  hls::stream<float> &v176,
  float v177
) {	// L291
  loop37: for (int v178 = 0; v178 < 32; v178++) {	// L292
    loop38: for (int v179 = 0; v179 < 256; v179++) {	// L293
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v180 = v174.read();	// L294
      float v181 = v175[v179];	// L295
      float v182 = v180 + v181;	// L296
      bool v183 = v182 > v177;	// L297
      float v184 = v183 ? (float)v182 : (float)v177;	// L298
      v176.write(v184);	// L299
    }
  }
  return ;	// L302
}

void node17(
  float v185[32][784],
  hls::stream<float> &v186,
  hls::stream<float> &v187,
  float v188
) {	// L304
  float v189[32][256];	// L305
  float v190[784][256];	// L306
  loop39: for (int v191 = 0; v191 < 32; v191++) {	// L307
    loop40: for (int v192 = 0; v192 < 784; v192++) {	// L308
      loop41: for (int v193 = 0; v193 < 256; v193++) {	// L309
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v194 = v185[v191][v192];	// L310
        if (v191 == 0) {	// L311
          float v195 = v186.read();	// L312
          v190[v192][v193] = v195;	// L313
        }
        float v196 = v190[v192][v193];	// L315
        if (v192 == 0) {	// L316
          v189[v191][v193] = v188;	// L317
        }
        float v197 = v189[v191][v193];	// L319
        float v198 = v194 * v196;	// L320
        float v199 = v197 + v198;	// L321
        v189[v191][v193] = v199;	// L322
        if ((v192 - 783) == 0) {	// L323
          float v200 = v189[v191][v193];	// L324
          v187.write(v200);	// L325
        }
      }
    }
  }
  return ;	// L330
}

void node18(
  float v201[256][784],
  hls::stream<float> &v202
) {	// L332
  loop42: for (int v203 = 0; v203 < 784; v203++) {	// L333
    loop43: for (int v204 = 0; v204 < 256; v204++) {	// L334
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v205 = v201[v204][v203];	// L335
      v202.write(v205);	// L336
    }
  }
  return ;	// L339
}

void forward(
  float v206[32][784],
  float v207[256],
  float v208[256][784],
  float v209[128],
  float v210[128][256],
  float v211[64],
  float v212[64][128],
  float v213[128],
  float v214[128][64],
  float v215[256],
  float v216[256][128],
  float v217[784],
  float v218[784][256],
  float v219[32][784]
) {	// L341
	#pragma HLS DATAFLOW
  hls::stream<float> v220("fifo_0");
	#pragma HLS STREAM variable=v220 depth=25088	// L344
  hls::stream<float> v221("fifo_1");
	#pragma HLS STREAM variable=v221 depth=25088	// L345
  float v222[256][784];	// L346
  hls::stream<float> v223("fifo_2");
	#pragma HLS STREAM variable=v223 depth=8192	// L347
  hls::stream<float> v224("fifo_3");
	#pragma HLS STREAM variable=v224 depth=8192	// L348
  float v225[128][256];	// L349
  hls::stream<float> v226("fifo_4");
	#pragma HLS STREAM variable=v226 depth=4096	// L350
  hls::stream<float> v227("fifo_5");
	#pragma HLS STREAM variable=v227 depth=4096	// L351
  float v228[64][128];	// L352
  hls::stream<float> v229("fifo_6");
	#pragma HLS STREAM variable=v229 depth=2048	// L353
  hls::stream<float> v230("fifo_7");
	#pragma HLS STREAM variable=v230 depth=2048	// L354
  float v231[128][64];	// L355
  hls::stream<float> v232("fifo_8");
	#pragma HLS STREAM variable=v232 depth=4096	// L356
  hls::stream<float> v233("fifo_9");
	#pragma HLS STREAM variable=v233 depth=4096	// L357
  hls::stream<float> v234("fifo_10");
	#pragma HLS STREAM variable=v234 depth=32768	// L358
  hls::stream<float> v235("fifo_11");
	#pragma HLS STREAM variable=v235 depth=8192	// L359
  hls::stream<float> v236("fifo_12");
	#pragma HLS STREAM variable=v236 depth=8192	// L360
  hls::stream<float> v237("fifo_13");
	#pragma HLS STREAM variable=v237 depth=200704	// L361
  node18(v208, v237);	// L362
  node17(v206, v237, v236, 0.000000);	// L363
  node16(v236, v207, v235, 0.000000);	// L364
  node15(v210, v234);	// L365
  node14(v234, v235, v233, 0.000000);	// L366
  node13(v233, v209, v232, 0.000000);	// L367
  node12(v212, v231);	// L368
  node11(v232, v231, v230, 0.000000);	// L369
  node10(v230, v211, v229);	// L370
  node9(v214, v228);	// L371
  node8(v228, v229, v227, 0.000000);	// L372
  node7(v227, v213, v226, 0.000000);	// L373
  node6(v216, v225);	// L374
  node5(v225, v226, v224, 0.000000);	// L375
  node4(v224, v215, v223, 0.000000);	// L376
  node3(v218, v222);	// L377
  node2(v222, v223, v221, 0.000000);	// L378
  node1(v221, v217, v220, 1.000000);	// L379
  node0(v220, v219);	// L380
  return ;	// L381
}

