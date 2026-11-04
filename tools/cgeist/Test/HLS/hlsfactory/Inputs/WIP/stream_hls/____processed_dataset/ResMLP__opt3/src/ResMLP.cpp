
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
  float v1[8][10]
) {	// L10
  loop0: for (int v2 = 0; v2 < 8; v2++) {	// L11
    loop1: for (int v3 = 0; v3 < 10; v3++) {	// L12
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop2: for (int v4 = 0; v4 < 1; v4++) {	// L13
        loop3: for (int v5 = 0; v5 < 1; v5++) {	// L14
          float v6 = v0[v4][v5].read();	// L15
          v1[(v2 + v4)][(v3 + v5)] = v6;	// L16
        }
      }
    }
  }
  return ;	// L21
}

void node1(
  hls::stream<float> v7[1][1],
  float v8[10],
  hls::stream<float> v9[1][1]
) {	// L23
  loop4: for (int v10 = 0; v10 < 8; v10++) {	// L24
    loop5: for (int v11 = 0; v11 < 10; v11++) {	// L25
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v12 = 0; v12 < 1; v12++) {	// L26
        loop7: for (int v13 = 0; v13 < 1; v13++) {	// L27
          float v14 = v7[v12][v13].read();	// L28
          float v15 = v8[(v11 + v13)];	// L29
          float v16 = v14 + v15;	// L30
          v9[v12][v13].write(v16);	// L31
        }
      }
    }
  }
  return ;	// L36
}

void node2(
  hls::stream<float> v17[16][1],
  hls::stream<float> v18[1][16],
  hls::stream<float> v19[1][1],
  float v20
) {	// L38
  float v21[8][10];	// L39
  float v22[256][10];	// L40
  float v23[8][256];	// L41
  loop8: for (int v24 = 0; v24 < 8; v24++) {	// L42
    loop9: for (int v25 = 0; v25 < 10; v25++) {	// L43
      loop10: for (int v26 = 0; v26 < 16; v26++) {	// L44
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop11: for (int v27 = 0; v27 < 1; v27++) {	// L45
          loop12: for (int v28 = 0; v28 < 1; v28++) {	// L46
            loop13: for (int v29 = 0; v29 < 16; v29++) {	// L47
              if ((v25 + v28) == 0) {	// L48
                float v30 = v18[v27][v29].read();	// L49
                v23[(v24 + v27)][(v29 + (v26 * 16))] = v30;	// L50
              }
              float v31 = v23[(v24 + v27)][(v29 + (v26 * 16))];	// L52
              if ((v24 + v27) == 0) {	// L53
                float v32 = v17[v29][v28].read();	// L54
                v22[(v29 + (v26 * 16))][(v25 + v28)] = v32;	// L55
              }
              float v33 = v22[(v29 + (v26 * 16))][(v25 + v28)];	// L57
              if ((v29 + (v26 * 16)) == 0) {	// L58
                v21[(v24 + v27)][(v25 + v28)] = v20;	// L59
              }
              float v34 = v21[(v24 + v27)][(v25 + v28)];	// L61
              float v35 = v31 * v33;	// L62
              float v36 = v34 + v35;	// L63
              v21[(v24 + v27)][(v25 + v28)] = v36;	// L64
              if (((v29 + (v26 * 16)) - 255) == 0) {	// L65
                float v37 = v21[(v24 + v27)][(v25 + v28)];	// L66
                v19[v27][v28].write(v37);	// L67
              }
            }
          }
        }
      }
    }
  }
  return ;	// L75
}

void node3(
  float v38[10][256],
  hls::stream<float> v39[16][1]
) {	// L77
  loop14: for (int v40 = 0; v40 < 10; v40++) {	// L78
    loop15: for (int v41 = 0; v41 < 16; v41++) {	// L79
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop16: for (int v42 = 0; v42 < 1; v42++) {	// L80
        loop17: for (int v43 = 0; v43 < 16; v43++) {	// L81
          float v44 = v38[(v40 + v42)][(v43 + (v41 * 16))];	// L82
          v39[v43][v42].write(v44);	// L83
        }
      }
    }
  }
  return ;	// L88
}

void node4(
  hls::stream<float> v45[1][16],
  float v46[256],
  hls::stream<float> v47[1][16]
) {	// L90
  loop18: for (int v48 = 0; v48 < 8; v48++) {	// L91
    loop19: for (int v49 = 0; v49 < 16; v49++) {	// L92
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop20: for (int v50 = 0; v50 < 1; v50++) {	// L93
        loop21: for (int v51 = 0; v51 < 16; v51++) {	// L94
          float v52 = v45[v50][v51].read();	// L95
          float v53 = v46[(v51 + (v49 * 16))];	// L96
          float v54 = v52 + v53;	// L97
          v47[v50][v51].write(v54);	// L98
        }
      }
    }
  }
  return ;	// L103
}

void node5(
  hls::stream<float> v55[1][8],
  hls::stream<float> v56[8][16],
  hls::stream<float> v57[1][16],
  float v58
) {	// L105
  float v59[8][256];	// L106
  float v60[512][256];	// L107
  float v61[8][512];	// L108
  loop22: for (int v62 = 0; v62 < 8; v62++) {	// L109
    loop23: for (int v63 = 0; v63 < 16; v63++) {	// L110
      loop24: for (int v64 = 0; v64 < 64; v64++) {	// L111
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop25: for (int v65 = 0; v65 < 1; v65++) {	// L112
          loop26: for (int v66 = 0; v66 < 16; v66++) {	// L113
            loop27: for (int v67 = 0; v67 < 8; v67++) {	// L114
              if ((v66 + (v63 * 16)) == 0) {	// L115
                float v68 = v55[v65][v67].read();	// L116
                v61[(v62 + v65)][(v67 + (v64 * 8))] = v68;	// L117
              }
              float v69 = v61[(v62 + v65)][(v67 + (v64 * 8))];	// L119
              if ((v62 + v65) == 0) {	// L120
                float v70 = v56[v67][v66].read();	// L121
                v60[(v67 + (v64 * 8))][(v66 + (v63 * 16))] = v70;	// L122
              }
              float v71 = v60[(v67 + (v64 * 8))][(v66 + (v63 * 16))];	// L124
              if ((v67 + (v64 * 8)) == 0) {	// L125
                v59[(v62 + v65)][(v66 + (v63 * 16))] = v58;	// L126
              }
              float v72 = v59[(v62 + v65)][(v66 + (v63 * 16))];	// L128
              float v73 = v69 * v71;	// L129
              float v74 = v72 + v73;	// L130
              v59[(v62 + v65)][(v66 + (v63 * 16))] = v74;	// L131
              if (((v67 + (v64 * 8)) - 511) == 0) {	// L132
                float v75 = v59[(v62 + v65)][(v66 + (v63 * 16))];	// L133
                v57[v65][v66].write(v75);	// L134
              }
            }
          }
        }
      }
    }
  }
  return ;	// L142
}

void node6(
  float v76[256][512],
  hls::stream<float> v77[8][16]
) {	// L144
  loop28: for (int v78 = 0; v78 < 16; v78++) {	// L145
    loop29: for (int v79 = 0; v79 < 64; v79++) {	// L146
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop30: for (int v80 = 0; v80 < 16; v80++) {	// L147
        loop31: for (int v81 = 0; v81 < 8; v81++) {	// L148
          float v82 = v76[(v80 + (v78 * 16))][(v81 + (v79 * 8))];	// L149
          v77[v81][v80].write(v82);	// L150
        }
      }
    }
  }
  return ;	// L155
}

void node7(
  hls::stream<float> v83[1][8],
  float v84[512],
  hls::stream<float> v85[1][8],
  hls::stream<float> v86[1][8],
  float v87
) {	// L157
  loop32: for (int v88 = 0; v88 < 8; v88++) {	// L158
    loop33: for (int v89 = 0; v89 < 64; v89++) {	// L159
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop34: for (int v90 = 0; v90 < 1; v90++) {	// L160
        loop35: for (int v91 = 0; v91 < 8; v91++) {	// L161
          float v92 = v83[v90][v91].read();	// L162
          float v93 = v84[(v91 + (v89 * 8))];	// L163
          float v94 = v85[v90][v91].read();	// L164
          float v95 = v92 + v93;	// L165
          float v96 = v95 + v94;	// L166
          bool v97 = v96 > v87;	// L167
          float v98 = v97 ? (float)v96 : (float)v87;	// L168
          v86[v90][v91].write(v98);	// L169
        }
      }
    }
  }
  return ;	// L174
}

void node8(
  hls::stream<float> v99[8][8],
  hls::stream<float> v100[1][8],
  hls::stream<float> v101[1][8],
  float v102
) {	// L176
  float v103[8][512];	// L177
  float v104[512][512];	// L178
  float v105[8][512];	// L179
  loop36: for (int v106 = 0; v106 < 8; v106++) {	// L180
    loop37: for (int v107 = 0; v107 < 64; v107++) {	// L181
      loop38: for (int v108 = 0; v108 < 64; v108++) {	// L182
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop39: for (int v109 = 0; v109 < 1; v109++) {	// L183
          loop40: for (int v110 = 0; v110 < 8; v110++) {	// L184
            loop41: for (int v111 = 0; v111 < 8; v111++) {	// L185
              if ((v110 + (v107 * 8)) == 0) {	// L186
                float v112 = v100[v109][v111].read();	// L187
                v105[(v106 + v109)][(v111 + (v108 * 8))] = v112;	// L188
              }
              float v113 = v105[(v106 + v109)][(v111 + (v108 * 8))];	// L190
              if ((v106 + v109) == 0) {	// L191
                float v114 = v99[v111][v110].read();	// L192
                v104[(v111 + (v108 * 8))][(v110 + (v107 * 8))] = v114;	// L193
              }
              float v115 = v104[(v111 + (v108 * 8))][(v110 + (v107 * 8))];	// L195
              if ((v111 + (v108 * 8)) == 0) {	// L196
                v103[(v106 + v109)][(v110 + (v107 * 8))] = v102;	// L197
              }
              float v116 = v103[(v106 + v109)][(v110 + (v107 * 8))];	// L199
              float v117 = v113 * v115;	// L200
              float v118 = v116 + v117;	// L201
              v103[(v106 + v109)][(v110 + (v107 * 8))] = v118;	// L202
              if (((v111 + (v108 * 8)) - 511) == 0) {	// L203
                float v119 = v103[(v106 + v109)][(v110 + (v107 * 8))];	// L204
                v101[v109][v110].write(v119);	// L205
              }
            }
          }
        }
      }
    }
  }
  return ;	// L213
}

void node9(
  float v120[512][512],
  hls::stream<float> v121[8][8]
) {	// L215
  loop42: for (int v122 = 0; v122 < 64; v122++) {	// L216
    loop43: for (int v123 = 0; v123 < 64; v123++) {	// L217
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop44: for (int v124 = 0; v124 < 8; v124++) {	// L218
        loop45: for (int v125 = 0; v125 < 8; v125++) {	// L219
          float v126 = v120[(v124 + (v122 * 8))][(v125 + (v123 * 8))];	// L220
          v121[v125][v124].write(v126);	// L221
        }
      }
    }
  }
  return ;	// L226
}

void node10(
  hls::stream<float> v127[1][8],
  hls::stream<float> v128[1][8],
  float v129
) {	// L228
  loop46: for (int v130 = 0; v130 < 8; v130++) {	// L229
    loop47: for (int v131 = 0; v131 < 64; v131++) {	// L230
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop48: for (int v132 = 0; v132 < 1; v132++) {	// L231
        loop49: for (int v133 = 0; v133 < 8; v133++) {	// L232
          float v134 = v127[v132][v133].read();	// L233
          bool v135 = v134 > v129;	// L234
          float v136 = v135 ? (float)v134 : (float)v129;	// L235
          v128[v132][v133].write(v136);	// L236
        }
      }
    }
  }
  return ;	// L241
}

void node11(
  hls::stream<float> v137[1][8],
  float v138[512],
  hls::stream<float> v139[1][8],
  hls::stream<float> v140[1][8]
) {	// L243
  loop50: for (int v141 = 0; v141 < 8; v141++) {	// L244
    loop51: for (int v142 = 0; v142 < 64; v142++) {	// L245
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop52: for (int v143 = 0; v143 < 1; v143++) {	// L246
        loop53: for (int v144 = 0; v144 < 8; v144++) {	// L247
          float v145 = v137[v143][v144].read();	// L248
          float v146 = v138[(v144 + (v142 * 8))];	// L249
          float v147 = v145 + v146;	// L250
          v139[v143][v144].write(v147);	// L251
          v140[v143][v144].write(v147);	// L252
        }
      }
    }
  }
  return ;	// L257
}

void node12(
  float v148[8][1024],
  hls::stream<float> v149[32][8],
  hls::stream<float> v150[1][8],
  float v151
) {	// L259
  float v152[8][512];	// L260
  float v153[1024][512];	// L261
  loop54: for (int v154 = 0; v154 < 8; v154++) {	// L262
    loop55: for (int v155 = 0; v155 < 64; v155++) {	// L263
      loop56: for (int v156 = 0; v156 < 32; v156++) {	// L264
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        loop57: for (int v157 = 0; v157 < 1; v157++) {	// L265
          loop58: for (int v158 = 0; v158 < 8; v158++) {	// L266
            loop59: for (int v159 = 0; v159 < 32; v159++) {	// L267
              float v160 = v148[(v154 + v157)][(v159 + (v156 * 32))];	// L268
              if ((v154 + v157) == 0) {	// L269
                float v161 = v149[v159][v158].read();	// L270
                v153[(v159 + (v156 * 32))][(v158 + (v155 * 8))] = v161;	// L271
              }
              float v162 = v153[(v159 + (v156 * 32))][(v158 + (v155 * 8))];	// L273
              if ((v159 + (v156 * 32)) == 0) {	// L274
                v152[(v154 + v157)][(v158 + (v155 * 8))] = v151;	// L275
              }
              float v163 = v152[(v154 + v157)][(v158 + (v155 * 8))];	// L277
              float v164 = v160 * v162;	// L278
              float v165 = v163 + v164;	// L279
              v152[(v154 + v157)][(v158 + (v155 * 8))] = v165;	// L280
              if (((v159 + (v156 * 32)) - 1023) == 0) {	// L281
                float v166 = v152[(v154 + v157)][(v158 + (v155 * 8))];	// L282
                v150[v157][v158].write(v166);	// L283
              }
            }
          }
        }
      }
    }
  }
  return ;	// L291
}

void node13(
  float v167[512][1024],
  hls::stream<float> v168[32][8]
) {	// L293
  loop60: for (int v169 = 0; v169 < 64; v169++) {	// L294
    loop61: for (int v170 = 0; v170 < 32; v170++) {	// L295
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop62: for (int v171 = 0; v171 < 8; v171++) {	// L296
        loop63: for (int v172 = 0; v172 < 32; v172++) {	// L297
          float v173 = v167[(v171 + (v169 * 8))][(v172 + (v170 * 32))];	// L298
          v168[v172][v171].write(v173);	// L299
        }
      }
    }
  }
  return ;	// L304
}

void forward(
  float v174[8][1024],
  float v175[512],
  float v176[512][1024],
  float v177[512],
  float v178[512][512],
  float v179[256],
  float v180[256][512],
  float v181[10],
  float v182[10][256],
  float v183[8][10]
) {	// L306
	#pragma HLS DATAFLOW
  hls::stream<float> v184[1][1];
	#pragma HLS STREAM variable=v184 depth=80	// L308
  hls::stream<float> v185[1][1];
	#pragma HLS STREAM variable=v185 depth=80	// L309
  hls::stream<float> v186[16][1];
	#pragma HLS STREAM variable=v186 depth=160	// L310
  hls::stream<float> v187[1][16];
	#pragma HLS STREAM variable=v187 depth=128	// L311
  hls::stream<float> v188[1][16];
	#pragma HLS STREAM variable=v188 depth=128	// L312
  hls::stream<float> v189[8][16];
	#pragma HLS STREAM variable=v189 depth=1024	// L313
  hls::stream<float> v190[1][8];
	#pragma HLS STREAM variable=v190 depth=512	// L314
  hls::stream<float> v191[1][8];
	#pragma HLS STREAM variable=v191 depth=512	// L315
  hls::stream<float> v192[8][8];
	#pragma HLS STREAM variable=v192 depth=4096	// L316
  hls::stream<float> v193[1][8];
	#pragma HLS STREAM variable=v193 depth=512	// L317
  hls::stream<float> v194[1][8];
	#pragma HLS STREAM variable=v194 depth=512	// L318
  hls::stream<float> v195[1][8];
	#pragma HLS STREAM variable=v195 depth=512	// L319
  hls::stream<float> v196[1][8];
	#pragma HLS STREAM variable=v196 depth=512	// L320
  hls::stream<float> v197[32][8];
	#pragma HLS STREAM variable=v197 depth=2048	// L321
  node13(v176, v197);	// L322
  node12(v174, v197, v196, 0.000000);	// L323
  node11(v196, v175, v195, v194);	// L324
  node10(v195, v193, 0.000000);	// L325
  node9(v178, v192);	// L326
  node8(v192, v193, v191, 0.000000);	// L327
  node7(v191, v177, v194, v190, 0.000000);	// L328
  node6(v180, v189);	// L329
  node5(v190, v189, v188, 0.000000);	// L330
  node4(v188, v179, v187);	// L331
  node3(v182, v186);	// L332
  node2(v186, v187, v185, 0.000000);	// L333
  node1(v185, v181, v184);	// L334
  node0(v184, v183);	// L335
  return ;	// L336
}

