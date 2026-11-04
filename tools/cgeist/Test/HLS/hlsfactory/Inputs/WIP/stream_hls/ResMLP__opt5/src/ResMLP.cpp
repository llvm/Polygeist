
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
) {	// L23
  loop0: for (int v2 = 0; v2 < 8; v2++) {	// L24
    loop1: for (int v3 = 0; v3 < 10; v3++) {	// L25
      #pragma HLS pipeline II=1
      loop2: for (int v4 = 0; v4 < 1; v4++) {	// L26
        loop3: for (int v5 = 0; v5 < 1; v5++) {	// L27
          float v6 = v0[v4][v5].read();	// L28
          v1[(v2 + v4)][(v3 + v5)] = v6;	// L29
        }
      }
    }
  }
  return ;	// L34
}

void node1(
  hls::stream<float> v7[1][1],
  float v8[10],
  hls::stream<float> v9[1][1]
) {	// L36
  loop4: for (int v10 = 0; v10 < 8; v10++) {	// L37
    loop5: for (int v11 = 0; v11 < 10; v11++) {	// L38
      #pragma HLS pipeline II=1
      loop6: for (int v12 = 0; v12 < 1; v12++) {	// L39
        loop7: for (int v13 = 0; v13 < 1; v13++) {	// L40
          float v14 = v7[v12][v13].read();	// L41
          float v15 = v8[(v11 + v13)];	// L42
          float v16 = v14 + v15;	// L43
          v9[v12][v13].write(v16);	// L44
        }
      }
    }
  }
  return ;	// L49
}

void node2(
  hls::stream<float> v17[1][4],
  hls::stream<float> v18[4][1],
  hls::stream<float> v19[1][1],
  float v20
) {	// L51
  float v21[8][10];	// L52
  float v22[256][10];	// L53
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=4

  float v23[8][256];	// L54
  #pragma HLS array_partition variable=v23 cyclic dim=2 factor=4

  loop8: for (int v24 = 0; v24 < 8; v24++) {	// L55
    loop9: for (int v25 = 0; v25 < 64; v25++) {	// L56
      loop10: for (int v26 = 0; v26 < 10; v26++) {	// L57
        #pragma HLS pipeline II=1
        loop11: for (int v27 = 0; v27 < 1; v27++) {	// L58
          loop12: for (int v28 = 0; v28 < 4; v28++) {	// L59
            loop13: for (int v29 = 0; v29 < 1; v29++) {	// L60
              if ((v26 + v29) == 0) {	// L61
                float v30 = v17[v27][v28].read();	// L62
                v23[(v24 + v27)][(v28 + (v25 * 4))] = v30;	// L63
              }
              float v31 = v23[(v24 + v27)][(v28 + (v25 * 4))];	// L65
              if ((v24 + v27) == 0) {	// L66
                float v32 = v18[v28][v29].read();	// L67
                v22[(v28 + (v25 * 4))][(v26 + v29)] = v32;	// L68
              }
              float v33 = v22[(v28 + (v25 * 4))][(v26 + v29)];	// L70
              if ((v28 + (v25 * 4)) == 0) {	// L71
                v21[(v24 + v27)][(v26 + v29)] = v20;	// L72
              }
              float v34 = v21[(v24 + v27)][(v26 + v29)];	// L74
              float v35 = v31 * v33;	// L75
              float v36 = v34 + v35;	// L76
              v21[(v24 + v27)][(v26 + v29)] = v36;	// L77
              if (((v28 + (v25 * 4)) - 255) == 0) {	// L78
                float v37 = v21[(v24 + v27)][(v26 + v29)];	// L79
                v19[v27][v29].write(v37);	// L80
              }
            }
          }
        }
      }
    }
  }
  return ;	// L88
}

void node3(
  float v38[10][256],
  hls::stream<float> v39[4][1]
) {	// L90
  #pragma HLS array_partition variable=v38 cyclic dim=2 factor=4

  loop14: for (int v40 = 0; v40 < 64; v40++) {	// L91
    loop15: for (int v41 = 0; v41 < 10; v41++) {	// L92
      #pragma HLS pipeline II=1
      loop16: for (int v42 = 0; v42 < 4; v42++) {	// L93
        loop17: for (int v43 = 0; v43 < 1; v43++) {	// L94
          float v44 = v38[(v41 + v43)][(v42 + (v40 * 4))];	// L95
          v39[v42][v43].write(v44);	// L96
        }
      }
    }
  }
  return ;	// L101
}

void node4(
  hls::stream<float> v45[1][4],
  float v46[256],
  hls::stream<float> v47[1][4]
) {	// L103
  #pragma HLS array_partition variable=v46 cyclic dim=1 factor=4

  loop18: for (int v48 = 0; v48 < 8; v48++) {	// L104
    loop19: for (int v49 = 0; v49 < 64; v49++) {	// L105
      #pragma HLS pipeline II=1
      loop20: for (int v50 = 0; v50 < 1; v50++) {	// L106
        loop21: for (int v51 = 0; v51 < 4; v51++) {	// L107
          float v52 = v45[v50][v51].read();	// L108
          float v53 = v46[(v51 + (v49 * 4))];	// L109
          float v54 = v52 + v53;	// L110
          v47[v50][v51].write(v54);	// L111
        }
      }
    }
  }
  return ;	// L116
}

void node5(
  hls::stream<float> v55[16][4],
  hls::stream<float> v56[1][16],
  hls::stream<float> v57[1][4],
  float v58
) {	// L118
  float v59[8][256];	// L119
  #pragma HLS array_partition variable=v59 cyclic dim=2 factor=4

  float v60[512][256];	// L120
  #pragma HLS array_partition variable=v60 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v60 cyclic dim=2 factor=4

  float v61[8][512];	// L121
  #pragma HLS array_partition variable=v61 cyclic dim=2 factor=16

  loop22: for (int v62 = 0; v62 < 8; v62++) {	// L122
    loop23: for (int v63 = 0; v63 < 32; v63++) {	// L123
      loop24: for (int v64 = 0; v64 < 64; v64++) {	// L124
        #pragma HLS pipeline II=1
        loop25: for (int v65 = 0; v65 < 1; v65++) {	// L125
          loop26: for (int v66 = 0; v66 < 16; v66++) {	// L126
            loop27: for (int v67 = 0; v67 < 4; v67++) {	// L127
              if ((v67 + (v64 * 4)) == 0) {	// L128
                float v68 = v56[v65][v66].read();	// L129
                v61[(v62 + v65)][(v66 + (v63 * 16))] = v68;	// L130
              }
              float v69 = v61[(v62 + v65)][(v66 + (v63 * 16))];	// L132
              if ((v62 + v65) == 0) {	// L133
                float v70 = v55[v66][v67].read();	// L134
                v60[(v66 + (v63 * 16))][(v67 + (v64 * 4))] = v70;	// L135
              }
              float v71 = v60[(v66 + (v63 * 16))][(v67 + (v64 * 4))];	// L137
              if ((v66 + (v63 * 16)) == 0) {	// L138
                v59[(v62 + v65)][(v67 + (v64 * 4))] = v58;	// L139
              }
              float v72 = v59[(v62 + v65)][(v67 + (v64 * 4))];	// L141
              float v73 = v69 * v71;	// L142
              float v74 = v72 + v73;	// L143
              v59[(v62 + v65)][(v67 + (v64 * 4))] = v74;	// L144
              if (((v66 + (v63 * 16)) - 511) == 0) {	// L145
                float v75 = v59[(v62 + v65)][(v67 + (v64 * 4))];	// L146
                v57[v65][v67].write(v75);	// L147
              }
            }
          }
        }
      }
    }
  }
  return ;	// L155
}

void node6(
  float v76[256][512],
  hls::stream<float> v77[16][4]
) {	// L157
  #pragma HLS array_partition variable=v76 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v76 cyclic dim=2 factor=16

  loop28: for (int v78 = 0; v78 < 32; v78++) {	// L158
    loop29: for (int v79 = 0; v79 < 64; v79++) {	// L159
      #pragma HLS pipeline II=1
      loop30: for (int v80 = 0; v80 < 16; v80++) {	// L160
        loop31: for (int v81 = 0; v81 < 4; v81++) {	// L161
          float v82 = v76[(v81 + (v79 * 4))][(v80 + (v78 * 16))];	// L162
          v77[v80][v81].write(v82);	// L163
        }
      }
    }
  }
  return ;	// L168
}

void node7(
  hls::stream<float> v83[1][16],
  float v84[512],
  hls::stream<float> v85[1][16],
  hls::stream<float> v86[1][16],
  float v87
) {	// L170
  #pragma HLS array_partition variable=v84 cyclic dim=1 factor=16

  loop32: for (int v88 = 0; v88 < 8; v88++) {	// L171
    loop33: for (int v89 = 0; v89 < 32; v89++) {	// L172
      #pragma HLS pipeline II=1
      loop34: for (int v90 = 0; v90 < 1; v90++) {	// L173
        loop35: for (int v91 = 0; v91 < 16; v91++) {	// L174
          float v92 = v83[v90][v91].read();	// L175
          float v93 = v84[(v91 + (v89 * 16))];	// L176
          float v94 = v85[v90][v91].read();	// L177
          float v95 = v92 + v93;	// L178
          float v96 = v95 + v94;	// L179
          bool v97 = v96 > v87;	// L180
          float v98 = v97 ? (float)v96 : (float)v87;	// L181
          v86[v90][v91].write(v98);	// L182
        }
      }
    }
  }
  return ;	// L187
}

void node8(
  hls::stream<float> v99[16][16],
  hls::stream<float> v100[1][16],
  hls::stream<float> v101[1][16],
  float v102
) {	// L189
  float v103[8][512];	// L190
  #pragma HLS array_partition variable=v103 cyclic dim=2 factor=16

  float v104[512][512];	// L191
  #pragma HLS array_partition variable=v104 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v104 cyclic dim=2 factor=16

  float v105[8][512];	// L192
  #pragma HLS array_partition variable=v105 cyclic dim=2 factor=16

  loop36: for (int v106 = 0; v106 < 8; v106++) {	// L193
    loop37: for (int v107 = 0; v107 < 32; v107++) {	// L194
      loop38: for (int v108 = 0; v108 < 32; v108++) {	// L195
        #pragma HLS pipeline II=1
        loop39: for (int v109 = 0; v109 < 1; v109++) {	// L196
          loop40: for (int v110 = 0; v110 < 16; v110++) {	// L197
            loop41: for (int v111 = 0; v111 < 16; v111++) {	// L198
              if ((v111 + (v108 * 16)) == 0) {	// L199
                float v112 = v100[v109][v110].read();	// L200
                v105[(v106 + v109)][(v110 + (v107 * 16))] = v112;	// L201
              }
              float v113 = v105[(v106 + v109)][(v110 + (v107 * 16))];	// L203
              if ((v106 + v109) == 0) {	// L204
                float v114 = v99[v110][v111].read();	// L205
                v104[(v110 + (v107 * 16))][(v111 + (v108 * 16))] = v114;	// L206
              }
              float v115 = v104[(v110 + (v107 * 16))][(v111 + (v108 * 16))];	// L208
              if ((v110 + (v107 * 16)) == 0) {	// L209
                v103[(v106 + v109)][(v111 + (v108 * 16))] = v102;	// L210
              }
              float v116 = v103[(v106 + v109)][(v111 + (v108 * 16))];	// L212
              float v117 = v113 * v115;	// L213
              float v118 = v116 + v117;	// L214
              v103[(v106 + v109)][(v111 + (v108 * 16))] = v118;	// L215
              if (((v110 + (v107 * 16)) - 511) == 0) {	// L216
                float v119 = v103[(v106 + v109)][(v111 + (v108 * 16))];	// L217
                v101[v109][v111].write(v119);	// L218
              }
            }
          }
        }
      }
    }
  }
  return ;	// L226
}

void node9(
  float v120[512][512],
  hls::stream<float> v121[16][16]
) {	// L228
  #pragma HLS array_partition variable=v120 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v120 cyclic dim=2 factor=16

  loop42: for (int v122 = 0; v122 < 32; v122++) {	// L229
    loop43: for (int v123 = 0; v123 < 32; v123++) {	// L230
      #pragma HLS pipeline II=1
      loop44: for (int v124 = 0; v124 < 16; v124++) {	// L231
        loop45: for (int v125 = 0; v125 < 16; v125++) {	// L232
          float v126 = v120[(v125 + (v123 * 16))][(v124 + (v122 * 16))];	// L233
          v121[v124][v125].write(v126);	// L234
        }
      }
    }
  }
  return ;	// L239
}

void node10(
  hls::stream<float> v127[1][16],
  hls::stream<float> v128[1][16],
  float v129
) {	// L241
  loop46: for (int v130 = 0; v130 < 8; v130++) {	// L242
    loop47: for (int v131 = 0; v131 < 32; v131++) {	// L243
      #pragma HLS pipeline II=1
      loop48: for (int v132 = 0; v132 < 1; v132++) {	// L244
        loop49: for (int v133 = 0; v133 < 16; v133++) {	// L245
          float v134 = v127[v132][v133].read();	// L246
          bool v135 = v134 > v129;	// L247
          float v136 = v135 ? (float)v134 : (float)v129;	// L248
          v128[v132][v133].write(v136);	// L249
        }
      }
    }
  }
  return ;	// L254
}

void node11(
  hls::stream<float> v137[1][16],
  float v138[512],
  hls::stream<float> v139[1][16],
  hls::stream<float> v140[1][16]
) {	// L256
  #pragma HLS array_partition variable=v138 cyclic dim=1 factor=16

  loop50: for (int v141 = 0; v141 < 8; v141++) {	// L257
    loop51: for (int v142 = 0; v142 < 32; v142++) {	// L258
      #pragma HLS pipeline II=1
      loop52: for (int v143 = 0; v143 < 1; v143++) {	// L259
        loop53: for (int v144 = 0; v144 < 16; v144++) {	// L260
          float v145 = v137[v143][v144].read();	// L261
          float v146 = v138[(v144 + (v142 * 16))];	// L262
          float v147 = v145 + v146;	// L263
          v139[v143][v144].write(v147);	// L264
          v140[v143][v144].write(v147);	// L265
        }
      }
    }
  }
  return ;	// L270
}

void node12(
  float v148[8][1024],
  hls::stream<float> v149[8][16],
  hls::stream<float> v150[1][16],
  float v151
) {	// L272
  #pragma HLS array_partition variable=v148 cyclic dim=2 factor=8

  float v152[8][512];	// L273
  #pragma HLS array_partition variable=v152 cyclic dim=2 factor=16

  float v153[1024][512];	// L274
  #pragma HLS array_partition variable=v153 cyclic dim=1 factor=8
  #pragma HLS array_partition variable=v153 cyclic dim=2 factor=16

  loop54: for (int v154 = 0; v154 < 8; v154++) {	// L275
    loop55: for (int v155 = 0; v155 < 128; v155++) {	// L276
      loop56: for (int v156 = 0; v156 < 32; v156++) {	// L277
        #pragma HLS pipeline II=1
        loop57: for (int v157 = 0; v157 < 1; v157++) {	// L278
          loop58: for (int v158 = 0; v158 < 8; v158++) {	// L279
            loop59: for (int v159 = 0; v159 < 16; v159++) {	// L280
              float v160 = v148[(v154 + v157)][(v158 + (v155 * 8))];	// L281
              if ((v154 + v157) == 0) {	// L282
                float v161 = v149[v158][v159].read();	// L283
                v153[(v158 + (v155 * 8))][(v159 + (v156 * 16))] = v161;	// L284
              }
              float v162 = v153[(v158 + (v155 * 8))][(v159 + (v156 * 16))];	// L286
              if ((v158 + (v155 * 8)) == 0) {	// L287
                v152[(v154 + v157)][(v159 + (v156 * 16))] = v151;	// L288
              }
              float v163 = v152[(v154 + v157)][(v159 + (v156 * 16))];	// L290
              float v164 = v160 * v162;	// L291
              float v165 = v163 + v164;	// L292
              v152[(v154 + v157)][(v159 + (v156 * 16))] = v165;	// L293
              if (((v158 + (v155 * 8)) - 1023) == 0) {	// L294
                float v166 = v152[(v154 + v157)][(v159 + (v156 * 16))];	// L295
                v150[v157][v159].write(v166);	// L296
              }
            }
          }
        }
      }
    }
  }
  return ;	// L304
}

void node13(
  float v167[512][1024],
  hls::stream<float> v168[8][16]
) {	// L306
  #pragma HLS array_partition variable=v167 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v167 cyclic dim=2 factor=8

  loop60: for (int v169 = 0; v169 < 128; v169++) {	// L307
    loop61: for (int v170 = 0; v170 < 32; v170++) {	// L308
      #pragma HLS pipeline II=1
      loop62: for (int v171 = 0; v171 < 8; v171++) {	// L309
        loop63: for (int v172 = 0; v172 < 16; v172++) {	// L310
          float v173 = v167[(v172 + (v170 * 16))][(v171 + (v169 * 8))];	// L311
          v168[v171][v172].write(v173);	// L312
        }
      }
    }
  }
  return ;	// L317
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
) {	// L319
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v174 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v175 cyclic dim=1 factor=16

  #pragma HLS array_partition variable=v176 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v176 cyclic dim=2 factor=8

  #pragma HLS array_partition variable=v177 cyclic dim=1 factor=16

  #pragma HLS array_partition variable=v178 cyclic dim=1 factor=16
  #pragma HLS array_partition variable=v178 cyclic dim=2 factor=16

  #pragma HLS array_partition variable=v179 cyclic dim=1 factor=4

  #pragma HLS array_partition variable=v180 cyclic dim=1 factor=4
  #pragma HLS array_partition variable=v180 cyclic dim=2 factor=16

  #pragma HLS array_partition variable=v182 cyclic dim=2 factor=4

  hls::stream<float> v184[1][1];
	#pragma HLS STREAM variable=v184 depth=80	// L321
  hls::stream<float> v185[1][1];
	#pragma HLS STREAM variable=v185 depth=80	// L322
  hls::stream<float> v186[4][1];
	#pragma HLS STREAM variable=v186 depth=640	// L323
  hls::stream<float> v187[1][4];
	#pragma HLS STREAM variable=v187 depth=512	// L324
  hls::stream<float> v188[1][4];
	#pragma HLS STREAM variable=v188 depth=512	// L325
  hls::stream<float> v189[16][4];
	#pragma HLS STREAM variable=v189 depth=2048	// L326
  hls::stream<float> v190[1][16];
	#pragma HLS STREAM variable=v190 depth=256	// L327
  hls::stream<float> v191[1][16];
	#pragma HLS STREAM variable=v191 depth=256	// L328
  hls::stream<float> v192[16][16];
	#pragma HLS STREAM variable=v192 depth=1024	// L329
  hls::stream<float> v193[1][16];
	#pragma HLS STREAM variable=v193 depth=256	// L330
  hls::stream<float> v194[1][16];
	#pragma HLS STREAM variable=v194 depth=256	// L331
  hls::stream<float> v195[1][16];
	#pragma HLS STREAM variable=v195 depth=256	// L332
  hls::stream<float> v196[1][16];
	#pragma HLS STREAM variable=v196 depth=256	// L333
  hls::stream<float> v197[8][16];
	#pragma HLS STREAM variable=v197 depth=4096	// L334
  node13(v176, v197);	// L335
  node12(v174, v197, v196, 0.000000);	// L336
  node11(v196, v175, v195, v194);	// L337
  node10(v195, v193, 0.000000);	// L338
  node9(v178, v192);	// L339
  node8(v192, v193, v191, 0.000000);	// L340
  node7(v191, v177, v194, v190, 0.000000);	// L341
  node6(v180, v189);	// L342
  node5(v189, v190, v188, 0.000000);	// L343
  node4(v188, v179, v187);	// L344
  node3(v182, v186);	// L345
  node2(v187, v186, v185, 0.000000);	// L346
  node1(v185, v181, v184);	// L347
  node0(v184, v183);	// L348
  return ;	// L349
}

