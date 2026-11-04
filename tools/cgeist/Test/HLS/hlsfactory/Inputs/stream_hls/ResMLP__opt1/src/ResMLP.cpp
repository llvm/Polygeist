
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
  float v1[8][10]
) {	// L6
  loop0: for (int v2 = 0; v2 < 8; v2++) {	// L7
    loop1: for (int v3 = 0; v3 < 10; v3++) {	// L8
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v4 = v0.read();	// L9
      v1[v2][v3] = v4;	// L10
    }
  }
  return ;	// L13
}

void node1(
  hls::stream<float> &v5,
  float v6[10],
  hls::stream<float> &v7
) {	// L15
  loop2: for (int v8 = 0; v8 < 8; v8++) {	// L16
    loop3: for (int v9 = 0; v9 < 10; v9++) {	// L17
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v10 = v5.read();	// L18
      float v11 = v6[v9];	// L19
      float v12 = v10 + v11;	// L20
      v7.write(v12);	// L21
    }
  }
  return ;	// L24
}

void node2(
  hls::stream<float> &v13,
  hls::stream<float> &v14,
  hls::stream<float> &v15,
  float v16
) {	// L26
  float v17[8][10];	// L27
  float v18[256][10];	// L28
  float v19[8][256];	// L29
  loop4: for (int v20 = 0; v20 < 8; v20++) {	// L30
    loop5: for (int v21 = 0; v21 < 10; v21++) {	// L31
      loop6: for (int v22 = 0; v22 < 256; v22++) {	// L32
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v21 == 0) {	// L33
          float v23 = v14.read();	// L34
          v19[v20][v22] = v23;	// L35
        }
        float v24 = v19[v20][v22];	// L37
        if (v20 == 0) {	// L38
          float v25 = v13.read();	// L39
          v18[v22][v21] = v25;	// L40
        }
        float v26 = v18[v22][v21];	// L42
        if (v22 == 0) {	// L43
          v17[v20][v21] = v16;	// L44
        }
        float v27 = v17[v20][v21];	// L46
        float v28 = v24 * v26;	// L47
        float v29 = v27 + v28;	// L48
        v17[v20][v21] = v29;	// L49
        if ((v22 - 255) == 0) {	// L50
          float v30 = v17[v20][v21];	// L51
          v15.write(v30);	// L52
        }
      }
    }
  }
  return ;	// L57
}

void node3(
  float v31[10][256],
  hls::stream<float> &v32
) {	// L59
  loop7: for (int v33 = 0; v33 < 10; v33++) {	// L60
    loop8: for (int v34 = 0; v34 < 256; v34++) {	// L61
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v35 = v31[v33][v34];	// L62
      v32.write(v35);	// L63
    }
  }
  return ;	// L66
}

void node4(
  hls::stream<float> &v36,
  float v37[256],
  hls::stream<float> &v38
) {	// L68
  loop9: for (int v39 = 0; v39 < 8; v39++) {	// L69
    loop10: for (int v40 = 0; v40 < 256; v40++) {	// L70
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v41 = v36.read();	// L71
      float v42 = v37[v40];	// L72
      float v43 = v41 + v42;	// L73
      v38.write(v43);	// L74
    }
  }
  return ;	// L77
}

void node5(
  hls::stream<float> &v44,
  hls::stream<float> &v45,
  hls::stream<float> &v46,
  float v47
) {	// L79
  float v48[8][256];	// L80
  float v49[512][256];	// L81
  float v50[8][512];	// L82
  loop11: for (int v51 = 0; v51 < 8; v51++) {	// L83
    loop12: for (int v52 = 0; v52 < 256; v52++) {	// L84
      loop13: for (int v53 = 0; v53 < 512; v53++) {	// L85
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v52 == 0) {	// L86
          float v54 = v44.read();	// L87
          v50[v51][v53] = v54;	// L88
        }
        float v55 = v50[v51][v53];	// L90
        if (v51 == 0) {	// L91
          float v56 = v45.read();	// L92
          v49[v53][v52] = v56;	// L93
        }
        float v57 = v49[v53][v52];	// L95
        if (v53 == 0) {	// L96
          v48[v51][v52] = v47;	// L97
        }
        float v58 = v48[v51][v52];	// L99
        float v59 = v55 * v57;	// L100
        float v60 = v58 + v59;	// L101
        v48[v51][v52] = v60;	// L102
        if ((v53 - 511) == 0) {	// L103
          float v61 = v48[v51][v52];	// L104
          v46.write(v61);	// L105
        }
      }
    }
  }
  return ;	// L110
}

void node6(
  float v62[256][512],
  hls::stream<float> &v63
) {	// L112
  loop14: for (int v64 = 0; v64 < 256; v64++) {	// L113
    loop15: for (int v65 = 0; v65 < 512; v65++) {	// L114
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v66 = v62[v64][v65];	// L115
      v63.write(v66);	// L116
    }
  }
  return ;	// L119
}

void node7(
  hls::stream<float> &v67,
  float v68[512],
  hls::stream<float> &v69,
  hls::stream<float> &v70,
  float v71
) {	// L121
  loop16: for (int v72 = 0; v72 < 8; v72++) {	// L122
    loop17: for (int v73 = 0; v73 < 512; v73++) {	// L123
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v74 = v67.read();	// L124
      float v75 = v68[v73];	// L125
      float v76 = v69.read();	// L126
      float v77 = v74 + v75;	// L127
      float v78 = v77 + v76;	// L128
      bool v79 = v78 > v71;	// L129
      float v80 = v79 ? (float)v78 : (float)v71;	// L130
      v70.write(v80);	// L131
    }
  }
  return ;	// L134
}

void node8(
  hls::stream<float> &v81,
  hls::stream<float> &v82,
  hls::stream<float> &v83,
  float v84
) {	// L136
  float v85[8][512];	// L137
  float v86[512][512];	// L138
  float v87[8][512];	// L139
  loop18: for (int v88 = 0; v88 < 8; v88++) {	// L140
    loop19: for (int v89 = 0; v89 < 512; v89++) {	// L141
      loop20: for (int v90 = 0; v90 < 512; v90++) {	// L142
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        if (v89 == 0) {	// L143
          float v91 = v82.read();	// L144
          v87[v88][v90] = v91;	// L145
        }
        float v92 = v87[v88][v90];	// L147
        if (v88 == 0) {	// L148
          float v93 = v81.read();	// L149
          v86[v90][v89] = v93;	// L150
        }
        float v94 = v86[v90][v89];	// L152
        if (v90 == 0) {	// L153
          v85[v88][v89] = v84;	// L154
        }
        float v95 = v85[v88][v89];	// L156
        float v96 = v92 * v94;	// L157
        float v97 = v95 + v96;	// L158
        v85[v88][v89] = v97;	// L159
        if ((v90 - 511) == 0) {	// L160
          float v98 = v85[v88][v89];	// L161
          v83.write(v98);	// L162
        }
      }
    }
  }
  return ;	// L167
}

void node9(
  float v99[512][512],
  hls::stream<float> &v100
) {	// L169
  loop21: for (int v101 = 0; v101 < 512; v101++) {	// L170
    loop22: for (int v102 = 0; v102 < 512; v102++) {	// L171
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v103 = v99[v101][v102];	// L172
      v100.write(v103);	// L173
    }
  }
  return ;	// L176
}

void node10(
  hls::stream<float> &v104,
  hls::stream<float> &v105,
  float v106
) {	// L178
  loop23: for (int v107 = 0; v107 < 8; v107++) {	// L179
    loop24: for (int v108 = 0; v108 < 512; v108++) {	// L180
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v109 = v104.read();	// L181
      bool v110 = v109 > v106;	// L182
      float v111 = v110 ? (float)v109 : (float)v106;	// L183
      v105.write(v111);	// L184
    }
  }
  return ;	// L187
}

void node11(
  hls::stream<float> &v112,
  float v113[512],
  hls::stream<float> &v114,
  hls::stream<float> &v115
) {	// L189
  loop25: for (int v116 = 0; v116 < 8; v116++) {	// L190
    loop26: for (int v117 = 0; v117 < 512; v117++) {	// L191
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v118 = v112.read();	// L192
      float v119 = v113[v117];	// L193
      float v120 = v118 + v119;	// L194
      v114.write(v120);	// L195
      v115.write(v120);	// L196
    }
  }
  return ;	// L199
}

void node12(
  float v121[8][1024],
  hls::stream<float> &v122,
  hls::stream<float> &v123,
  float v124
) {	// L201
  float v125[8][512];	// L202
  float v126[1024][512];	// L203
  loop27: for (int v127 = 0; v127 < 8; v127++) {	// L204
    loop28: for (int v128 = 0; v128 < 512; v128++) {	// L205
      loop29: for (int v129 = 0; v129 < 1024; v129++) {	// L206
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        float v130 = v121[v127][v129];	// L207
        if (v127 == 0) {	// L208
          float v131 = v122.read();	// L209
          v126[v129][v128] = v131;	// L210
        }
        float v132 = v126[v129][v128];	// L212
        if (v129 == 0) {	// L213
          v125[v127][v128] = v124;	// L214
        }
        float v133 = v125[v127][v128];	// L216
        float v134 = v130 * v132;	// L217
        float v135 = v133 + v134;	// L218
        v125[v127][v128] = v135;	// L219
        if ((v129 - 1023) == 0) {	// L220
          float v136 = v125[v127][v128];	// L221
          v123.write(v136);	// L222
        }
      }
    }
  }
  return ;	// L227
}

void node13(
  float v137[512][1024],
  hls::stream<float> &v138
) {	// L229
  loop30: for (int v139 = 0; v139 < 512; v139++) {	// L230
    loop31: for (int v140 = 0; v140 < 1024; v140++) {	// L231
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      float v141 = v137[v139][v140];	// L232
      v138.write(v141);	// L233
    }
  }
  return ;	// L236
}

void forward(
  float v142[8][1024],
  float v143[512],
  float v144[512][1024],
  float v145[512],
  float v146[512][512],
  float v147[256],
  float v148[256][512],
  float v149[10],
  float v150[10][256],
  float v151[8][10]
) {	// L238
	#pragma HLS DATAFLOW
  hls::stream<float> v152("fifo_0");
	#pragma HLS STREAM variable=v152 depth=80	// L240
  hls::stream<float> v153("fifo_1");
	#pragma HLS STREAM variable=v153 depth=80	// L241
  hls::stream<float> v154("fifo_2");
	#pragma HLS STREAM variable=v154 depth=2560	// L242
  hls::stream<float> v155("fifo_3");
	#pragma HLS STREAM variable=v155 depth=2048	// L243
  hls::stream<float> v156("fifo_4");
	#pragma HLS STREAM variable=v156 depth=2048	// L244
  hls::stream<float> v157("fifo_5");
	#pragma HLS STREAM variable=v157 depth=131072	// L245
  hls::stream<float> v158("fifo_6");
	#pragma HLS STREAM variable=v158 depth=4096	// L246
  hls::stream<float> v159("fifo_7");
	#pragma HLS STREAM variable=v159 depth=4096	// L247
  hls::stream<float> v160("fifo_8");
	#pragma HLS STREAM variable=v160 depth=262144	// L248
  hls::stream<float> v161("fifo_9");
	#pragma HLS STREAM variable=v161 depth=4096	// L249
  hls::stream<float> v162("fifo_10");
	#pragma HLS STREAM variable=v162 depth=4096	// L250
  hls::stream<float> v163("fifo_11");
	#pragma HLS STREAM variable=v163 depth=4096	// L251
  hls::stream<float> v164("fifo_12");
	#pragma HLS STREAM variable=v164 depth=4096	// L252
  hls::stream<float> v165("fifo_13");
	#pragma HLS STREAM variable=v165 depth=524288	// L253
  node13(v144, v165);	// L254
  node12(v142, v165, v164, 0.000000);	// L255
  node11(v164, v143, v163, v162);	// L256
  node10(v163, v161, 0.000000);	// L257
  node9(v146, v160);	// L258
  node8(v160, v161, v159, 0.000000);	// L259
  node7(v159, v145, v162, v158, 0.000000);	// L260
  node6(v148, v157);	// L261
  node5(v158, v157, v156, 0.000000);	// L262
  node4(v156, v147, v155);	// L263
  node3(v150, v154);	// L264
  node2(v154, v155, v153, 0.000000);	// L265
  node1(v153, v149, v152);	// L266
  node0(v152, v151);	// L267
  return ;	// L268
}

